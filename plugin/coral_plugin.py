#!/usr/bin/env python3
"""Minimal Kubernetes device plugin for a Google Coral Edge TPU USB accelerator.

Why a device plugin and not just a hostPath mount:
  a hostPath bind-mount of /dev/bus/usb/NNN/DDD gets the *file* into the
  container, but containerd's cgroup device allowlist still refuses open()
  (EPERM). Only the device plugin can tell the runtime to add the device to
  that allowlist, which is what Allocate() does here.

Exposes: home.com/coral
Pods request it with:
  resources:
    limits:
      home.com/coral: 1
"""
import concurrent.futures
import glob
import logging
import os
import signal
import sys
import time

import grpc
import api_pb2
import api_pb2_grpc

RESOURCE_NAME = "home.com/coral"
API_VERSION = "v1beta1"
ENDPOINT = "coral.sock"
PLUGIN_DIR = os.environ.get("PLUGIN_DIR", "/var/lib/kubelet/device-plugins")
POLL_SECONDS = int(os.environ.get("POLL_SECONDS", "5"))

# USB identities the Edge TPU enumerates under.
CORAL_USB_IDS = {
    ("1a6e", "089a"),  # Global Unichip Corp - boot/flash mode
    ("1a6e", "0892"),  # Global Unichip Corp - runtime
    ("18d1", "9302"),  # Google Inc. - runtime (what libusb talks to)
}

log = logging.getLogger("coral-plugin")


def discover():
    """Return [{id, node, healthy}] for every Coral device visible in sysfs."""
    devices = []
    for idv in sorted(glob.glob("/sys/bus/usb/devices/*/idVendor")):
        try:
            vendor = open(idv).read().strip()
            dev_dir = os.path.dirname(idv)
            product = open(os.path.join(dev_dir, "idProduct")).read().strip()
            if (vendor, product) not in CORAL_USB_IDS:
                continue
            busnum = int(open(os.path.join(dev_dir, "busnum")).read().strip())
            devnum = int(open(os.path.join(dev_dir, "devnum")).read().strip())
            node = f"/dev/bus/usb/{busnum:03d}/{devnum:03d}"
            # Health = "the node exists and matches a Coral USB id".
            # A non-privileged plugin container cannot open() or even access()
            # the node (the cgroup device allowlist denies it), so a real I/O
            # probe is not possible here without making the plugin privileged.
            healthy = os.path.exists(node)
            devices.append({
                "id": os.path.basename(dev_dir),
                "node": node,
                "healthy": healthy,
            })
        except Exception as exc:
            log.warning("scan failed for %s: %s", idv, exc)
    return devices


class CoralPlugin(api_pb2_grpc.DevicePluginServicer):
    def __init__(self):
        self.devices = []

    def GetDevicePluginOptions(self, request, context):
        return api_pb2.DevicePluginOptions(
            pre_start_required=False,
            get_preferred_allocation_available=False,
        )

    def ListAndWatch(self, request, context):
        last = None
        while True:
            devices = discover()
            snapshot = sorted((d["id"], d["node"], d["healthy"]) for d in devices)
            if snapshot != last:
                last = snapshot
                self.devices = devices
                log.info("device list: %s", snapshot)
                yield api_pb2.ListAndWatchResponse(
                    devices=[
                        api_pb2.Device(ID=d["id"], health="Healthy" if d["healthy"] else "Unhealthy")
                        for d in devices
                    ]
                )
            time.sleep(POLL_SECONDS)

    def Allocate(self, request, context):
        by_id = {d["id"]: d for d in self.devices}
        responses = []
        for container in request.container_requests:
            devices = []
            for device_id in container.devices_ids:
                dev = by_id.get(device_id)
                if dev is None:
                    return api_pb2.AllocateResponse(
                        container_responses=[api_pb2.ContainerAllocateResponse()]
                    )
                devices.append(api_pb2.DeviceSpec(
                    container_path=dev["node"],
                    host_path=dev["node"],
                    permissions="rw",
                ))
            responses.append(api_pb2.ContainerAllocateResponse(
                devices=devices,
                envs={"CORAL_DEVICE_NODE": devices[0].host_path if devices else ""},
            ))
        return api_pb2.AllocateResponse(container_responses=responses)

    def GetPreferredAllocation(self, request, context):
        return api_pb2.PreferredAllocationResponse()

    def PreStartContainer(self, request, context):
        return api_pb2.PreStartContainerResponse()


def register():
    channel = grpc.insecure_channel("unix://" + os.path.join(PLUGIN_DIR, "kubelet.sock"))
    stub = api_pb2_grpc.RegistrationStub(channel)
    resp = stub.Register(api_pb2.RegisterRequest(
        version=API_VERSION,
        endpoint=ENDPOINT,
        resource_name=RESOURCE_NAME,
    ))
    log.info("registered %s as %s", RESOURCE_NAME, ENDPOINT)
    return resp


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    device_path = os.path.join(PLUGIN_DIR, ENDPOINT)
    if os.path.exists(device_path):
        os.remove(device_path)

    if not discover():
        log.warning("no Coral device found at startup; plugin will keep watching")

    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=4))
    api_pb2_grpc.add_DevicePluginServicer_to_server(CoralPlugin(), server)
    # grpcio < 1.62 only exposes add_insecure_port, which accepts unix:// addresses.
    server.add_insecure_port("unix://" + device_path)
    server.start()

    # kubelet may not be up yet, and it re-scans the plugin dir on restart, so
    # keep retrying instead of exiting on the first failure.
    for attempt in range(1, 31):
        try:
            register()
            break
        except Exception as exc:
            log.warning("registration failed (attempt %d): %s", attempt, exc)
            time.sleep(5)
    else:
        log.error("could not register with kubelet")
        server.stop(None)
        return 1

    signal.signal(signal.SIGTERM, lambda *_: server.stop(None))
    signal.signal(signal.SIGINT, lambda *_: server.stop(None))
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())

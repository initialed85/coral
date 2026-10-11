# Coral Edge TPU on k3s

Containerised YOLO object detection on the Google Coral USB Accelerator that is
physically attached to the **betta** node.

```
betta (k3s node, amd64)
  └── /dev/bus/usb/002/003   Google Coral, USB id 18d1:9302
        └── device plugin  ->  resource home.com/coral
              └── deployment coral-yolo (FastAPI, non-privileged)
                    └── https://coral.yourserver.org
```

The Kubernetes manifests live in the `k8s/` directory.

## Why a device plugin

A plain `hostPath` mount of the USB char device puts the *file* into the container,
but containerd's cgroup device allowlist still refuses `open()` (`EPERM`). Only a
device plugin can tell the runtime to add the device to that allowlist, which is what
`Allocate()` does in `plugin/coral_plugin.py`.

The plugin exposes `home.com/coral`; pods request it with:

```yaml
resources:
  limits:
    home.com/coral: 1
```

It is one physical device, so only one pod can hold it at a time.

## Notes / gotchas

- **pycoral is not on PyPI.** The `pycoral` package there is the unrelated Allen
  Coral Atlas CLI. Google's pycoral ships as wheels from GitHub releases, and the
  Linux wheels only go up to `cp39` → the image uses `python:3.9-slim-bullseye`.
- **Google's Coral apt repo is gone** (`packages.cloud.google.com/apt/...edg-tpu` is
  404), so `libedgetpu.so.1.0` is vendored in `vendor/`.
- **Debian bullseye is EOL**, so the Dockerfile pulls apt from `archive.debian.org`.
- **pycoral's `_pywrap_coral` is built against NumPy 1.x** → `numpy<2` is pinned.
- **FastAPI needs `python-multipart`** for the multipart `file` upload, otherwise the
  app crashes at import time.
- **The board enumerates under two USB ids** (`1a6e:089a` boot mode, `18d1:9302`
  runtime). Detection matches on either, and the udev rule covers both.
- **Builds run on the x86 Docker daemon** at `tcp://192.168.11.128:2375` (`build.sh`),
  because pycoral wheels are x86_64-only and betta has no Docker.
- **YOLO models are precompiled** from `jveitchmichaelis/edgetpu-yolo`, so no Edge TPU
  compiler is needed.

## Files

| Path | Purpose |
| --- | --- |
| `Dockerfile` | YOLO FastAPI service image (`initialed85/coral-yolo`) |
| `Dockerfile.plugin` | device plugin image (`initialed85/coral-plugin`) |
| `build.sh` | build + push both images on the x86 daemon |
| `plugin/coral_plugin.py` | gRPC device plugin, exposes `home.com/coral` |
| `app/server.py` | FastAPI service |
| `app/validate.py` | hardware + inference smoke test |
| `app/coral_devices.py` | sysfs detection of the Coral (both USB ids) |
| `app/edgetpumodel.py` | model wrapper: quantised input/output, optional NMS |
| `app/nms.py` | non-maximum suppression |
| `app/utils.py` | letterbox resize, box plotting, COCO class map |
| `app/coco.yaml` | 80 COCO class names for YOLO |
| `app/coco_labels.txt` | labels for the MobileNet validation model |
| `host/99-coral.rules` | udev rule (installed on betta at `/etc/udev/rules.d/`) |
| `vendor/` | `libedgetpu.so.1.0`, pycoral + tflite-runtime wheels, `.tflite` models |
| `k8s/` | dev manifests (namespace, daemonset, deployment, validation pod) |
| `upstream/` | original `edgetpu-yolo` sources, for reference |

## Build

```bash
./build.sh          # builds and pushes both images on the x86 daemon
```

## Deploy

```bash
kubectl label nodes betta coral=usb --overwrite
kubectl apply -f ./k8s/
```

Validation (scale the service down first — single device):

```bash
kubectl -n coral scale deploy coral-yolo --replicas=0
kubectl -n coral apply -f k8s/validate-pod.yaml
kubectl -n coral logs pod/coral-validate
kubectl -n coral scale deploy coral-yolo --replicas=1
```

## Usage

```bash
curl -s -F "file=@photo.jpg" https://coral.yourserver.org/detect
curl -s -F "url=https://example.com/photo.jpg" https://coral.yourserver.org/detect
curl -s -F "file=@photo.jpg" -o out.png https://coral.yourserver.org/detect/image
curl -s -F "file=@photo.jpg" -F "conf=0.4" https://coral.yourserver.org/detect
```

`-F` is required (multipart/form-data). Optional form field: `conf`.

## Current state (verified 2026-10-11)

- `betta` advertises `home.com/coral: 1`, node labelled `coral=usb`.
- Non-privileged pod can `open()` the device and run inference.
- `coral-validate`: `OK: Edge TPU is working`, ~7.4 ms per MobileNet invoke (~135 fps).
- `coral-yolo` `/detect` on `grace_hopper.bmp` → `person 0.91`, `tie 0.63`, ~54 ms
  including NMS (~18 fps end-to-end).
- Ingress `coral.yourserver.org` live behind traefik + cert-manager.

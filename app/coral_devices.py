"""Detect Google Coral Edge TPU USB accelerators visible to this container.

pycoral 2.0.0 dropped list_accelerators(), so we scan sysfs + /dev/bus/usb.

The board enumerates under two USB identities depending on whether the Apex
firmware is loaded, and the identity can change after a USB reset:
  1a6e:089a / 1a6e:0892  Global Unichip Corp.
  18d1:9302              Google Inc.  (what libusb actually talks to)

So match on either vendor id rather than a single vendor:product pair.
"""
import glob
import os

CORAL_USB_IDS = {
    ("1a6e", "089a"),
    ("1a6e", "0892"),
    ("18d1", "9302"),
}
CORAL_VENDORS = {v for v, _ in CORAL_USB_IDS}


def coral_devices():
    """Return a list of dicts describing visible Coral USB accelerators."""
    found = []
    for idv in sorted(glob.glob("/sys/bus/usb/devices/*/idVendor")):
        try:
            vendor = open(idv).read().strip()
            if vendor not in CORAL_VENDORS:
                continue
            dev_dir = os.path.dirname(idv)
            product = open(os.path.join(dev_dir, "idProduct")).read().strip()
            if (vendor, product) not in CORAL_USB_IDS:
                continue
            busnum = int(open(os.path.join(dev_dir, "busnum")).read().strip())
            devnum = int(open(os.path.join(dev_dir, "devnum")).read().strip())
            node = f"/dev/bus/usb/{busnum:03d}/{devnum:03d}"
            pfile = os.path.join(dev_dir, "product")
            product_name = open(pfile).read().strip() if os.path.exists(pfile) else ""
            exists = os.path.exists(node)
            found.append({
                "sysfs": os.path.basename(dev_dir),
                "bus": busnum,
                "device": devnum,
                "node": node,
                "id": f"{vendor}:{product}",
                "product": product_name,
                "node_exists": exists,
                "node_readable": os.access(node, os.R_OK) if exists else False,
                "node_writable": os.access(node, os.W_OK) if exists else False,
            })
        except Exception as exc:  # sysfs files can vanish mid-scan
            found.append({"sysfs": os.path.basename(os.path.dirname(idv)), "error": str(exc)})
    return found


def has_coral():
    return bool(coral_devices())

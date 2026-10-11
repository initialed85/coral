#!/usr/bin/env python3
"""Validate that a Google Coral Edge TPU USB accelerator is reachable from inside
this container and can actually run inference.

Exit 0 == TPU works. Exit non-zero == something is wrong (device not passed
through, permissions, missing libedgetpu, ...).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import cv2

from coral_devices import coral_devices
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common, detect

DEFAULT_MODEL = "/app/models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
DEFAULT_IMAGE = "/app/grace_hopper.bmp"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--image", default=DEFAULT_IMAGE)
    p.add_argument("--labels", default="/app/coco_labels.txt")
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--device", default="", help="edgetpu device name ('' = first available)")
    return p.parse_args()


def main():
    args = parse_args()

    devs = coral_devices()
    print("coral devices visible:", json.dumps(devs, indent=1))
    if not devs:
        print("FAIL: no Coral USB accelerator visible in this container")
        return 1

    # The real test is being able to open the node. os.access() is not enough:
    # a bind-mounted device node can look writable but still be refused by the
    # runtime's cgroup device allowlist (that is what the device plugin fixes).
    for d in devs:
        if not d.get("node_exists"):
            print(f"FAIL: {d['node']} is not present in this container")
            return 1
        try:
            fd = os.open(d["node"], os.O_RDWR)
        except OSError as exc:
            print(f"FAIL: cannot open {d['node']}: {exc}")
            return 1
        os.close(fd)
        print(f"open OK: {d['node']}")

    t0 = time.time()
    interpreter = make_interpreter(args.model, device=args.device)
    interpreter.allocate_tensors()
    print(f"interpreter allocated in {time.time() - t0:.2f}s")

    w, h = common.input_size(interpreter)
    labels = [l.strip() for l in open(args.labels)]

    img = cv2.imread(args.image)
    if img is None:
        print(f"FAIL: cannot read {args.image}")
        return 1
    tensor = cv2.resize(img, (w, h)).astype("uint8")

    dt = 0.0
    for _ in range(20):
        common.set_input(interpreter, tensor)
        t_start = time.time()
        interpreter.invoke()
        dt = time.time() - t_start

    print(f"inference: {dt * 1000:.1f} ms -> {1 / dt:.1f} fps")

    # Output tensor order varies between models, so use pycoral's adapter
    # instead of assuming the "classic" count/scores/classes/boxes layout.
    labels = [l.strip() for l in open(args.labels)]
    objects = detect.get_objects(interpreter, score_threshold=args.threshold)
    print(f"detections: {len(objects)}")
    for obj in objects:
        name = labels[obj.id] if obj.id < len(labels) else str(obj.id)
        bbox = [obj.bbox.xmin, obj.bbox.ymin, obj.bbox.xmax, obj.bbox.ymax]
        print(f"  {name:12s} score={obj.score:.3f} box={bbox}")

    if not objects:
        print("WARN: TPU responded but produced no detections")
    print("OK: Edge TPU is working")
    return 0


if __name__ == "__main__":
    sys.exit(main())

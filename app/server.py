#!/usr/bin/env python3
"""YOLOv5 object detection served on a Google Coral Edge TPU.

Endpoints:
  GET  /              model + device info + throughput
  GET  /healthz       liveness/readiness
  POST /detect        multipart 'file' or form 'url' -> JSON detections
  POST /detect/image  multipart 'file' or form 'url' -> annotated PNG
"""
import logging
import os
import time
import urllib.request
from typing import Optional

import numpy as np
import cv2
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from coral_devices import coral_devices
from edgetpumodel import EdgeTPUModel
from utils import plot_one_box, get_image_tensor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("coral-yolo")

MODEL = os.environ.get("CORAL_MODEL", "/app/models/yolov5s-int8-224_edgetpu.tflite")
NAMES = os.environ.get("CORAL_NAMES", "/app/coco.yaml")
CONF = float(os.environ.get("CORAL_CONF", "0.25"))
IOU = float(os.environ.get("CORAL_IOU", "0.45"))
V8 = os.environ.get("CORAL_V8", "0") == "1"
PORT = int(os.environ.get("PORT", "8080"))

app = FastAPI(title="coral-yolo", version="1.0")

model = None
started_at = time.time()
requests = 0
infer_s = 0.0


@app.on_event("startup")
def startup():
    global model
    logger.info("loading model %s", MODEL)
    model = EdgeTPUModel(MODEL, NAMES, conf_thresh=CONF, iou_thresh=IOU, v8=V8)
    logger.info("model ready, input size %s, %d classes", model.get_image_size(), len(model.names))


async def read_raw(file: Optional[UploadFile], url: Optional[str]) -> bytes:
    if file is not None:
        return await file.read()
    if url:
        try:
            return urllib.request.urlopen(url, timeout=15).read()
        except urllib.error.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"url returned {e.code}") from None
        except urllib.error.URLError as e:
            raise HTTPException(status_code=502, detail=f"url fetch failed: {e.reason}") from None
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"url fetch failed: {e}") from None
    raise HTTPException(status_code=400, detail="post 'file' (multipart) or 'url'")


def read_image(raw: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="could not decode image")
    return img


def run(img: np.ndarray) -> np.ndarray:
    """Run YOLO on a BGR image and return an Nx6 array (x1,y1,x2,y2,conf,class).

    Letterbox to the model's square input, then map the normalised boxes back to
    the original frame via the model's own scaling helper.
    """
    full, net, pad = get_image_tensor(img, model.input_size[0])
    det = model.forward(net, with_nms=True)
    if len(det) and len(det[0]):
        det[0][:, :4] = model.get_scaled_coords(det[0][:, :4], full, pad)
        return det[0]
    return np.zeros((0, 6), dtype=np.float32)


def serialize(det):
    out = []
    for box in det:
        out.append({
            "label": model.names[int(box[5])],
            "class_id": int(box[5]),
            "confidence": round(float(box[4]), 4),
            "box": [round(float(v), 1) for v in box[:4]],
        })
    return out


@app.get("/")
def info():
    return {
        "model": MODEL,
        "input_size": [int(x) for x in model.input_size],
        "classes": len(model.names),
        "conf_thresh": model.conf_thresh,
        "iou_thresh": model.iou_thresh,
        "accelerators": coral_devices(),
        "uptime_s": round(time.time() - started_at, 1),
        "requests": requests,
        "avg_inference_ms": round(1000 * infer_s / requests, 2) if requests else None,
    }


@app.get("/healthz")
def healthz():
    if not coral_devices():
        raise HTTPException(status_code=503, detail="no Coral USB accelerator visible")
    return {"ok": True}


@app.post("/detect")
async def detect(file: UploadFile = File(None), url: str = Form(None), conf: float = Form(None)):
    global requests, infer_s
    if conf is not None:
        model.conf_thresh = conf

    raw = await read_raw(file, url)
    img = read_image(raw)
    t0 = time.time()
    det = run(img)
    dt = time.time() - t0
    requests += 1
    infer_s += dt

    return JSONResponse({
        "detections": serialize(det),
        "count": len(det),
        "inference_ms": round(dt * 1000, 2),
        "image_size": [img.shape[1], img.shape[0]],
    })


@app.post("/detect/image")
async def detect_image(file: UploadFile = File(None), url: str = Form(None), conf: float = Form(None)):
    if conf is not None:
        model.conf_thresh = conf
    img = read_image(await read_raw(file, url))
    det = run(img)
    for box in det:
        cls, conf = int(box[5]), float(box[4])
        img = plot_one_box(
            tuple(box[:4]), img,
            color=model.colors(cls, True),
            label=f"{model.names[cls]} {conf:.2f}",
        )
    return Response(cv2.imencode(".png", img)[1].tobytes(), media_type="image/png")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image

from tlc.engine import classify_proba, get_default_model_path, resolve_threshold
from tlc.preprocess import InputType, load_image


class NoTrafficLightFound(Exception):
    pass

def get_default_detector_path() -> Path:
    env_path = os.environ.get("TLC_DETECTOR_MODEL")
    if env_path:
        return Path(env_path)
    return Path(__file__).parent / "models" / "yolov8n.onnx"

def pad_box(
    box: tuple[float, float, float, float],
    img_w: int,
    img_h: int,
    pad_pct: float = 0.15,
) -> tuple[int, int, int, int]:
    xmin, ymin, xmax, ymax = box
    w = xmax - xmin
    h = ymax - ymin
    pad_w = w * pad_pct
    pad_h = h * pad_pct
    x0 = max(0, round(xmin - pad_w))
    y0 = max(0, round(ymin - pad_h))
    x1 = min(img_w, round(xmax + pad_w))
    y1 = min(img_h, round(ymax + pad_h))
    return x0, y0, x1, y1

def detect_traffic_lights(
    img: Image.Image,
    detector_path: str | Path | None = None,
    conf_thresh: float = 0.25,
) -> list[tuple[float, float, float, float]]:
    det_path = Path(detector_path) if detector_path is not None else get_default_detector_path()
    if not det_path.is_file():
        raise FileNotFoundError(f"Traffic light detector model not found at {det_path}")

    session = ort.InferenceSession(str(det_path.resolve()), providers=["CPUExecutionProvider"])
    orig_w, orig_h = img.size
    resized = img.resize((640, 640), resample=Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
    tensor = np.expand_dims(arr, axis=0)

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: tensor})
    preds = outputs[0][0]

    boxes: list[tuple[float, float, float, float]] = []
    if preds.shape[0] > preds.shape[1]:
        preds = preds.T

    x_factor = orig_w / 640.0
    y_factor = orig_h / 640.0

    traffic_light_class_idx = 9
    for i in range(preds.shape[1]):
        col = preds[:, i]
        cls_scores = col[4:]
        if cls_scores.size <= traffic_light_class_idx:
            continue
        score = float(cls_scores[traffic_light_class_idx])
        if score >= conf_thresh:
            xc, yc, w, h = col[0] * x_factor, col[1] * y_factor, col[2] * x_factor, col[3] * y_factor
            xmin = max(0.0, xc - w / 2.0)
            ymin = max(0.0, yc - h / 2.0)
            xmax = min(float(orig_w), xc + w / 2.0)
            ymax = min(float(orig_h), yc + h / 2.0)
            boxes.append((xmin, ymin, xmax, ymax))

    return boxes

def classify_scene_details(
    source: InputType,
    threshold: float | None = None,
    model_path: str | Path | None = None,
    detector_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    img = load_image(source)
    boxes = detect_traffic_lights(img, detector_path=detector_path)
    if not boxes:
        raise NoTrafficLightFound("No traffic light found in the scene image")

    cls_model_path = Path(model_path) if model_path is not None else get_default_model_path()
    th = resolve_threshold(cls_model_path, threshold)

    results: list[dict[str, Any]] = []
    w, h = img.size
    for box in boxes:
        x0, y0, x1, y1 = pad_box(box, w, h, pad_pct=0.15)
        crop = img.crop((x0, y0, x1, y1))
        p_red = classify_proba(crop, model_path=cls_model_path)
        label = "red" if p_red >= th else "not_red"
        results.append({
            "box": [x0, y0, x1, y1],
            "p_red": p_red,
            "label": label,
        })
    return results

def classify_scene_crop(
    source: InputType,
    threshold: float | None = None,
    model_path: str | Path | None = None,
    detector_path: str | Path | None = None,
) -> str:
    details = classify_scene_details(
        source,
        threshold=threshold,
        model_path=model_path,
        detector_path=detector_path,
    )
    if any(d["label"] == "red" for d in details):
        return "red"
    return "not_red"

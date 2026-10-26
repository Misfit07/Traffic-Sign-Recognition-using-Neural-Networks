from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from tlc.preprocess import InputType, load_image

_SIGN_SESSION_LOCK = threading.Lock()
_SIGN_SESSION: ort.InferenceSession | None = None
_SIGN_LABELS_CACHE: dict[int, str] | None = None


def get_default_sign_model_path() -> Path:
    env_path = os.environ.get("TLC_SIGN_MODEL")
    if env_path:
        return Path(env_path)
    return Path(__file__).parent / "models" / "traffic_sign.onnx"


def get_default_sign_labels_path() -> Path:
    return Path(__file__).parent / "models" / "sign_labels.json"


def load_sign_labels() -> dict[int, str]:
    global _SIGN_LABELS_CACHE
    if _SIGN_LABELS_CACHE is not None:
        return _SIGN_LABELS_CACHE

    labels_path = get_default_sign_labels_path()
    if labels_path.is_file():
        with open(labels_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            _SIGN_LABELS_CACHE = {int(k): str(v) for k, v in data.items()}
            return _SIGN_LABELS_CACHE

    _SIGN_LABELS_CACHE = {i: f"Class {i}" for i in range(43)}
    return _SIGN_LABELS_CACHE


def get_sign_session(model_path: str | Path | None = None) -> ort.InferenceSession:
    global _SIGN_SESSION
    if _SIGN_SESSION is not None and model_path is None:
        return _SIGN_SESSION

    with _SIGN_SESSION_LOCK:
        if _SIGN_SESSION is not None and model_path is None:
            return _SIGN_SESSION

        path = Path(model_path) if model_path is not None else get_default_sign_model_path()
        if not path.is_file():
            raise FileNotFoundError(f"Traffic sign model not found at {path}")

        session = ort.InferenceSession(str(path.resolve()), providers=["CPUExecutionProvider"])
        if model_path is None:
            _SIGN_SESSION = session
        return session


def apply_clahe(img: Image.Image) -> Image.Image:
    arr = np.array(img.convert("RGB"))
    img_lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])
    rgb_arr = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(rgb_arr)


def preprocess_sign(source: InputType) -> tuple[np.ndarray, Image.Image]:
    img = load_image(source)
    clahe_img = apply_clahe(img)
    resized = clahe_img.resize((32, 32), Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0

    mean = np.array([0.3337, 0.3064, 0.3171], dtype=np.float32)
    std = np.array([0.2672, 0.2564, 0.2629], dtype=np.float32)
    norm = (arr - mean) / std
    chw = norm.transpose(2, 0, 1)
    tensor = np.expand_dims(chw, axis=0)
    return tensor, clahe_img


def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def classify_sign(
    source: InputType,
    confidence_threshold: float = 75.0,
    model_path: str | Path | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    tensor, _ = preprocess_sign(source)
    session = get_sign_session(model_path)

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: tensor})
    logits = outputs[0]
    probs = softmax(logits).ravel()

    top_indices = np.argsort(probs)[-3:][::-1]
    labels_map = load_sign_labels()

    top_class_id = int(top_indices[0])
    top_prob = float(probs[top_class_id]) * 100.0
    is_confident = bool(top_prob >= confidence_threshold)

    top3 = []
    for idx in top_indices:
        cid = int(idx)
        top3.append({
            "class_id": cid,
            "label": labels_map.get(cid, f"Class {cid}"),
            "confidence": round(float(probs[cid]) * 100.0, 2),
        })

    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000.0

    return {
        "class_id": top_class_id,
        "label": labels_map.get(top_class_id, f"Class {top_class_id}"),
        "confidence": round(top_prob, 2),
        "is_confident": is_confident,
        "top3": top3,
        "latency_ms": round(latency_ms, 2),
    }

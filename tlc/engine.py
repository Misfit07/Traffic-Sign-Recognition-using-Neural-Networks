from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from tlc.preprocess import InputType, preprocess_batch, preprocess_single

logger = logging.getLogger("tlc")

_SESSION_LOCK = threading.Lock()
_SESSIONS: dict[str, ort.InferenceSession] = {}
_METADATA_CACHE: dict[str, dict[str, Any]] = {}

def get_default_model_path() -> Path:
    env_path = os.environ.get("TLC_MODEL")
    if env_path:
        return Path(env_path)
    return Path(__file__).parent / "models" / "traffic_light.onnx"

def get_metadata_path(model_path: Path) -> Path:
    sibling = model_path.parent / "labels.json"
    if sibling.is_file():
        return sibling
    fallback = Path(__file__).parent / "models" / "labels.json"
    return fallback

def load_metadata(model_path: Path) -> dict[str, Any]:
    key = str(model_path.resolve())
    if key in _METADATA_CACHE:
        return _METADATA_CACHE[key]
    meta_path = get_metadata_path(model_path)
    if meta_path.is_file():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                _METADATA_CACHE[key] = data
                return data
        except (OSError, json.JSONDecodeError) as e:
            logger.debug("Failed to read metadata from %s: %s", meta_path, e)
    default_meta = {"labels": ["not_red", "red"], "threshold": 0.5}
    _METADATA_CACHE[key] = default_meta
    return default_meta

def get_session(model_path: str | Path | None = None) -> ort.InferenceSession:
    p = Path(model_path) if model_path is not None else get_default_model_path()
    resolved = str(p.resolve())
    if resolved in _SESSIONS:
        return _SESSIONS[resolved]

    with _SESSION_LOCK:
        if resolved in _SESSIONS:
            return _SESSIONS[resolved]

        if not p.is_file():
            raise FileNotFoundError(f"Model file not found: {p}")

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        threads_str = os.environ.get("TLC_THREADS")
        if threads_str and threads_str.isdigit():
            opts.intra_op_num_threads = int(threads_str)

        available = ort.get_available_providers()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in available
            else ["CPUExecutionProvider"]
        )

        logger.debug("Initializing ONNX session from %s with providers %s", resolved, providers)
        session = ort.InferenceSession(resolved, sess_options=opts, providers=providers)
        _SESSIONS[resolved] = session
        return session

def resolve_threshold(
    model_path: Path,
    override_threshold: float | None = None,
) -> float:
    if override_threshold is not None:
        return float(override_threshold)
    env_th = os.environ.get("TLC_THRESHOLD")
    if env_th:
        try:
            return float(env_th)
        except ValueError:
            logger.debug("Invalid TLC_THRESHOLD value '%s', ignoring", env_th)
    meta = load_metadata(model_path)
    return float(meta.get("threshold", 0.5))

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def classify_proba(
    source: InputType,
    model_path: str | Path | None = None,
) -> float:
    resolved_path = Path(model_path) if model_path is not None else get_default_model_path()
    session = get_session(resolved_path)
    tensor = preprocess_single(source)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: tensor})
    logits = outputs[0]
    p_red = float(sigmoid(logits).ravel()[0])
    return p_red

def classify(
    source: InputType,
    threshold: float | None = None,
    model_path: str | Path | None = None,
    scene: bool = False,
) -> str:
    if scene:
        from tlc.detect import classify_scene_crop
        return classify_scene_crop(source, threshold=threshold, model_path=model_path)

    resolved_path = Path(model_path) if model_path is not None else get_default_model_path()
    th = resolve_threshold(resolved_path, threshold)
    p_red = classify_proba(source, model_path=resolved_path)
    meta = load_metadata(resolved_path)
    labels = meta.get("labels", ["not_red", "red"])
    return labels[1] if p_red >= th else labels[0]

def is_red(
    source: InputType,
    threshold: float | None = None,
    model_path: str | Path | None = None,
) -> bool:
    return classify(source, threshold=threshold, model_path=model_path) == "red"

def classify_batch(
    sources: list[Any],
    threshold: float | None = None,
    model_path: str | Path | None = None,
) -> list[str]:
    if not sources:
        return []
    resolved_path = Path(model_path) if model_path is not None else get_default_model_path()
    th = resolve_threshold(resolved_path, threshold)
    session = get_session(resolved_path)
    meta = load_metadata(resolved_path)
    labels = meta.get("labels", ["not_red", "red"])

    batch_tensor = preprocess_batch(sources)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: batch_tensor})
    logits = outputs[0]
    probs = sigmoid(logits).ravel()

    return [labels[1] if p >= th else labels[0] for p in probs]

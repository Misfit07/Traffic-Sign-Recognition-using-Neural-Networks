from __future__ import annotations

from PIL import Image

from tlc.sign_engine import (
    apply_clahe,
    classify_sign,
    get_default_sign_model_path,
    load_sign_labels,
)

SIGN_MODEL_EXISTS = get_default_sign_model_path().is_file()


def test_sign_labels_loaded() -> None:
    labels = load_sign_labels()
    assert len(labels) == 43
    assert labels[14] == "Stop"
    assert labels[13] == "Yield"
    assert "Speed limit" in labels[1]


def test_apply_clahe_output_dimensions() -> None:
    img = Image.new("RGB", (64, 64), (100, 150, 200))
    equalized = apply_clahe(img)
    assert equalized.size == (64, 64)
    assert equalized.mode == "RGB"


def test_classify_sign_inference() -> None:
    if not SIGN_MODEL_EXISTS:
        return
    img = Image.new("RGB", (48, 48), (220, 20, 20))
    res = classify_sign(img, confidence_threshold=70.0)
    assert "class_id" in res
    assert "label" in res
    assert "confidence" in res
    assert "top3" in res
    assert len(res["top3"]) == 3
    assert isinstance(res["is_confident"], bool)
    assert res["latency_ms"] >= 0.0

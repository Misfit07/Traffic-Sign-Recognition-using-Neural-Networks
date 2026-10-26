from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tlc.engine import (
    classify,
    classify_batch,
    classify_proba,
    get_default_model_path,
    is_red,
)
from tlc.preprocess import load_image

MODEL_EXISTS = get_default_model_path().is_file()

def test_missing_model_raises_file_not_found(tmp_path: Path) -> None:
    non_existent = tmp_path / "non_existent.onnx"
    img = Image.new("RGB", (64, 64), (255, 0, 0))
    with pytest.raises(FileNotFoundError):
        classify(img, model_path=non_existent)

def test_invalid_input_types_raise_value_error() -> None:
    with pytest.raises(FileNotFoundError):
        load_image("does_not_exist_image.jpg")
    with pytest.raises(ValueError):
        load_image(b"")
    with pytest.raises(ValueError):
        load_image(np.zeros((10, 10), dtype=np.uint8))

@pytest.mark.skipif(not MODEL_EXISTS, reason="Trained model not yet exported to tlc/models/")
def test_input_types_parity(tmp_path: Path) -> None:
    img = Image.new("RGB", (80, 100), (255, 10, 10))
    img_path = tmp_path / "test_sample.jpg"
    img.save(img_path, format="JPEG")

    raw_bytes = img_path.read_bytes()
    arr = np.asarray(img, dtype=np.uint8)

    out_path = classify(img_path)
    out_pil = classify(img)
    out_arr = classify(arr)
    out_bytes = classify(raw_bytes)

    assert out_path in ("red", "not_red")
    assert out_path == out_pil == out_arr == out_bytes

@pytest.mark.skipif(not MODEL_EXISTS, reason="Trained model not yet exported to tlc/models/")
def test_classify_batch_matches_single() -> None:
    img1 = Image.new("RGB", (50, 50), (255, 0, 0))
    img2 = Image.new("RGB", (50, 50), (0, 255, 0))
    single1 = classify(img1)
    single2 = classify(img2)
    batch_res = classify_batch([img1, img2])
    assert batch_res == [single1, single2]

@pytest.mark.skipif(not MODEL_EXISTS, reason="Trained model not yet exported to tlc/models/")
def test_is_red_and_proba() -> None:
    img = Image.new("RGB", (64, 64), (100, 100, 100))
    proba = classify_proba(img)
    assert 0.0 <= proba <= 1.0
    label = classify(img)
    assert is_red(img) == (label == "red")

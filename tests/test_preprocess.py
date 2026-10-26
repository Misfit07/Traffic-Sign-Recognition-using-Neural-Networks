from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from tlc.preprocess import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    letterbox_image,
    load_image,
    preprocess_batch,
    preprocess_single,
)


def test_load_image_pil() -> None:
    img = Image.new("RGBA", (100, 200), (255, 0, 0, 128))
    loaded = load_image(img)
    assert loaded.mode == "RGB"
    assert loaded.size == (100, 200)

def test_load_image_numpy() -> None:
    arr = np.zeros((50, 60, 3), dtype=np.uint8)
    arr[:, :, 0] = 255
    loaded = load_image(arr)
    assert loaded.mode == "RGB"
    assert loaded.size == (60, 50)

def test_load_image_bytes() -> None:
    img = Image.new("RGB", (30, 40), (0, 255, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    loaded = load_image(buf.getvalue())
    assert loaded.mode == "RGB"
    assert loaded.size == (30, 40)

def test_load_image_modes() -> None:
    for mode in ["L", "P", "CMYK"]:
        img = Image.new(mode, (40, 40), 128 if mode == "L" else (0, 0, 0, 0) if mode == "CMYK" else 0)
        loaded = load_image(img)
        assert loaded.mode == "RGB"

def test_load_image_invalid_inputs(tmp_path: pytest.TempPathFactory) -> None:
    with pytest.raises(FileNotFoundError):
        load_image("non_existent_file_xyz_123.jpg")

    with pytest.raises(ValueError):
        load_image(b"")

    with pytest.raises(ValueError):
        load_image(np.zeros((50, 50), dtype=np.uint8))

    with pytest.raises(ValueError):
        load_image(np.zeros((50, 50, 3), dtype=np.float32))

def test_letterbox_tall_aspect_ratio() -> None:
    tall_img = Image.new("RGB", (50, 100), (255, 255, 255))
    canvas = letterbox_image(tall_img, target_size=(128, 128))
    assert canvas.size == (128, 128)
    arr = np.asarray(canvas)
    assert np.all(arr[:, 0, :] == 0)
    assert np.all(arr[:, -1, :] == 0)
    assert np.any(arr[:, 64, :] == 255)

def test_letterbox_wide_aspect_ratio() -> None:
    wide_img = Image.new("RGB", (100, 50), (255, 255, 255))
    canvas = letterbox_image(wide_img, target_size=(128, 128))
    assert canvas.size == (128, 128)
    arr = np.asarray(canvas)
    assert np.all(arr[0, :, :] == 0)
    assert np.all(arr[-1, :, :] == 0)
    assert np.any(arr[64, :, :] == 255)

def test_preprocess_single_shape_dtype_and_determinism() -> None:
    img = Image.new("RGB", (80, 120), (200, 50, 50))
    tensor1 = preprocess_single(img, target_size=(128, 128))
    tensor2 = preprocess_single(img, target_size=(128, 128))
    assert tensor1.shape == (1, 3, 128, 128)
    assert tensor1.dtype == np.float32
    np.testing.assert_array_almost_equal(tensor1, tensor2)

def test_preprocess_normalization_values() -> None:
    black_img = Image.new("RGB", (128, 128), (0, 0, 0))
    tensor = preprocess_single(black_img, target_size=(128, 128))
    expected_c0 = (0.0 - IMAGENET_MEAN[0, 0, 0]) / IMAGENET_STD[0, 0, 0]
    expected_c1 = (0.0 - IMAGENET_MEAN[1, 0, 0]) / IMAGENET_STD[1, 0, 0]
    expected_c2 = (0.0 - IMAGENET_MEAN[2, 0, 0]) / IMAGENET_STD[2, 0, 0]
    assert np.allclose(tensor[0, 0], expected_c0, atol=1e-4)
    assert np.allclose(tensor[0, 1], expected_c1, atol=1e-4)
    assert np.allclose(tensor[0, 2], expected_c2, atol=1e-4)

def test_preprocess_batch_parity() -> None:
    img1 = Image.new("RGB", (64, 64), (10, 20, 30))
    img2 = Image.new("RGB", (32, 96), (200, 100, 50))
    batch = preprocess_batch([img1, img2], target_size=(128, 128))
    s1 = preprocess_single(img1, target_size=(128, 128))
    s2 = preprocess_single(img2, target_size=(128, 128))
    assert batch.shape == (2, 3, 128, 128)
    np.testing.assert_array_almost_equal(batch[0:1], s1)
    np.testing.assert_array_almost_equal(batch[1:2], s2)

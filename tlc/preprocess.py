from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

InputType = str | Path | Image.Image | np.ndarray | bytes

def load_image(source: InputType) -> Image.Image:
    if isinstance(source, (str, Path)):
        p = Path(source)
        if not p.is_file():
            raise FileNotFoundError(f"Image file does not exist: {p}")
        try:
            with Image.open(p) as img:
                img.load()
                transposed = ImageOps.exif_transpose(img)
                return transposed.convert("RGB") if transposed else img.convert("RGB")
        except FileNotFoundError:
            raise
        except Exception as e:
            raise ValueError(f"Failed to decode image from path: {p}") from e

    if isinstance(source, bytes):
        if not source:
            raise ValueError("Empty image bytes provided")
        try:
            with Image.open(io.BytesIO(source)) as img:
                img.load()
                transposed = ImageOps.exif_transpose(img)
                return transposed.convert("RGB") if transposed else img.convert("RGB")
        except Exception as e:
            raise ValueError("Failed to decode image from bytes") from e

    if isinstance(source, Image.Image):
        try:
            transposed = ImageOps.exif_transpose(source)
            return transposed.convert("RGB") if transposed else source.convert("RGB")
        except Exception as e:
            raise ValueError("Failed to process PIL Image") from e

    if isinstance(source, np.ndarray):
        if source.ndim != 3 or source.shape[2] != 3:
            raise ValueError(f"Expected numpy array of shape (H, W, 3), got shape {source.shape}")
        if source.dtype != np.uint8:
            raise ValueError(f"Expected uint8 numpy array, got dtype {source.dtype}")
        try:
            return Image.fromarray(source, mode="RGB")
        except Exception as e:
            raise ValueError("Failed to create Image from numpy array") from e

    raise ValueError(f"Unsupported input type: {type(source)}")

def letterbox_image(
    img: Image.Image,
    target_size: tuple[int, int] = (128, 128),
) -> Image.Image:
    target_w, target_h = target_size
    orig_w, orig_h = img.size
    if orig_w <= 0 or orig_h <= 0:
        raise ValueError(f"Invalid image dimensions: {orig_w}x{orig_h}")

    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = max(1, round(orig_w * scale))
    new_h = max(1, round(orig_h * scale))

    if orig_w > target_w * 2 and orig_h > target_h * 2:
        factor_w = orig_w // target_w
        factor_h = orig_h // target_h
        factor = min(factor_w, factor_h)
        if factor > 1:
            img = img.reduce(factor)

    resized = img.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    pad_left = (target_w - new_w) // 2
    pad_top = (target_h - new_h) // 2
    canvas.paste(resized, (pad_left, pad_top))
    return canvas

def preprocess_single(
    source: InputType,
    target_size: tuple[int, int] = (128, 128),
) -> np.ndarray:
    pil_img = load_image(source)
    letterboxed = letterbox_image(pil_img, target_size)
    arr = np.asarray(letterboxed, dtype=np.float32)
    arr = arr.transpose(2, 0, 1) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return np.expand_dims(arr, axis=0).astype(np.float32)

def preprocess_batch(
    sources: list[Any],
    target_size: tuple[int, int] = (128, 128),
) -> np.ndarray:
    if not sources:
        return np.empty((0, 3, target_size[1], target_size[0]), dtype=np.float32)
    batch = np.empty((len(sources), 3, target_size[1], target_size[0]), dtype=np.float32)
    for idx, src in enumerate(sources):
        pil_img = load_image(src)
        letterboxed = letterbox_image(pil_img, target_size)
        arr = np.asarray(letterboxed, dtype=np.float32)
        arr = arr.transpose(2, 0, 1) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        batch[idx] = arr
    return batch

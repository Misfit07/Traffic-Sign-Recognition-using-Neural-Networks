from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


def clean_mislabeled_samples() -> None:
    green_dir = Path("data/raw/not_red/green")
    red_dir = Path("data/raw/red")

    for f in green_dir.glob("*.*"):
        with Image.open(f) as im:
            arr = np.array(im.convert("RGB"), dtype=np.float32)
        bright_mask = (arr[:, :, 0] > 120) | (arr[:, :, 1] > 120) | (arr[:, :, 2] > 120)
        if np.any(bright_mask):
            bp = arr[bright_mask]
            br_r, br_g = bp[:, 0].mean(), bp[:, 1].mean()
            if br_r > br_g * 1.5 and br_r > 150:
                dest = red_dir / f.name
                shutil.move(str(f), str(dest))

    for f in list(red_dir.glob("*.*")):
        with Image.open(f) as im:
            arr = np.array(im.convert("RGB"), dtype=np.float32)
        bright_mask = (arr[:, :, 0] > 100) | (arr[:, :, 1] > 100) | (arr[:, :, 2] > 100)
        if np.any(bright_mask):
            bp = arr[bright_mask]
            br_r, br_g, br_b = bp[:, 0].mean(), bp[:, 1].mean(), bp[:, 2].mean()
            if (br_g > br_r * 1.3 or br_b > br_r * 1.3) and br_r < 100:
                f.unlink(missing_ok=True)
        else:
            if arr.mean() < 25:
                f.unlink(missing_ok=True)


def get_boxes(xml_path: Path) -> list[tuple[float, float, float, float]]:
    tree = ET.parse(xml_path)
    boxes: list[tuple[float, float, float, float]] = []
    for obj in tree.getroot().findall("object"):
        b = obj.find("bndbox")
        if b is not None:
            boxes.append((
                float(b.findtext("xmin", "0")),
                float(b.findtext("ymin", "0")),
                float(b.findtext("xmax", "0")),
                float(b.findtext("ymax", "0")),
            ))
    return boxes


def overlaps_any(
    crop_box: tuple[float, float, float, float],
    boxes: list[tuple[float, float, float, float]],
    margin: float = 60.0,
) -> bool:
    cx0, cy0, cx1, cy1 = crop_box
    for bx0, by0, bx1, by1 in boxes:
        if not (
            cx1 < bx0 - margin
            or cx0 > bx1 + margin
            or cy1 < by0 - margin
            or cy0 > by1 + margin
        ):
            return True
    return False


def extract_hard_negatives(target_count: int = 350) -> None:
    ann_dir = Path("data/extracted/s2tld/s2tld_dataset/Annotations")
    img_dir = Path("data/extracted/s2tld/s2tld_dataset/JPEGImages")
    out_dir = Path("data/raw/not_red/hard_negatives")
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    xml_files = sorted(ann_dir.glob("*.xml"))
    rng.shuffle(xml_files)

    saved = 0
    for xml_path in xml_files:
        if saved >= target_count:
            break
        img_path = img_dir / f"{xml_path.stem}.jpg"
        if not img_path.exists():
            continue

        try:
            with Image.open(img_path) as im:
                im_rgb = im.convert("RGB")
                w, h = im_rgb.size
                boxes = get_boxes(xml_path)

                for attempt in range(8):
                    if saved >= target_count:
                        break
                    cw = rng.integers(100, 260)
                    ch = rng.integers(100, 260)
                    if cw >= w or ch >= h:
                        continue
                    x0 = rng.integers(0, w - cw)
                    y0 = rng.integers(0, h - ch)
                    box = (float(x0), float(y0), float(x0 + cw), float(y0 + ch))
                    if overlaps_any(box, boxes):
                        continue

                    crop = im_rgb.crop((x0, y0, x0 + cw, y0 + ch))
                    arr = np.array(crop, dtype=np.float32)
                    std_dev = arr.std()
                    if std_dev < 15:
                        continue

                    out_name = f"hard_neg_{xml_path.stem}_{attempt:02d}.jpg"
                    crop.save(out_dir / out_name, quality=95)
                    saved += 1
        except (UnidentifiedImageError, OSError):
            pass


def main() -> None:
    clean_mislabeled_samples()
    extract_hard_negatives(target_count=350)


if __name__ == "__main__":
    main()

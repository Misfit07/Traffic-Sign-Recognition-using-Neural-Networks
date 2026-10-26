from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image


def parse_class_map(map_str: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not map_str.strip():
        return mapping
    for pair in map_str.split(","):
        if ":" in pair:
            src, dst = pair.split(":", 1)
            mapping[src.strip()] = dst.strip()
    return mapping

def pad_box(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    img_w: int,
    img_h: int,
    pad_pct: float,
) -> tuple[int, int, int, int]:
    w = xmax - xmin
    h = ymax - ymin
    pad_w = w * pad_pct
    pad_h = h * pad_pct
    x0 = max(0, round(xmin - pad_w))
    y0 = max(0, round(ymin - pad_h))
    x1 = min(img_w, round(xmax + pad_w))
    y1 = min(img_h, round(ymax + pad_h))
    return x0, y0, x1, y1

def parse_voc_boxes(xml_path: Path) -> list[tuple[str, tuple[float, float, float, float]]]:
    boxes: list[tuple[str, tuple[float, float, float, float]]] = []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for obj in root.findall("object"):
        name_elem = obj.find("name")
        if name_elem is None or not name_elem.text:
            continue
        label = name_elem.text.strip()
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        xmin = float(bndbox.findtext("xmin", "0"))
        ymin = float(bndbox.findtext("ymin", "0"))
        xmax = float(bndbox.findtext("xmax", "0"))
        ymax = float(bndbox.findtext("ymax", "0"))
        boxes.append((label, (xmin, ymin, xmax, ymax)))
    return boxes

def parse_yolo_boxes(
    txt_path: Path,
    img_w: int,
    img_h: int,
) -> list[tuple[str, tuple[float, float, float, float]]]:
    boxes: list[tuple[str, tuple[float, float, float, float]]] = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = parts[0]
            x_c = float(parts[1]) * img_w
            y_c = float(parts[2]) * img_h
            w = float(parts[3]) * img_w
            h = float(parts[4]) * img_h
            xmin = x_c - w / 2.0
            ymin = y_c - h / 2.0
            xmax = x_c + w / 2.0
            ymax = y_c + h / 2.0
            boxes.append((cls_id, (xmin, ymin, xmax, ymax)))
    return boxes

def crop_and_save(
    img_path: Path,
    boxes: list[tuple[str, tuple[float, float, float, float]]],
    output_dir: Path,
    class_map: dict[str, str],
    pad_pct: float,
) -> int:
    with Image.open(img_path) as img:
        img_rgb = img.convert("RGB")
        w, h = img_rgb.size
        saved_count = 0
        for idx, (label, (xmin, ymin, xmax, ymax)) in enumerate(boxes):
            target_class = class_map.get(label)
            if not target_class:
                continue
            x0, y0, x1, y1 = pad_box(xmin, ymin, xmax, ymax, w, h, pad_pct)
            if x1 <= x0 or y1 <= y0:
                continue
            crop = img_rgb.crop((x0, y0, x1, y1))
            dest_dir = output_dir / target_class
            dest_dir.mkdir(parents=True, exist_ok=True)
            out_filename = f"{img_path.stem}_crop_{idx:03d}.jpg"
            crop.save(dest_dir / out_filename, quality=95)
            saved_count += 1
    return saved_count

def process_crops(
    images_dir: Path,
    labels_dir: Path,
    fmt: str,
    output_dir: Path,
    class_map: dict[str, str],
    pad_pct: float,
) -> int:
    total_saved = 0
    img_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    for img_path in sorted(images_dir.glob("*.*")):
        if img_path.suffix.lower() not in img_extensions:
            continue
        if fmt == "voc":
            ann_path = labels_dir / f"{img_path.stem}.xml"
            if not ann_path.exists():
                continue
            boxes = parse_voc_boxes(ann_path)
        else:
            ann_path = labels_dir / f"{img_path.stem}.txt"
            if not ann_path.exists():
                continue
            with Image.open(img_path) as im:
                w, h = im.size
            boxes = parse_yolo_boxes(ann_path, w, h)
        total_saved += crop_and_save(img_path, boxes, output_dir, class_map, pad_pct)
    return total_saved

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--format", choices=["yolo", "voc"], default="yolo")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--pad-pct", type=float, default=0.15)
    parser.add_argument(
        "--class-map",
        type=str,
        default="0:red,1:not_red,2:not_red,3:not_red,Red:red,RedLeft:red,RedRight:red,Yellow:not_red,Green:not_red,off:not_red",
    )
    args = parser.parse_args()
    cmap = parse_class_map(args.class_map)
    total = process_crops(
        args.images_dir,
        args.labels_dir,
        args.format,
        args.output_dir,
        cmap,
        args.pad_pct,
    )
    print(f"Extracted {total} crops into {args.output_dir}")

if __name__ == "__main__":
    main()

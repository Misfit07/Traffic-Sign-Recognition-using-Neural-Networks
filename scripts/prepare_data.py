from __future__ import annotations

import argparse
import csv
import hashlib
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def extract_group_prefix(filename: str) -> str:
    stem = Path(filename).stem
    if "--" in stem:
        return stem.split("--")[0]
    if "_" in stem:
        parts = stem.rsplit("_", 1)
        if parts[1].isdigit():
            return parts[0]
    return stem

def validate_and_collect_images(
    data_dir: Path,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    records: list[dict[str, str]] = []
    seen_hashes: set[str] = set()
    subsource_counts: dict[str, int] = defaultdict(int)
    corrupt_count = 0
    duplicate_count = 0

    for cls_name in ["red", "not_red"]:
        cls_dir = data_dir / cls_name
        if not cls_dir.is_dir():
            continue

        for path in sorted(cls_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in VALID_EXTENSIONS:
                continue

            try:
                with Image.open(path) as img:
                    img.verify()
                with Image.open(path) as img:
                    img.load()
            except (OSError, SyntaxError, Image.DecompressionBombError, ValueError):
                corrupt_count += 1
                continue

            file_hash = compute_sha256(path)
            if file_hash in seen_hashes:
                duplicate_count += 1
                continue
            seen_hashes.add(file_hash)

            rel_to_cls = path.relative_to(cls_dir)
            if len(rel_to_cls.parts) > 1:
                sub_source = rel_to_cls.parts[0]
            else:
                sub_source = "direct"

            if cls_name == "not_red":
                subsource_counts[sub_source] += 1

            records.append({
                "filepath": str(path.resolve()).replace("\\", "/"),
                "label": cls_name,
                "sub_source": sub_source,
                "group": f"{cls_name}_{extract_group_prefix(path.name)}",
            })

    print(f"Validated images. Corrupt dropped: {corrupt_count}, Duplicates dropped: {duplicate_count}")
    return records, subsource_counts

def split_groups(
    records: list[dict[str, str]],
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    rng = random.Random(seed)
    by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for rec in records:
        by_class[rec["label"]].append(rec)

    train_records: list[dict[str, str]] = []
    val_records: list[dict[str, str]] = []
    test_records: list[dict[str, str]] = []

    for items in by_class.values():
        groups: dict[str, list[dict[str, str]]] = defaultdict(list)
        for item in items:
            groups[item["group"]].append(item)

        group_keys = sorted(groups.keys())
        rng.shuffle(group_keys)

        total_cls_items = len(items)
        target_train = round(total_cls_items * train_ratio)
        target_val = round(total_cls_items * val_ratio)

        cur_train: list[dict[str, str]] = []
        cur_val: list[dict[str, str]] = []
        cur_test: list[dict[str, str]] = []

        for g in group_keys:
            g_items = groups[g]
            if len(cur_train) + len(g_items) <= target_train or (
                len(cur_train) < target_train and not cur_val
            ):
                cur_train.extend(g_items)
            elif len(cur_val) + len(g_items) <= target_val or (
                len(cur_val) < target_val and not cur_test
            ):
                cur_val.extend(g_items)
            else:
                cur_test.extend(g_items)

        if not cur_val and cur_test:
            cur_val.append(cur_test.pop())
        if not cur_test and cur_val:
            cur_test.append(cur_val.pop())

        train_records.extend(cur_train)
        val_records.extend(cur_val)
        test_records.extend(cur_test)

    return train_records, val_records, test_records

def write_split_csv(output_path: Path, records: list[dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "label", "sub_source", "group"])
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)

def print_composition(
    train_recs: list[dict[str, str]],
    val_recs: list[dict[str, str]],
    test_recs: list[dict[str, str]],
    subsource_counts: dict[str, int],
) -> None:
    splits = {"train": train_recs, "val": val_recs, "test": test_recs}
    print("\n--- Split Composition ---")
    for name, recs in splits.items():
        counts: dict[str, int] = defaultdict(int)
        for r in recs:
            counts[r["label"]] += 1
        print(f"{name:5s}: Total={len(recs):5d} | red={counts['red']:5d} | not_red={counts['not_red']:5d}")

    if subsource_counts:
        print("\n--- 'not_red' Sub-source Composition ---")
        for sub, cnt in sorted(subsource_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {sub:15s}: {cnt:5d}")

    total_images = len(train_recs) + len(val_recs) + len(test_recs)
    if total_images == 0:
        print("\nERROR: No valid images found in data directory!")
        return

    red_total = sum(1 for r in train_recs + val_recs + test_recs if r["label"] == "red")
    red_ratio = red_total / total_images

    train_red = sum(1 for r in train_recs if r["label"] == "red")
    train_not_red = sum(1 for r in train_recs if r["label"] == "not_red")

    warnings: list[str] = []
    if train_red < 300:
        warnings.append(
            f"LOUD WARNING: 'red' class has only {train_red} training images (< 300 minimum target)! Add more real red traffic light crops."
        )
    if train_not_red < 300:
        warnings.append(
            f"LOUD WARNING: 'not_red' class has only {train_not_red} training images (< 300 minimum target)! Add more non-red crops (especially amber/yellow and green)."
        )
    if red_ratio < 0.25:
        warnings.append(
            f"LOUD WARNING: Red class represents only {red_ratio:.1%} of data (< 25% minimum target)! Add more red traffic lights."
        )
    elif red_ratio > 0.75:
        warnings.append(
            f"LOUD WARNING: Red class represents {red_ratio:.1%} of data (> 75% maximum target)! Add more non-red traffic lights."
        )

    if warnings:
        print("\n" + "=" * 80)
        for w in warnings:
            print(w)
        print("=" * 80)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()

    records, subsource_counts = validate_and_collect_images(args.data_dir)
    if not records:
        print(f"No valid images found in {args.data_dir}. Place images into {args.data_dir}/red and {args.data_dir}/not_red.")
        return

    train_recs, val_recs, test_recs = split_groups(
        records,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    write_split_csv(args.output_dir / "train.csv", train_recs)
    write_split_csv(args.output_dir / "val.csv", val_recs)
    write_split_csv(args.output_dir / "test.csv", test_recs)

    print_composition(train_recs, val_recs, test_recs, subsource_counts)

if __name__ == "__main__":
    main()

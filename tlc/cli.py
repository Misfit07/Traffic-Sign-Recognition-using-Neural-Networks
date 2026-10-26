from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from tlc.engine import (
    classify,
    classify_batch,
    classify_proba,
    get_default_model_path,
    resolve_threshold,
)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

def collect_image_files(target: Path, recursive: bool = False) -> list[Path]:
    if target.is_file():
        return [target]
    if not target.is_dir():
        raise FileNotFoundError(f"Path not found: {target}")
    glob_fn = target.rglob if recursive else target.glob
    return [p for p in sorted(glob_fn("*")) if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS]

def run_predict(args: argparse.Namespace) -> int:
    target = Path(args.path)
    files = collect_image_files(target, recursive=args.recursive)
    if not files:
        sys.stderr.write(f"No valid image files found in {target}\n")
        return 1

    model_p = Path(args.model) if args.model else get_default_model_path()
    th = resolve_threshold(model_p, args.threshold)

    if args.json:
        records: list[dict[str, Any]] = []
        for f in files:
            p_red = classify_proba(f, model_path=model_p)
            label = "red" if p_red >= th else "not_red"
            records.append({
                "file": str(f).replace("\\", "/"),
                "label": label,
                "p_red": round(p_red, 4),
            })
        print(json.dumps(records, indent=2))
        return 0

    batch_size = max(1, args.batch_size)
    for i in range(0, len(files), batch_size):
        chunk = files[i : i + batch_size]
        if args.scene:
            for f in chunk:
                lbl = classify(f, threshold=th, model_path=model_p, scene=True)
                print(f"{f.name}\t{lbl}")
        else:
            labels = classify_batch(chunk, threshold=th, model_path=model_p)
            for f, lbl in zip(chunk, labels, strict=False):
                print(f"{f.name}\t{lbl}")
    return 0

def run_watch(args: argparse.Namespace) -> int:
    watch_dir = Path(args.dir)
    if not watch_dir.is_dir():
        sys.stderr.write(f"Directory not found: {watch_dir}\n")
        return 1

    model_p = Path(args.model) if args.model else get_default_model_path()
    th = resolve_threshold(model_p, args.threshold)
    seen: set[str] = set()

    print(f"Watching {watch_dir} (poll interval: {args.interval}s)... Press Ctrl+C to exit.")
    try:
        while True:
            current_files = collect_image_files(watch_dir, recursive=args.recursive)
            for f in current_files:
                resolved = str(f.resolve())
                if resolved not in seen:
                    seen.add(resolved)
                    p_red = classify_proba(f, model_path=model_p)
                    label = "red" if p_red >= th else "not_red"
                    print(f"{f.name}\t{label}\t{p_red:.4f}")
                    sys.stdout.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0

def run_bench(args: argparse.Namespace) -> int:
    bench_dir = Path(args.dir)
    files = collect_image_files(bench_dir, recursive=args.recursive)
    if not files:
        sys.stderr.write(f"No valid images found for benchmark in {bench_dir}\n")
        return 1

    model_p = Path(args.model) if args.model else get_default_model_path()
    print(f"Benchmarking inference on {len(files)} image(s)...")

    warmup_n = min(len(files), 10)
    for i in range(warmup_n):
        classify(files[i], model_path=model_p)

    single_times: list[float] = []
    repeat_count = max(len(files), 50)
    for i in range(repeat_count):
        target_f = files[i % len(files)]
        t0 = time.perf_counter()
        classify(target_f, model_path=model_p)
        t1 = time.perf_counter()
        single_times.append((t1 - t0) * 1000.0)

    mean_ms = float(np.mean(single_times))
    p50_ms = float(np.percentile(single_times, 50))
    p95_ms = float(np.percentile(single_times, 95))

    batch_size = 64
    eval_files = (files * (max(1, (batch_size * 4) // len(files)) + 1))[: batch_size * 8]
    batch_times: list[float] = []

    for i in range(0, len(eval_files), batch_size):
        chunk = eval_files[i : i + batch_size]
        t0 = time.perf_counter()
        classify_batch(chunk, model_path=model_p)
        t1 = time.perf_counter()
        batch_times.append(t1 - t0)

    total_images = len(eval_files)
    total_batch_sec = sum(batch_times)
    throughput = total_images / total_batch_sec if total_batch_sec > 0 else 0.0

    print("\n--- Latency & Throughput Benchmark Report ---")
    print("Single-image E2E (Decode + Preprocess + Infer):")
    print(f"  Mean latency: {mean_ms:.2f} ms")
    print(f"  P50 latency : {p50_ms:.2f} ms")
    print(f"  P95 latency : {p95_ms:.2f} ms")
    print(f"Batch inference (batch_size={batch_size}):")
    print(f"  Throughput  : {throughput:.1f} images/sec")
    return 0

def main() -> None:
    parser = argparse.ArgumentParser(prog="tlc")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pred_p = subparsers.add_parser("predict")
    pred_p.add_argument("path", type=str)
    pred_p.add_argument("-r", "--recursive", action="store_true")
    pred_p.add_argument("--json", action="store_true")
    pred_p.add_argument("--batch-size", type=int, default=64)
    pred_p.add_argument("--threshold", type=float, default=None)
    pred_p.add_argument("--model", type=str, default=None)
    pred_p.add_argument("--scene", action="store_true")

    watch_p = subparsers.add_parser("watch")
    watch_p.add_argument("dir", type=str)
    watch_p.add_argument("-r", "--recursive", action="store_true")
    watch_p.add_argument("--interval", type=float, default=1.0)
    watch_p.add_argument("--threshold", type=float, default=None)
    watch_p.add_argument("--model", type=str, default=None)

    bench_p = subparsers.add_parser("bench")
    bench_p.add_argument("dir", type=str)
    bench_p.add_argument("-r", "--recursive", action="store_true")
    bench_p.add_argument("--threshold", type=float, default=None)
    bench_p.add_argument("--model", type=str, default=None)

    ui_p = subparsers.add_parser("ui")
    ui_p.add_argument("--host", type=str, default="127.0.0.1")
    ui_p.add_argument("--port", type=int, default=8000)
    ui_p.add_argument("--no-browser", action="store_true")

    web_p = subparsers.add_parser("web")
    web_p.add_argument("--host", type=str, default="127.0.0.1")
    web_p.add_argument("--port", type=int, default=8000)
    web_p.add_argument("--no-browser", action="store_true")

    args = parser.parse_args()
    try:
        if args.command == "predict":
            code = run_predict(args)
        elif args.command == "watch":
            code = run_watch(args)
        elif args.command == "bench":
            code = run_bench(args)
        elif args.command in ("ui", "web"):
            from tlc.web import run_server

            run_server(host=args.host, port=args.port, open_browser=not args.no_browser)
            code = 0
        else:
            code = 1
        sys.exit(code)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()

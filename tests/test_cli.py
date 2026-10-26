from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from tlc import classify
from tlc.engine import get_default_model_path

MODEL_EXISTS = get_default_model_path().is_file()
TEST_CSV_PATH = Path("data/splits/test.csv")

def test_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "tlc", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "predict" in result.stdout
    assert "watch" in result.stdout
    assert "bench" in result.stdout

@pytest.mark.skipif(not MODEL_EXISTS, reason="Trained model not yet exported")
def test_cli_predict_single_image(tmp_path: Path) -> None:
    img_path = tmp_path / "sample.jpg"
    Image.new("RGB", (64, 64), (255, 0, 0)).save(img_path)

    res = subprocess.run(
        [sys.executable, "-m", "tlc", "predict", str(img_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0
    assert "sample.jpg" in res.stdout
    assert ("red" in res.stdout or "not_red" in res.stdout)

@pytest.mark.skipif(not MODEL_EXISTS, reason="Trained model not yet exported")
def test_cli_predict_json(tmp_path: Path) -> None:
    img_path = tmp_path / "sample2.jpg"
    Image.new("RGB", (64, 64), (0, 255, 0)).save(img_path)

    res = subprocess.run(
        [sys.executable, "-m", "tlc", "predict", str(tmp_path), "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0
    data = json.loads(res.stdout)
    assert isinstance(data, list)
    assert len(data) >= 1
    assert "label" in data[0]
    assert "p_red" in data[0]

@pytest.mark.skipif(not MODEL_EXISTS, reason="Trained model not yet exported")
def test_cli_bench(tmp_path: Path) -> None:
    for i in range(5):
        Image.new("RGB", (64, 64), (i * 20, 100, 50)).save(tmp_path / f"bench_{i}.jpg")

    res = subprocess.run(
        [sys.executable, "-m", "tlc", "bench", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0
    assert "Mean latency" in res.stdout
    assert "Throughput" in res.stdout

@pytest.mark.skipif(
    not (MODEL_EXISTS and TEST_CSV_PATH.is_file()),
    reason="Regression test requires trained model and data/splits/test.csv",
)
def test_regression_ten_real_images_per_class() -> None:
    manifest = Path("tests/regression_samples.json")
    if not manifest.is_file():
        pytest.skip("Regression samples manifest not found")

    with open(manifest, "r", encoding="utf-8") as f:
        data = json.load(f)

    red_samples = [Path(p) for p in data.get("red", []) if Path(p).is_file()]
    not_red_samples = [Path(p) for p in data.get("not_red", []) if Path(p).is_file()]

    if len(red_samples) < 10 or len(not_red_samples) < 10:
        pytest.skip("Not enough regression test samples found on disk")

    for p in red_samples:
        pred = classify(p)
        assert pred == "red", f"Regression failure on {p}: expected 'red', got '{pred}'"

    for p in not_red_samples:
        pred = classify(p)
        assert pred == "not_red", f"Regression failure on {p}: expected 'not_red', got '{pred}'"

from __future__ import annotations

import io
import json
import threading
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
from PIL import Image

from tlc.engine import get_default_model_path
from tlc.sign_engine import get_default_sign_model_path
from tlc.web import TLCRequestHandler

MODEL_EXISTS = get_default_model_path().is_file()
SIGN_MODEL_EXISTS = get_default_sign_model_path().is_file()


@pytest.fixture(scope="module")
def server_address() -> tuple[str, int]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), TLCRequestHandler)
    host, port = server.server_address
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield host, port
    server.shutdown()
    server.server_close()


def test_web_index_get(server_address: tuple[str, int]) -> None:
    host, port = server_address
    url = f"http://{host}:{port}/"
    with urllib.request.urlopen(url) as response:
        assert response.status == 200
        html = response.read().decode("utf-8")
        assert "<title>TLC - Traffic Light Classifier</title>" in html
        assert "dropzone" in html


@pytest.mark.skipif(not MODEL_EXISTS, reason="Trained model not yet exported")
def test_web_api_classify_scene_no_light(server_address: tuple[str, int], tmp_path: Path) -> None:
    host, port = server_address
    img = Image.new("RGB", (64, 64), (128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    params = urllib.parse.urlencode({"filename": "test.jpg", "threshold": 0.5, "scene": "true"})
    url = f"http://{host}:{port}/api/classify?{params}"
    req = urllib.request.Request(
        url,
        data=img_bytes,
        headers={"Content-Type": "application/octet-stream"},
        method="POST",
    )
    with urllib.request.urlopen(req) as response:
        assert response.status == 200
        data = json.loads(response.read().decode("utf-8"))
        assert "label" in data
        assert "p_red" in data
        assert "latency_ms" in data
        assert data["label"] == "no_traffic_light_found"


@pytest.mark.skipif(not MODEL_EXISTS, reason="Trained model not yet exported")
def test_web_api_classify_direct_crop(server_address: tuple[str, int], tmp_path: Path) -> None:
    host, port = server_address
    img = Image.new("RGB", (64, 64), (255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    params = urllib.parse.urlencode({"filename": "test.jpg", "threshold": 0.5, "scene": "false"})
    url = f"http://{host}:{port}/api/classify?{params}"
    req = urllib.request.Request(
        url,
        data=img_bytes,
        headers={"Content-Type": "application/octet-stream"},
        method="POST",
    )
    with urllib.request.urlopen(req) as response:
        assert response.status == 200
        data = json.loads(response.read().decode("utf-8"))
        assert "label" in data
        assert "p_red" in data
        assert "latency_ms" in data
        assert data["label"] in ("red", "not_red")


@pytest.mark.skipif(not SIGN_MODEL_EXISTS, reason="Sign model not present")
def test_web_api_classify_sign_post(server_address: tuple[str, int], tmp_path: Path) -> None:
    host, port = server_address
    img = Image.new("RGB", (48, 48), (200, 30, 30))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    params = urllib.parse.urlencode({"filename": "sign.jpg", "confidence_threshold": 70.0})
    url = f"http://{host}:{port}/api/classify_sign?{params}"
    req = urllib.request.Request(
        url,
        data=img_bytes,
        headers={"Content-Type": "application/octet-stream"},
        method="POST",
    )
    with urllib.request.urlopen(req) as response:
        assert response.status == 200
        data = json.loads(response.read().decode("utf-8"))
        assert "label" in data
        assert "confidence" in data
        assert "top3" in data
        assert len(data["top3"]) == 3
        assert "clahe_b64" in data
        assert data["clahe_b64"].startswith("data:image/jpeg;base64,")

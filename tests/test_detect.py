from __future__ import annotations

from PIL import Image

from tlc.detect import (
    classify_scene,
    detect_and_classify,
    detect_traffic_lights,
    get_default_detector_path,
)

DETECTOR_EXISTS = get_default_detector_path().is_file()


def test_detect_no_traffic_light_on_blank() -> None:
    if not DETECTOR_EXISTS:
        return
    img = Image.new("RGB", (300, 300), (100, 100, 100))
    boxes = detect_traffic_lights(img)
    assert len(boxes) == 0

    result = classify_scene(img)
    assert result == "no_traffic_light_found"

    result_alias = detect_and_classify(img)
    assert result_alias == "no_traffic_light_found"


def test_detect_handles_rgba_without_error() -> None:
    if not DETECTOR_EXISTS:
        return
    img_rgba = Image.new("RGBA", (200, 200), (255, 0, 0, 128))
    boxes = detect_traffic_lights(img_rgba)
    assert isinstance(boxes, list)

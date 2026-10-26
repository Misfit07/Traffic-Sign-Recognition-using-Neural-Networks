# Real Traffic Light Dataset Acquisition & Preparation Guide

This project strictly requires **real traffic light images** with real variations (lighting, distance, weather, background, sensor noise). Synthetic, dummy, or generated images are not permitted.

---

## 1. Recommended Public Datasets

### A. Bosch Small Traffic Light Dataset (BSTLD)
- **Source**: [https://hci.iwr.uni-heidelberg.de/content/bosch-small-traffic-lights-dataset](https://hci.iwr.uni-heidelberg.de/content/bosch-small-traffic-lights-dataset) / [GitHub mirror](https://github.com/bosch-ros-pkg/bstld)
- **Format**: High-resolution 720p/1080p camera frames with YAML / Pascal VOC style annotations.
- **Classes**: `Red`, `Yellow`, `Green`, `RedLeft`, `RedRight`, `GreenLeft`, `GreenRight`, `off`.
- **Why it's ideal**: Contains real driving conditions, varied day/night lighting, and critical amber/yellow negatives.

### B. LISA Traffic Light Dataset
- **Source**: [Laboratory for Intelligent & Safe Automobiles (UC San Diego)](https://lucasb-eyer.github.io/traffic-lights-dataset/)
- **Format**: Continuous driving video sequences (day & night) with XML bounding box annotations.
- **Classes**: `go` (green), `goForward`, `goLeft`, `stop` (red), `stopLeft`, `warning` (yellow), `warningLeft`.
- **Why it's ideal**: Extensive night sequences with lens flare and street lights. Note that frame sequences share prefixes, which `scripts/prepare_data.py` respects to prevent data leakage across splits.

### C. S2TLD (Small-Scale & Large-Scale Traffic Light Dataset)
- **Source**: [https://github.com/Thinklab-SJTU/S2TLD](https://github.com/Thinklab-SJTU/S2TLD)
- **Format**: 5,780 images captured in normal street conditions with VOC XML annotations.
- **Classes**: `red`, `yellow`, `green`, `off`, `wait_on`.

### D. Kaggle Traffic Light Datasets
- **Examples**:
  - Kaggle Traffic Light Detection (YOLO format): search for "traffic-light-detection" or "self-driving-car-traffic-lights".
  - BDD100K traffic light crops.

---

## 2. Target Directory Structure

The classifier training pipeline expects crops placed into `data/raw/`:

```
data/raw/
├── red/
│   ├── red_0001.jpg
│   ├── red_0002.png
│   └── ...
└── not_red/
    ├── green/           # Optional subfolders for visibility
    │   └── green_0001.jpg
    ├── yellow/          # Critical hard negatives
    │   └── yellow_0001.jpg
    ├── off/             # Unlit signals
    │   └── off_0001.jpg
    └── hard_negatives/  # Red brake lights, red road signs, red buildings
        └── non_light_0001.jpg
```

---

## 3. Cropping Bounding Boxes with `scripts/crop_from_boxes.py`

If you download datasets with bounding box annotations (YOLO `.txt` or Pascal VOC `.xml`), use `scripts/crop_from_boxes.py` to extract crops with 15% contextual padding.

### Example: Cropping YOLO Annotations
```bash
python scripts/crop_from_boxes.py \
    --images-dir path/to/dataset/images \
    --labels-dir path/to/dataset/labels \
    --format yolo \
    --output-dir data/raw \
    --pad-pct 0.15 \
    --class-map "0:red,1:not_red,2:not_red,3:not_red"
```

### Example: Cropping Pascal VOC XML Annotations
```bash
python scripts/crop_from_boxes.py \
    --images-dir path/to/dataset/images \
    --labels-dir path/to/dataset/annotations \
    --format voc \
    --output-dir data/raw \
    --pad-pct 0.15 \
    --class-map "Red:red,RedLeft:red,RedRight:red,Yellow:not_red,Green:not_red,off:not_red"
```

---

## 4. Validating and Splitting Data

Once images are placed in `data/raw/`, run:
```bash
python scripts/prepare_data.py --data-dir data/raw --output-dir data/splits
```

This will:
1. Verify every image opens without corruption.
2. Deduplicate exact images (SHA-256).
3. Group sequence frames by prefix to avoid data leakage.
4. Create stratified 80/10/10 `train.csv`, `val.csv`, and `test.csv`.
5. Warn if any class has $< 300$ samples or if class balance is outside $[25\%, 75\%]$.

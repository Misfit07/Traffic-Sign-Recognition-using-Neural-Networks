# Traffic Light Red Classifier (`tlc`)

A production-grade, zero-torch inference traffic light "is it red" classifier.

## Quickstart

```python
from tlc import classify, is_red, classify_proba, classify_batch

# Single image classification ("red" or "not_red")
label = classify("path/to/crop.jpg")

# Probabilistic score (p_red)
prob = classify_proba("path/to/crop.jpg")

# Boolean helper
flag = is_red("path/to/crop.jpg")

# Batch classification
labels = classify_batch(["img1.jpg", "img2.jpg"])
```

## CLI Usage

```bash
# Predict single image or directory
python -m tlc predict path/to/images -r

# Machine-readable JSON output
python -m tlc predict path/to/images --json

# Folder polling watcher
python -m tlc watch incoming_dir/ --interval 0.5

# End-to-end latency & throughput benchmark
python -m tlc bench test_images/
```

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import onnxruntime as ort
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from train.dataset import (
    TrafficLightDataset,
    compute_pos_weight,
    create_weighted_sampler,
)
from train.evaluate import (
    copy_worst_errors,
    evaluate_metrics,
    plot_confusion_matrix,
    plot_pr_curve,
    save_metrics_report,
    sweep_threshold_on_val,
)
from train.export import (
    export_to_onnx,
    get_file_size_mb,
    quantize_to_int8,
    verify_onnx_parity,
)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_model() -> nn.Module:
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 1024),
        nn.Hardswish(),
        nn.Dropout(p=0.2),
        nn.Linear(1024, 1),
    )
    return model

def smooth_labels(targets: torch.Tensor, smoothing: float = 0.05) -> torch.Tensor:
    return targets * (1.0 - 2.0 * smoothing) + smoothing

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    label_smoothing: float = 0.05,
) -> float:
    model.train()
    total_loss = 0.0
    use_cuda = device.type == "cuda"

    for batch_x, batch_y, _ in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        smooth_y = smooth_labels(batch_y, label_smoothing)

        optimizer.zero_grad()
        if scaler and use_cuda:
            with torch.amp.autocast("cuda"):
                logits = model(batch_x)
                loss = criterion(logits, smooth_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(batch_x)
            loss = criterion(logits, smooth_y)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item()) * batch_x.size(0)

    return total_loss / len(dataloader.dataset)

def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model.eval()
    all_targets: list[float] = []
    all_probs: list[float] = []
    all_paths: list[str] = []

    with torch.no_grad():
        for batch_x, batch_y, paths in dataloader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            all_probs.extend(probs.tolist())
            all_targets.extend(batch_y.numpy().ravel().tolist())
            all_paths.extend(paths)

    return np.array(all_targets, dtype=int), np.array(all_probs, dtype=np.float32), all_paths

def train_pipeline(
    data_dir: Path,
    epochs: int,
    batch_size: int,
    img_size: int,
    device_name: str,
    output_dir: Path,
    imbalance_mode: str = "pos_weight",
) -> dict[str, Any]:
    set_seed(42)
    device = torch.device(device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu")
    print(f"Training on device: {device}")

    train_csv = data_dir / "train.csv"
    val_csv = data_dir / "val.csv"
    test_csv = data_dir / "test.csv"

    if not train_csv.is_file() or not val_csv.is_file() or not test_csv.is_file():
        raise FileNotFoundError(f"Missing split CSVs in {data_dir}. Run scripts/prepare_data.py first.")

    train_ds = TrafficLightDataset(train_csv, is_train=True, target_size=(img_size, img_size))
    val_ds = TrafficLightDataset(val_csv, is_train=False, target_size=(img_size, img_size))
    test_ds = TrafficLightDataset(test_csv, is_train=False, target_size=(img_size, img_size))

    if len(train_ds) == 0:
        raise ValueError("Train dataset is empty! Aborting training.")

    sampler = create_weighted_sampler(train_ds) if imbalance_mode == "sampler" else None
    shuffle = sampler is None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = build_model().to(device)

    if imbalance_mode == "pos_weight":
        pos_weight = compute_pos_weight(train_ds).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Using BCEWithLogitsLoss pos_weight: {pos_weight.item():.4f}")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print("Using standard BCEWithLogitsLoss with WeightedRandomSampler")

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    for p in model.features.parameters():
        p.requires_grad = False
    head_optimizer = AdamW(
        [p for p in model.classifier.parameters() if p.requires_grad],
        lr=1e-3,
        weight_decay=1e-4,
    )

    print("\n--- Phase 1: Backbone Frozen (3 Epochs, lr=1e-3) ---")
    best_val_f1 = -1.0
    best_weights_path = output_dir / "best_model.pt"
    output_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(1, 4):
        train_loss = train_one_epoch(model, train_loader, head_optimizer, criterion, device, scaler)
        val_y, val_p, _ = evaluate_model(model, val_loader, device)
        val_preds = (val_p >= 0.5).astype(int)
        val_rec = recall_score(val_y, val_preds, pos_label=1, zero_division=0)
        val_f1 = f1_score(val_y, val_preds, pos_label=1, zero_division=0)
        print(f"Epoch {ep:2d}/3 (Head) | Train Loss: {train_loss:.4f} | Val Recall: {val_rec:.4f} | Val F1: {val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_weights_path)

    print("\n--- Phase 2: Full Fine-Tuning (AdamW lr=1e-4, Cosine Decay, Early Stopping Patience=6) ---")
    for p in model.features.parameters():
        p.requires_grad = True

    ft_optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    remaining_epochs = max(1, epochs - 3)
    scheduler = CosineAnnealingLR(ft_optimizer, T_max=remaining_epochs, eta_min=1e-6)

    patience = 6
    epochs_no_improve = 0

    for ep in range(4, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, ft_optimizer, criterion, device, scaler)
        scheduler.step()
        val_y, val_p, _ = evaluate_model(model, val_loader, device)
        val_preds = (val_p >= 0.5).astype(int)
        val_rec = recall_score(val_y, val_preds, pos_label=1, zero_division=0)
        val_prec = precision_score(val_y, val_preds, pos_label=1, zero_division=0)
        val_f1 = f1_score(val_y, val_preds, pos_label=1, zero_division=0)

        improved = val_f1 > best_val_f1
        if improved:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_weights_path)
            mark = "*"
        else:
            epochs_no_improve += 1
            mark = " "

        print(
            f"Epoch {ep:2d}/{epochs} | Loss: {train_loss:.4f} | Val Prec: {val_prec:.4f} | "
            f"Val Recall: {val_rec:.4f} | Val F1: {val_f1:.4f} {mark}"
        )

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {ep} epochs (val F1 did not improve for {patience} epochs).")
            break

    print(f"\nLoading best checkpoint from {best_weights_path} with Val F1={best_val_f1:.4f}")
    model.load_state_dict(torch.load(best_weights_path, map_location=device, weights_only=True))

    val_y, val_p, _ = evaluate_model(model, val_loader, device)
    chosen_threshold, val_metrics_at_th, selection_reason = sweep_threshold_on_val(
        val_y, val_p, target_recall=0.98
    )
    print("\nThreshold optimization on VAL set:")
    print(f"  Chosen threshold : {chosen_threshold:.3f}")
    print(f"  Val Red Recall   : {val_metrics_at_th['red_recall']:.4f}")
    print(f"  Val Red Precision: {val_metrics_at_th['red_precision']:.4f}")
    print(f"  Val Red F1       : {val_metrics_at_th['f1']:.4f}")
    print(f"  Reason           : {selection_reason}")

    fp32_onnx_path = output_dir / "traffic_light_fp32.onnx"
    export_to_onnx(model.to("cpu"), fp32_onnx_path, img_size=img_size, opset_version=17)
    print(f"Exported FP32 ONNX model to {fp32_onnx_path} ({get_file_size_mb(fp32_onnx_path):.2f} MB)")

    parity_err = verify_onnx_parity(model.to("cpu"), fp32_onnx_path, val_loader, max_samples=32)
    print(f"Verified PyTorch vs ONNX parity on 32 val images: max diff = {parity_err:.6e} (tolerance <= 1e-4)")

    int8_onnx_path = output_dir / "traffic_light_int8.onnx"
    try:
        quantize_to_int8(fp32_onnx_path, int8_onnx_path)
        int8_val_p: list[float] = []
        int8_session = ort.InferenceSession(str(int8_onnx_path), providers=["CPUExecutionProvider"])
        input_name = int8_session.get_inputs()[0].name
        for bx, _, _ in val_loader:
            out = int8_session.run(None, {input_name: bx.numpy()})[0]
            int8_val_p.extend((1.0 / (1.0 + np.exp(-out))).ravel().tolist())

        int8_val_metrics = evaluate_metrics(val_y, np.array(int8_val_p), chosen_threshold)
        recall_drop = val_metrics_at_th["red_recall"] - int8_val_metrics["red_recall"]
        f1_drop = val_metrics_at_th["f1"] - int8_val_metrics["red_f1"]

        keep_int8 = recall_drop <= 0.005 and f1_drop <= 0.005
        print(
            f"INT8 Quantization: FP32 F1={val_metrics_at_th['f1']:.4f} -> INT8 F1={int8_val_metrics['red_f1']:.4f} "
            f"(drop: {f1_drop:.4f}), Recall drop: {recall_drop:.4f}"
        )
    except (OSError, RuntimeError, ValueError) as e:
        print(f"Quantization failed or skipped: {e}")
        keep_int8 = False

    final_model_path = output_dir / "traffic_light.onnx"
    if keep_int8 and int8_onnx_path.is_file():
        if final_model_path.exists():
            final_model_path.unlink()
        int8_onnx_path.rename(final_model_path)
        selected_quant = True
        print(f"Kept INT8 model as {final_model_path} ({get_file_size_mb(final_model_path):.2f} MB)")
    else:
        if final_model_path.exists():
            final_model_path.unlink()
        fp32_onnx_path.rename(final_model_path)
        selected_quant = False
        print(f"Kept FP32 model as {final_model_path} ({get_file_size_mb(final_model_path):.2f} MB)")

    pkg_models_dir = Path(__file__).parent.parent / "tlc" / "models"
    pkg_models_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(final_model_path, pkg_models_dir / "traffic_light.onnx")

    labels_data = {
        "labels": ["not_red", "red"],
        "threshold": float(chosen_threshold),
        "imbalance_strategy": imbalance_mode,
        "selection_reason": selection_reason,
        "quantized": selected_quant,
        "model_size_mb": get_file_size_mb(final_model_path),
    }
    with open(output_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(labels_data, f, indent=2)
    with open(pkg_models_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(labels_data, f, indent=2)

    print("\n--- Held-Out Test Evaluation (ONCE) ---")
    test_y, test_p, test_paths = evaluate_model(model.to(device), test_loader, device)
    test_metrics = evaluate_metrics(test_y, test_p, chosen_threshold)

    print(f"Test Accuracy      : {test_metrics['accuracy']:.4f}")
    print(f"Test Red Precision : {test_metrics['red_precision']:.4f}")
    print(f"Test Red Recall    : {test_metrics['red_recall']:.4f}")
    print(f"Test Red F1        : {test_metrics['red_f1']:.4f}")
    print(f"Test ROC-AUC       : {test_metrics['roc_auc']:.4f}")
    print(f"Test PR-AUC        : {test_metrics['pr_auc']:.4f}")

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    test_preds = (test_p >= chosen_threshold).astype(int)
    plot_confusion_matrix(test_y, test_preds, reports_dir / "confusion_matrix.png")
    plot_pr_curve(test_y, test_p, reports_dir / "pr_curve.png")

    saved_fr, saved_mr = copy_worst_errors(
        test_paths, test_y, test_p, chosen_threshold, reports_dir, top_k=20
    )
    print(f"Triage: Copied {saved_fr} false_red and {saved_mr} missed_red to {reports_dir / 'errors'}")

    full_report = {
        "threshold": chosen_threshold,
        "val_metrics": val_metrics_at_th,
        "test_metrics": test_metrics,
        "labels": ["not_red", "red"],
        "selection_reason": selection_reason,
        "imbalance_strategy": imbalance_mode,
        "model_size_mb": get_file_size_mb(final_model_path),
        "parity_max_diff": parity_err,
        "quantized": selected_quant,
    }
    save_metrics_report(full_report, reports_dir / "metrics.json")
    save_metrics_report(full_report, output_dir / "metrics.json")

    return full_report

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/splits"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=Path("models"))
    parser.add_argument("--imbalance-mode", choices=["pos_weight", "sampler"], default="pos_weight")
    args = parser.parse_args()

    train_pipeline(
        data_dir=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        device_name=args.device,
        output_dir=args.out,
        imbalance_mode=args.imbalance_mode,
    )

if __name__ == "__main__":
    main()

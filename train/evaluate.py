from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def sweep_threshold_on_val(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    target_recall: float = 0.98,
) -> tuple[float, dict[str, float], str]:
    thresholds = np.linspace(0.01, 0.99, 99)
    best_th_target = -1.0
    metrics_at_target: dict[str, float] = {}

    max_f1_th = 0.5
    max_f1 = -1.0
    metrics_at_max_f1: dict[str, float] = {}

    for th in thresholds:
        preds = (y_probs >= th).astype(int)
        rec = float(recall_score(y_true, preds, pos_label=1, zero_division=0))
        prec = float(precision_score(y_true, preds, pos_label=1, zero_division=0))
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        acc = float(accuracy_score(y_true, preds))

        cur_metrics = {
            "threshold": float(th),
            "red_recall": rec,
            "red_precision": prec,
            "f1": f1,
            "accuracy": acc,
        }

        if f1 > max_f1:
            max_f1 = f1
            max_f1_th = float(th)
            metrics_at_max_f1 = cur_metrics

        if rec >= target_recall and th > best_th_target:
            best_th_target = float(th)
            metrics_at_target = cur_metrics

    if best_th_target > 0:
        reason = f"Selected highest threshold ({best_th_target:.3f}) achieving target red recall >= {target_recall:.2f}"
        return best_th_target, metrics_at_target, reason

    reason = f"Target red recall {target_recall:.2f} unreachable; selected max-F1 threshold ({max_f1_th:.3f})"
    return max_f1_th, metrics_at_max_f1, reason

def evaluate_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    y_pred = (y_probs >= threshold).astype(int)
    acc = float(accuracy_score(y_true, y_pred))
    rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    try:
        roc_auc = float(roc_auc_score(y_true, y_probs))
    except ValueError:
        roc_auc = 0.0

    p_curve, r_curve, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = float(auc(r_curve, p_curve))

    return {
        "threshold": float(threshold),
        "accuracy": acc,
        "red_precision": prec,
        "red_recall": rec,
        "red_f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    classes = ["not_red", "red"]
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title="Confusion Matrix (Test Set)",
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

def plot_pr_curve(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall, precision)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="darkorange", lw=2, label=f"PR curve (AUC = {pr_auc:.3f})")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall (red class)")
    ax.set_ylabel("Precision (red class)")
    ax.set_title("Precision-Recall Curve (Test Set)")
    ax.legend(loc="lower left")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

def copy_worst_errors(
    file_paths: list[str],
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float,
    reports_dir: Path,
    top_k: int = 20,
) -> tuple[int, int]:
    false_red_dir = reports_dir / "errors" / "false_red"
    missed_red_dir = reports_dir / "errors" / "missed_red"

    if false_red_dir.exists():
        shutil.rmtree(false_red_dir)
    if missed_red_dir.exists():
        shutil.rmtree(missed_red_dir)

    false_red_dir.mkdir(parents=True, exist_ok=True)
    missed_red_dir.mkdir(parents=True, exist_ok=True)

    false_reds: list[tuple[float, str]] = []
    missed_reds: list[tuple[float, str]] = []

    for path_str, true_lbl, prob in zip(file_paths, y_true, y_probs, strict=False):
        pred_lbl = 1 if prob >= threshold else 0
        if true_lbl == 0 and pred_lbl == 1:
            false_reds.append((float(prob), path_str))
        elif true_lbl == 1 and pred_lbl == 0:
            missed_reds.append((float(prob), path_str))

    false_reds.sort(key=lambda x: x[0], reverse=True)
    missed_reds.sort(key=lambda x: x[0])

    saved_fr = 0
    for prob, p_str in false_reds[:top_k]:
        src = Path(p_str)
        if src.is_file():
            dst = false_red_dir / f"p_{prob:.4f}_{src.name}"
            shutil.copy2(src, dst)
            saved_fr += 1

    saved_mr = 0
    for prob, p_str in missed_reds[:top_k]:
        src = Path(p_str)
        if src.is_file():
            dst = missed_red_dir / f"p_{prob:.4f}_{src.name}"
            shutil.copy2(src, dst)
            saved_mr += 1

    return saved_fr, saved_mr

def save_metrics_report(
    metrics: dict[str, Any],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

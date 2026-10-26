from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from torch import nn


def export_to_onnx(
    model: nn.Module,
    output_path: Path,
    img_size: int = 128,
    opset_version: int = 17,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy_input = torch.randn(1, 3, img_size, img_size, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    return output_path

def verify_onnx_parity(
    model: nn.Module,
    onnx_path: Path,
    val_loader: Any,
    max_samples: int = 32,
    atol: float = 1e-4,
) -> float:
    model.eval()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    collected = 0
    max_prob_diff = 0.0
    with torch.no_grad():
        for batch_x, _, _ in val_loader:
            batch_x_cpu = batch_x.to("cpu")
            torch_out = model(batch_x_cpu).numpy()
            onnx_out = session.run(None, {input_name: batch_x_cpu.numpy()})[0]
            torch_p = 1.0 / (1.0 + np.exp(-torch_out))
            onnx_p = 1.0 / (1.0 + np.exp(-onnx_out))
            diff = float(np.max(np.abs(torch_p - onnx_p)))
            max_prob_diff = max(max_prob_diff, diff)
            collected += batch_x.size(0)
            if collected >= max_samples:
                break

    if max_prob_diff > atol:
        raise ValueError(
            f"ONNX parity check failed: maximum difference {max_prob_diff:.6f} exceeded tolerance {atol:.6f}"
        )
    return max_prob_diff

def quantize_to_int8(
    fp32_onnx_path: Path,
    int8_onnx_path: Path,
) -> Path:
    int8_onnx_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=str(fp32_onnx_path),
        model_output=str(int8_onnx_path),
        weight_type=QuantType.QUInt8,
    )
    return int8_onnx_path

def get_file_size_mb(path: Path) -> float:
    return os.path.getsize(path) / (1024.0 * 1024.0)

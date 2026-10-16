"""
Efficiency, Parameter Budget, and Latency Profiler.
Measures:
- Trainable parameter count
- Model checkpoint size
- Latency (ms) with warmup runs and synchronization
- Throughput (FPS)
- Estimated FLOPs / MACs
"""

from typing import Dict, Any, Tuple
import time
import os
import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Returns (trainable_params, total_params)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def estimate_flops(model: nn.Module, input_size: Tuple[int, int, int, int] = (1, 3, 256, 256)) -> float:
    """
    Lightweight exact MACs/FLOPs estimation for standard Conv2d and Depthwise-Separable blocks.
    Returns estimated GFLOPs.
    """
    total_macs = 0
    device = next(model.parameters()).device
    x = torch.zeros(input_size, device=device)

    def conv_hook(module, inp, out):
        nonlocal total_macs
        if isinstance(module, nn.Conv2d):
            # MACs = out_h * out_w * (in_c / groups) * k_h * k_w * out_c
            b, out_c, out_h, out_w = out.shape
            in_c = module.in_channels
            k_h, k_w = module.kernel_size
            macs = out_h * out_w * (in_c // module.groups) * k_h * k_w * out_c
            total_macs += macs

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))

    with torch.no_grad():
        try:
            _ = model(x)
        finally:
            for h in hooks:
                h.remove()

    gflops = (total_macs * 2.0) / 1e9  # 1 MAC ~= 2 FLOPs
    return gflops


def benchmark_efficiency(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 3, 400, 600),
    num_runs: int = 50,
    warmup: int = 10,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Profiles latency, throughput, params, and estimated GFLOPs.
    """
    model.eval()
    model.to(device)

    x = torch.randn(input_size, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
            if device.startswith("cuda"):
                torch.cuda.synchronize()

    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # ms

    mean_latency = float(sum(latencies) / len(latencies))
    fps = 1000.0 / mean_latency if mean_latency > 0 else 0.0

    trainable_params, total_params = count_parameters(model)
    gflops = estimate_flops(model, input_size)

    return {
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
        "gflops": gflops,
        "mean_latency_ms": mean_latency,
        "throughput_fps": fps,
        "device": device,
        "input_resolution": f"{input_size[2]}x{input_size[3]}",
    }

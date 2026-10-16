"""
Benchmark script: Parameters, Model Checkpoint Size, FLOPs, Latency, and Throughput.
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from models.lumaguard_net import LumaGuardNet
from metrics.efficiency import benchmark_efficiency, count_parameters


def main():
    parser = argparse.ArgumentParser(description="Benchmark LumaGuard-Net Efficiency")
    parser.add_argument("--height", type=int, default=400, help="Input height")
    parser.add_argument("--width", type=int, default=600, help="Input width")
    parser.add_argument("--runs", type=int, default=20, help="Benchmark iteration count")
    parser.add_argument("--device", type=str, default="cpu", help="'cpu' or 'cuda'")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA not available on this machine. Falling back to CPU.")
        device = "cpu"

    model = LumaGuardNet()
    trainable_p, total_p = count_parameters(model)

    print("=" * 60)
    print(" LumaGuard-Net Efficiency and Resource Benchmark")
    print("=" * 60)
    print(f"Trainable Parameters: {trainable_p:,}")
    print(f"Total Parameters:     {total_p:,}")
    print(f"Parameter Budget (<1M Target): {'PASS' if trainable_p < 1_000_000 else 'FAIL'}")

    res = benchmark_efficiency(
        model,
        input_size=(1, 3, args.height, args.width),
        num_runs=args.runs,
        warmup=5,
        device=device,
    )

    print(f"Benchmarked Device:   {res['device']}")
    print(f"Input Resolution:     {res['input_resolution']}")
    print(f"Estimated GFLOPs:     {res['gflops']:.3f} GFLOPs")
    print(f"Mean Latency:         {res['mean_latency_ms']:.2f} ms")
    print(f"Throughput:           {res['throughput_fps']:.2f} FPS")
    print("=" * 60)


if __name__ == "__main__":
    main()

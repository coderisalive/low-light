"""
Script to automate ablation experiments across architecture configurations:
1. Full LumaGuard-Net
2. Without Noise Estimation (w/o N)
3. Without Reliability Gate (w/o G)
4. Without Color Separation (Direct RGB Head)
5. Standard Convolution Backbone (w/o Depthwise-Separable)
6. Without Masked Blind-Spot Consistency (w/o L_mask)
7. Without Structural Loss (w/o L_struct)
"""

import copy
import os
import sys
from pathlib import Path
import csv
import yaml

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.lumaguard_net import LumaGuardNet
from metrics.efficiency import count_parameters


def run_ablation_summary():
    base_cfg = {
        "model": {
            "base_channels": 24,
            "feature_channels": 32,
            "num_blocks": 4,
            "use_dw_conv": True,
            "use_reliability_map": True,
            "use_reliability_gate": True,
            "use_noise_estimation": True,
            "use_separate_color": True,
            "act_type": "leaky_relu",
        },
        "loss": {
            "w_mask": 0.5,
            "w_exp": 8.0,
            "w_smooth": 15.0,
            "w_color": 3.0,
            "w_struct": 0.1,
            "w_identity": 0.05,
        },
    }

    ablations = [
        ("Full LumaGuard-Net", base_cfg),
        (
            "w/o Noise Estimation",
            copy.deepcopy(base_cfg),
            {"model": {"use_noise_estimation": False}},
        ),
        (
            "w/o Reliability Gate",
            copy.deepcopy(base_cfg),
            {"model": {"use_reliability_gate": False}},
        ),
        (
            "w/o Separate Color Head (RGB Direct)",
            copy.deepcopy(base_cfg),
            {"model": {"use_separate_color": False}},
        ),
        (
            "Standard Conv (w/o Depthwise-Sep)",
            copy.deepcopy(base_cfg),
            {"model": {"use_dw_conv": False}},
        ),
        (
            "w/o Masked Consistency",
            copy.deepcopy(base_cfg),
            {"loss": {"w_mask": 0.0}},
        ),
        (
            "w/o Structural Loss",
            copy.deepcopy(base_cfg),
            {"loss": {"w_struct": 0.0}},
        ),
    ]

    out_csv = Path("experiments/ablation_matrix.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    print("=" * 80)
    print(f"{'Ablation Variant':<38} | {'Params':<10} | {'Status'}")
    print("=" * 80)

    for item in ablations:
        name = item[0]
        cfg = item[1]
        if len(item) > 2:
            updates = item[2]
            for section, sub_dict in updates.items():
                cfg[section].update(sub_dict)

        model = LumaGuardNet(
            base_channels=cfg["model"]["base_channels"],
            feature_channels=cfg["model"]["feature_channels"],
            num_blocks=cfg["model"]["num_blocks"],
            use_dw_conv=cfg["model"]["use_dw_conv"],
            use_reliability_map=cfg["model"]["use_reliability_map"],
            use_reliability_gate=cfg["model"]["use_reliability_gate"],
            use_noise_estimation=cfg["model"]["use_noise_estimation"],
            use_separate_color=cfg["model"]["use_separate_color"],
        )
        trainable_p, _ = count_parameters(model)
        rows.append({
            "variant": name,
            "params": trainable_p,
            "status": "Ready for Training & Eval",
        })
        print(f"{name:<38} | {trainable_p:<10,} | Ready")

    print("=" * 80)

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "params", "status"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved ablation matrix to: {out_csv}")


if __name__ == "__main__":
    run_ablation_summary()

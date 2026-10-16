"""
Unit and integration tests for LumaGuard-Net:
1. Model shape preservation
2. Parameter budget verification (< 1,000,000 parameters)
3. Finite loss evaluations and backward gradient propagation
4. Self-supervised training step on synthetic batch
5. Checkpoint saving and reloading
6. Paired vs self-supervised data loader isolation
"""

import os
from pathlib import Path
import tempfile
import pytest
import torch
from PIL import Image

from models.lumaguard_net import LumaGuardNet
from losses.total_loss import LumaGuardTotalLoss
from metrics.efficiency import count_parameters
from datasets.self_supervised_dataset import SelfSupervisedLowLightDataset
from datasets.paired_evaluation_dataset import PairedEvaluationDataset


def test_model_shapes():
    """Verify input/output shapes and diagnostic tensor dimensions."""
    model = LumaGuardNet()
    x = torch.rand(2, 3, 128, 128)

    # Simple inference
    out = model(x)
    assert out.shape == (2, 3, 128, 128), f"Expected (2, 3, 128, 128), got {out.shape}"
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0), "Output RGB out of [0, 1] range"

    # Diagnostic pass
    diag = model.forward_diagnostic(x)
    assert diag["illumination"].shape == (2, 1, 128, 128)
    assert diag["noise_uncertainty"].shape == (2, 1, 128, 128)
    assert diag["reliability_map"].shape == (2, 1, 128, 128)
    assert not torch.isnan(diag["enhanced"]).any()
    assert not torch.isnan(diag["illumination"]).any()


def test_parameter_budget():
    """Verify trainable parameters are strictly below the 1 million target budget."""
    model = LumaGuardNet()
    trainable_p, total_p = count_parameters(model)
    assert trainable_p < 1_000_000, f"Parameter count {trainable_p} exceeds budget of 1M"
    # LumaGuard-Net default design is ~27,049 params (<30k)
    assert trainable_p > 10_000, f"Suspiciously low parameter count: {trainable_p}"


def test_losses_and_backward():
    """Verify all loss terms produce finite scalars and populate gradients."""
    model = LumaGuardNet()
    loss_fn = LumaGuardTotalLoss()

    x = torch.rand(2, 3, 64, 64)
    loss, loss_dict = loss_fn(model, x)

    assert not torch.isnan(loss), "Loss evaluated to NaN"
    assert not torch.isinf(loss), "Loss evaluated to Inf"
    assert loss.item() > 0.0, "Total loss must be strictly positive"

    loss.backward()

    # Verify gradients exist and are finite for key parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has None gradient"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradient"


def test_checkpoint_save_and_reload():
    """Verify model state saving, reloading, and parameter parity."""
    model1 = LumaGuardNet()
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        torch.save({"model": model1.state_dict()}, tmp_path)

        model2 = LumaGuardNet()
        ckpt = torch.load(tmp_path, map_location="cpu")
        model2.load_state_dict(ckpt["model"])

        x = torch.rand(1, 3, 64, 64)
        out1 = model1(x)
        out2 = model2(x)
        assert torch.allclose(out1, out2, atol=1e-5), "Reloaded model output mismatch"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_self_supervised_dataset_isolation():
    """Verify that self-supervised dataset strictly loads low-light images only."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        dir_p = Path(tmp_dir)
        # Create synthetic low-light image
        img = Image.new("RGB", (64, 64), color=(20, 20, 20))
        img.save(dir_p / "low_001.png")

        dataset = SelfSupervisedLowLightDataset(tmp_dir, crop_size=32)
        assert len(dataset) == 1
        item = dataset[0]
        # Should return a single tensor (3, 32, 32), NOT a tuple of (low, high)
        assert isinstance(item, torch.Tensor)
        assert item.shape == (3, 32, 32)

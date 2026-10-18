"""Tests for paired RGB/chrominance restoration fine-tuning."""
import torch

from scripts.finetune_chrominance import paired_restoration_loss


def test_paired_restoration_loss_is_finite_and_backpropagates():
    prediction = torch.rand(2, 3, 48, 48, requires_grad=True)
    target = torch.rand(2, 3, 48, 48)

    loss, terms = paired_restoration_loss(prediction, target)
    assert torch.isfinite(loss)
    assert set(terms) == {
        "loss_total", "loss_rgb", "loss_luminance",
        "loss_chrominance", "loss_gradient", "loss_color_angle",
        "loss_contrast",
    }

    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_chrominance_weight_changes_total_loss():
    target = torch.full((1, 3, 32, 32), 0.5)
    prediction = target.clone()
    prediction[:, 0] = 0.85
    prediction.requires_grad_(True)

    low_chroma_weight, _ = paired_restoration_loss(
        prediction, target, w_chrominance=0.0
    )
    high_chroma_weight, _ = paired_restoration_loss(
        prediction, target, w_chrominance=3.0
    )

    assert torch.isfinite(low_chroma_weight)
    assert torch.isfinite(high_chroma_weight)
    assert high_chroma_weight > low_chroma_weight


def test_local_contrast_loss_is_finite_and_weighted():
    target = torch.rand(1, 3, 32, 32)
    prediction = torch.nn.functional.avg_pool2d(target, kernel_size=3, stride=1, padding=1)
    prediction.requires_grad_(True)

    loss_no_contrast, _ = paired_restoration_loss(
        prediction, target, w_contrast=0.0
    )
    loss_with_contrast, terms = paired_restoration_loss(
        prediction, target, w_contrast=1.0
    )

    assert torch.isfinite(terms["loss_contrast"])
    assert loss_with_contrast >= loss_no_contrast


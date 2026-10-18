import torch

from losses.paired_rgb_restoration import paired_rgb_restoration_loss
from models.paired_rgb_restoration_net import PairedRGBRestorationNet


def test_paired_rgb_restoration_net_preserves_shape_and_initial_identity():
    model = PairedRGBRestorationNet(base_channels=8)
    x = torch.rand(2, 3, 65, 71)
    y = model(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert y.min().item() >= 0.0
    assert y.max().item() <= 1.0
    # Zero-initialized logit residual yields identity up to numerical clamp tolerance.
    assert torch.max(torch.abs(y - x)).item() < 2e-4


def test_paired_rgb_loss_has_finite_gradients():
    model = PairedRGBRestorationNet(base_channels=8)
    x = torch.rand(2, 3, 64, 64)
    target = torch.rand_like(x)
    output = model(x)
    loss, terms = paired_rgb_restoration_loss(output, target)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in terms.values())
    loss.backward()
    assert model.rgb_residual.weight.grad is not None
    assert torch.isfinite(model.rgb_residual.weight.grad).all()

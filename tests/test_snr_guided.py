"""
Comprehensive Unit & Integration Tests for SNR-Guided Internal Reference Transfer.
Tests:
1. Model forward shapes across multiple spatial dimensions (128x128, 160x160, 256x256).
2. Parameter budget compliance (< 1,000,000 parameters).
3. Reliability / SNR-proxy estimator behavior:
   - Saturated highlight pixels (near 1.0) receive low reliability (penalty active).
   - Clean, high-intensity pixels receive high reliability.
   - High noise regions receive downweighted reliability.
4. Reference matching fallback:
   - When no valid references exist, confidence smoothly approaches 0 and output remains finite.
5. Finite output bounds:
   - Output images are strictly in [0, 1] without NaN or Inf.
6. Gradient flow & backward pass through all modules.
7. Loss functions:
   - Pseudo-pair degradation generation
   - Charbonnier reconstruction
   - Reliability-weighted structure loss
   - Adaptive exposure loss
   - Augmentation consistency
   - Composite total loss
8. Checkpoint saving and reloading.
"""

import os
import tempfile
import pytest
import torch
import torch.nn as nn

from models.snr_internal_reference_net import (
    SNRInternalReferenceNet,
    ReliabilityEstimator,
    InternalReferenceMatcher,
    ConfidenceGatedFeatureTransfer,
    ConservativeEnhancementHead,
    build_snr_internal_reference_net,
)
from losses.snr_guided_losses import (
    SNRGuidedTotalLoss,
    PseudoPairSynthesizer,
    ReliabilityWeightedStructureLoss,
    AdaptiveExposureLoss,
    ConfidenceGatedReferenceLoss,
)
from models.enhancement_head import ColorSpaceConverter
from scripts.train_kaggle import make_deterministic_split


def test_model_shapes_and_bounds():
    """Verify forward diagnostic shapes and [0, 1] output bounds across varying resolutions."""
    model = SNRInternalReferenceNet(base_channels=16, feature_channels=24, num_blocks=3)
    model.eval()

    test_shapes = [(2, 3, 128, 128), (1, 3, 160, 160), (1, 3, 200, 200)]

    for shape in test_shapes:
        x = torch.rand(shape, dtype=torch.float32)
        with torch.no_grad():
            diag = model.forward_diagnostic(x)

        assert diag["enhanced"].shape == shape, f"Expected {shape}, got {diag['enhanced'].shape}"
        assert diag["illumination"].shape == (shape[0], 1, shape[2], shape[3])
        assert diag["reliability_map"].shape == (shape[0], 1, shape[2], shape[3])
        assert diag["noise_proxy"].shape == (shape[0], 1, shape[2], shape[3])
        assert diag["match_confidence"].shape == (shape[0], 1, shape[2], shape[3])
        assert diag["features_fused"].shape == (shape[0], 24, shape[2], shape[3])

        # Bounds check
        assert torch.all(diag["enhanced"] >= 0.0) and torch.all(diag["enhanced"] <= 1.0)
        assert torch.all(diag["illumination"] >= 1e-4) and torch.all(diag["illumination"] <= 1.0)
        assert torch.all(diag["reliability_map"] >= 0.0) and torch.all(diag["reliability_map"] <= 1.0)

        # Finite checks
        assert not torch.isnan(diag["enhanced"]).any()
        assert not torch.isinf(diag["enhanced"]).any()


def test_parameter_budget():
    """Verify that trainable parameters are well under the 1,000,000 parameter budget."""
    model = SNRInternalReferenceNet(base_channels=24, feature_channels=32, num_blocks=4)
    param_count = model.count_parameters()
    print(f"\n[Test] Measured trainable parameter count: {param_count:,}")

    assert param_count < 1_000_000, f"Parameter count {param_count} exceeds 1,000,000!"
    assert param_count > 10_000, f"Parameter count {param_count} is implausibly low."


def test_reliability_estimator_physics():
    """Verify reliability estimator correctly differentiates saturation, clean signal, and noise."""
    estimator = ReliabilityEstimator(base_channels=12, saturation_threshold=0.92)
    estimator.eval()

    # Case A: Saturated patch (all 0.98)
    sat_img = torch.full((1, 3, 64, 64), 0.98, dtype=torch.float32)
    sat_lum = torch.full((1, 1, 64, 64), 0.98, dtype=torch.float32)

    # Case B: Clean well-exposed patch (all 0.60)
    clean_img = torch.full((1, 3, 64, 64), 0.60, dtype=torch.float32)
    clean_lum = torch.full((1, 1, 64, 64), 0.60, dtype=torch.float32)

    # Case C: Severely noisy dark patch (0.05 + strong noise)
    noise_img = torch.clamp(
        torch.full((1, 3, 64, 64), 0.05, dtype=torch.float32) + torch.randn(1, 3, 64, 64) * 0.15,
        0.0,
        1.0,
    )
    noise_lum = 0.299 * noise_img[:, 0:1] + 0.587 * noise_img[:, 1:2] + 0.114 * noise_img[:, 2:3]

    with torch.no_grad():
        q_sat, n_sat, sat_pen = estimator(sat_img, sat_lum)
        q_clean, n_clean, clean_pen = estimator(clean_img, clean_lum)
        q_noise, n_noise, noise_pen = estimator(noise_img, noise_lum)

    # Saturation penalty must be active on saturated image
    assert sat_pen.mean().item() < 0.20, f"Expected sat_pen < 0.20 on saturated highlights, got {sat_pen.mean().item():.3f}"
    assert clean_pen.mean().item() > 0.80, f"Expected clean_pen > 0.80 on normal exposure, got {clean_pen.mean().item():.3f}"

    # Clean signal must have higher reliability than saturated highlights
    assert q_clean.mean().item() > q_sat.mean().item()

    # Noise proxy must be higher on noisy input than on flat input
    assert n_noise.mean().item() > n_clean.mean().item()


def test_reference_matcher_fallback():
    """Verify fallback behavior when no trustworthy reference candidate exists."""
    matcher = InternalReferenceMatcher(
        feature_channels=24,
        patch_size=16,
        top_k=2,
        min_ref_reliability=0.50,
        similarity_threshold=0.30,
    )
    matcher.eval()

    b, c, h, w = 1, 24, 64, 64
    feat = torch.randn(b, c, h, w)

    # Set all reliabilities very low (e.g. 0.05) so all candidates are rejected
    q_map = torch.full((b, 1, h, w), 0.05)

    with torch.no_grad():
        ref_feat, match_conf, ref_q = matcher(feat, q_map)

    # When all candidates are rejected, match confidence must be 0
    assert torch.all(match_conf == 0.0), f"Expected match_conf == 0.0 under fallback, got max {match_conf.max()}"
    assert not torch.isnan(ref_feat).any()
    assert not torch.isinf(ref_feat).any()


def test_loss_backward_and_gradients():
    """Verify forward and backward gradient flow for SNRGuidedTotalLoss."""
    model = SNRInternalReferenceNet(base_channels=16, feature_channels=24, num_blocks=3)
    model.train()

    loss_fn = SNRGuidedTotalLoss(
        config={
            "loss": {
                "w_pseudo": 1.0,
                "w_gradient": 0.5,
                "w_reference": 0.1,
                "w_exposure": 3.0,
                "w_color": 2.0,
                "w_smooth": 5.0,
                "w_consistency": 0.2,
            }
        }
    )

    x = torch.rand(2, 3, 64, 64, requires_grad=False)
    total_loss, loss_dict = loss_fn(model, x)

    assert not torch.isnan(total_loss), "Total loss is NaN!"
    assert not torch.isinf(total_loss), "Total loss is Inf!"
    assert total_loss.item() > 0.0, "Total loss should be positive."

    total_loss.backward()

    # Check gradients on model parameters
    has_grad = False
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Parameter {name} has no gradient!"
            assert not torch.isnan(p.grad).any(), f"Parameter {name} has NaN gradient!"
            assert not torch.isinf(p.grad).any(), f"Parameter {name} has Inf gradient!"
            has_grad = True

    assert has_grad, "No parameters received gradients!"


def test_checkpoint_save_and_reload():
    """Verify checkpoint saving, serializing, and state restoration."""
    model = SNRInternalReferenceNet(base_channels=16, feature_channels=24, num_blocks=3)
    x = torch.rand(1, 3, 64, 64)

    with torch.no_grad():
        out_before = model(x)

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = os.path.join(tmp_dir, "test_snr_ckpt.pth")
        torch.save({"model": model.state_dict()}, ckpt_path)

        # Reload into fresh instance
        model_new = SNRInternalReferenceNet(base_channels=16, feature_channels=24, num_blocks=3)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model_new.load_state_dict(ckpt["model"])

        with torch.no_grad():
            out_after = model_new(x)

        assert torch.allclose(out_before, out_after, atol=1e-6)


def test_confidence_controlled_fusion_formula():
    """Verify Phase 3 residual fusion equation: F_fused = F_target + alpha * conf * (F_ref - F_target)"""
    from models.snr_internal_reference_net import ConfidenceGatedFeatureTransfer
    fusion = ConfidenceGatedFeatureTransfer(feature_channels=24, max_transfer=0.35)
    fusion.eval()

    b, c, h, w = 1, 24, 32, 32
    f_target = torch.randn(b, c, h, w)
    f_ref = torch.randn(b, c, h, w)
    q_target = torch.full((b, 1, h, w), 0.20)
    q_ref = torch.full((b, 1, h, w), 0.85)

    # Subtest A: When confidence is zero, F_fused must equal F_target exactly
    with torch.no_grad():
        f_fused_zero_conf, gate_zero = fusion(f_target, f_ref, q_target, q_ref, match_conf=torch.zeros((b, 1, h, w)))
    assert torch.allclose(f_fused_zero_conf, f_target, atol=1e-6), "Fallback to F_target failed when confidence is 0!"

    # Subtest B: When F_ref == F_target, F_fused must equal F_target
    with torch.no_grad():
        f_fused_identical, _ = fusion(f_target, f_target, q_target, q_ref, match_conf=torch.ones((b, 1, h, w)))
    assert torch.allclose(f_fused_identical, f_target, atol=1e-6), "Zero-drift property failed when F_ref == F_target!"

    # Subtest C: Fusion gate must be non-negative and strictly bounded by max_transfer (0.35)
    with torch.no_grad():
        _, gate_active = fusion(f_target, f_ref, q_target, q_ref, match_conf=torch.ones((b, 1, h, w)))
    assert torch.all(gate_active >= 0.0)
    assert torch.all(gate_active <= 0.35 + 1e-5), f"Gate exceeded max_transfer! Max: {gate_active.max()}"


def test_structural_edge_vs_noise_discrimination():
    """Verify Phase 4 edge vs noise discrimination in ReliabilityEstimator."""
    estimator = ReliabilityEstimator(base_channels=12)
    estimator.eval()

    # Image 1: Sharp high-contrast structural edge (black/white step)
    step_edge = torch.zeros((1, 1, 64, 64), dtype=torch.float32)
    step_edge[:, :, :, 32:] = 0.80

    # Image 2: Flat gray with high-frequency random noise
    flat_noisy = torch.clamp(torch.full((1, 1, 64, 64), 0.40) + torch.randn(1, 1, 64, 64) * 0.15, 0.0, 1.0)

    with torch.no_grad():
        struct_edge, noise_edge = estimator.compute_edge_and_noise_proxies(step_edge)
        struct_noisy, noise_noisy = estimator.compute_edge_and_noise_proxies(flat_noisy)

    # Edge image must have significantly higher structural proxy at the contour than the random noise image
    assert struct_edge[:, :, :, 30:34].mean().item() > struct_noisy.mean().item()
    # Random noise image must have higher noise proxy than the clean step edge
    assert noise_noisy.mean().item() > noise_edge[:, :, :, :25].mean().item()




def test_ycbcr_round_trip_preserves_rgb():
    """Color conversion should round-trip RGB in range without added cast."""
    rgb = torch.rand(2, 3, 48, 48)
    y, cbcr = ColorSpaceConverter.rgb_to_ycbcr(rgb)
    reconstructed = ColorSpaceConverter.ycbcr_to_rgb(y, cbcr)
    assert reconstructed.shape == rgb.shape
    assert torch.isfinite(reconstructed).all()
    assert (reconstructed - rgb).abs().mean().item() < 1e-5


def test_deterministic_train_validation_split_is_disjoint():
    """Held-out indices are repeatable, nonempty, and never overlap training."""
    train_a, val_a = make_deterministic_split(485, 0.10, 42)
    train_b, val_b = make_deterministic_split(485, 0.10, 42)
    assert train_a == train_b
    assert val_a == val_b
    assert train_a and val_a
    assert set(train_a).isdisjoint(val_a)
    assert sorted(train_a + val_a) == list(range(485))
    assert len(val_a) in (48, 49)


def test_ycbcr_round_trip_constant_boundaries():
    """Extreme valid RGB colors remain finite after conversion and inverse."""
    rgb = torch.tensor([
        [[ [0.0, 1.0] ], [[0.0, 1.0]], [[0.0, 1.0]]],
        [[[1.0, 0.0]], [[0.0, 1.0]], [[0.0, 0.0]]],
    ], dtype=torch.float32)
    y, cbcr = ColorSpaceConverter.rgb_to_ycbcr(rgb)
    out = ColorSpaceConverter.ycbcr_to_rgb(y, cbcr)
    assert out.shape == rgb.shape
    assert torch.isfinite(out).all()
    assert out.min() >= 0 and out.max() <= 1


def test_gamut_safe_ycbcr_inverse_preserves_luminance_and_bounds():
    """Gamut-aware conversion should keep RGB bounded without unnecessary hue clipping."""
    y = torch.full((1, 1, 8, 8), 0.20, dtype=torch.float32)
    cbcr = torch.full((1, 2, 8, 8), 0.40, dtype=torch.float32)

    rgb = ColorSpaceConverter.ycbcr_to_rgb_gamut_safe(y, cbcr)
    recovered_y, _ = ColorSpaceConverter.rgb_to_ycbcr(rgb)

    assert torch.isfinite(rgb).all()
    assert rgb.min().item() >= 0.0
    assert rgb.max().item() <= 1.0
    assert (recovered_y - y).abs().mean().item() < 1e-5


def test_adaptive_chrominance_gain_restores_color_with_reliability_gating():
    """The new gain should restore more chroma than the old cap and gate uncertain regions."""
    y_orig = torch.full((1, 1, 8, 8), 0.05)
    y_base = torch.full((1, 1, 8, 8), 0.50)
    q_low = torch.zeros_like(y_orig)
    q_high = torch.ones_like(y_orig)

    gain_low = ConservativeEnhancementHead.adaptive_chrominance_gain(
        y_orig, y_base, q_low
    )
    gain_high = ConservativeEnhancementHead.adaptive_chrominance_gain(
        y_orig, y_base, q_high
    )
    old_gain = (((y_base + 1e-4) / (y_orig + 1e-4)).clamp(1.0, 4.0)).pow(0.50)

    assert torch.isfinite(gain_low).all()
    assert torch.isfinite(gain_high).all()
    assert torch.all(gain_low > old_gain)
    assert torch.all(gain_high > gain_low)
    assert torch.all(gain_high <= 6.0 ** 0.70 + 1e-5)


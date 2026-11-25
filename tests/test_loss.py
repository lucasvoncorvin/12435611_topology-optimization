# Tests for loss functions
# test all loss functions

import pytest
import torch
from torch import nn

from src.loss import (
    BinaryMissmatchLoss,
    CustomLoss,
    FloatingMaterialLoss,
    LoadDiscrepancyLoss,
    ModularLoss,
    SinkhornDistance,
    VolumeFractionLoss,
)


def test_binary_missmatch_loss():
    pred = torch.tensor([0.0, 0.5, 1.0])
    assert BinaryMissmatchLoss()(pred).item() == pytest.approx(1 / 12)


def test_volume_fraction_loss():
    pred = torch.tensor([[0.25, 0.25], [0.75, 0.75]])
    target = torch.tensor([0.25, 0.5])
    assert VolumeFractionLoss()(pred, target).item() == 0.125


def test_volume_fraction_loss_pairs_column_targets_by_sample():
    fractions = torch.tensor([[0.25], [0.75]])
    assert VolumeFractionLoss()(fractions, fractions).item() == 0.0


def test_volume_fraction_loss_on_density_images():
    pred = torch.stack((torch.full((64, 64), 0.25), torch.full((64, 64), 0.75)))
    target = torch.tensor([[0.25], [0.5]])
    assert VolumeFractionLoss()(pred, target).item() == 0.125


def test_modular_loss_uses_requested_volume_fraction():
    from src.train import make_loss

    pred = torch.full((2, 64, 64), 0.5, requires_grad=True)
    target = torch.ones_like(pred)
    fractions = torch.tensor([[0.25], [0.5]])
    loss = make_loss("mse_bmvf_equal")(pred, target, target_volume_fraction=fractions)
    assert loss.item() == pytest.approx((0.25 + 0.25 + 0.125) / 3)
    loss.backward()
    assert torch.isfinite(pred.grad).all()


def test_load_discrepancy_loss():
    # Create a prediction with one cell activated near the load.
    pred = torch.zeros(1, 64 * 64)
    pred[0, 31] = 1.0  # One of the two cells touching the load.
    load = torch.tensor([[0.5, 0.0, 1.0, 0.0]])
    assert LoadDiscrepancyLoss(patch_size=64)(pred, load).item() == 0.5


def test_floating_material_loss():
    # Create a connected and an isolated material configuration for testing.
    connected = torch.zeros(1, 64 * 64)
    connected[0, 31] = 1.0  # Cell touching the load.
    isolated = connected.clone()
    connected[0, 30] = 1.0
    isolated[0, 32 * 64 + 32] = 1.0

    load = torch.tensor([[0.5, 0.0, 1.0, 0.0]])
    loss = FloatingMaterialLoss(patch_size=64, max_iteration=30)
    assert loss(isolated, load).item() > loss(connected, load).item()


def test_modular_loss_normalizes_weights():
    loss = ModularLoss(nn.ModuleList([nn.L1Loss(), nn.MSELoss()]), weights=[1, 3])
    assert loss(torch.tensor([0.5]), torch.tensor([0.0])).item() == 0.3125


def test_custom_loss_combines_base_and_auxiliary_losses():
    pred = torch.zeros(1, 64 * 64)
    target = torch.ones_like(pred)
    load = torch.tensor([[0.5, 0.0, 1.0, 0.0]])
    loss = CustomLoss(nn.L1Loss(), patch_size=64, auxiliary_loss_weight=0.25)
    expected = 1 + 0.25 * (0.5 + 1 / (64 * 64))
    assert loss(pred, target, load, torch.tensor([0.5])).item() == pytest.approx(expected)


def test_sinkhorn_loss():
    pred = torch.zeros(2, 64 * 64)
    pred[:, 0] = 1.0
    target = pred.clone()
    target[1, 0] = 0.0
    target[1, 4 * 8**2] = 1.0  # First cell of the fifth patch, at column 32.

    loss = SinkhornDistance(patch_size=8, reduction="none", check_every=1)
    expected = torch.tensor([0.0, (32 / 64) ** 2])
    torch.testing.assert_close(loss(pred, target), expected, atol=1e-3, rtol=0)

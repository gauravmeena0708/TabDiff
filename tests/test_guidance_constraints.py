import torch
from tabdiff.guidance import (
    Equality, GreaterThan, LessThan, NotEqual, Mean, Fraction,
)


def test_equality_loss_zero_at_target():
    c = Equality(idx=0, target=2.0, scale=1.0)
    x = torch.tensor([[2.0], [2.0]])
    assert torch.isclose(c.loss(x), torch.tensor(0.0))


def test_greater_than_penalizes_below_target():
    c = GreaterThan(idx=0, target=1.0, scale=1.0)
    below = torch.tensor([[0.0]])
    above = torch.tensor([[2.0]])
    assert c.loss(below).item() > 0.0
    assert c.loss(above).item() == 0.0


def test_less_than_penalizes_above_target():
    c = LessThan(idx=0, target=1.0, scale=1.0)
    assert c.loss(torch.tensor([[2.0]])).item() > 0.0
    assert c.loss(torch.tensor([[0.0]])).item() == 0.0


def test_not_equal_penalizes_within_margin():
    c = NotEqual(idx=0, target=0.0, margin=0.1, scale=1.0)
    assert c.loss(torch.tensor([[0.0]])).item() > 0.0   # exactly at target
    assert c.loss(torch.tensor([[1.0]])).item() == 0.0  # far away


def test_mean_loss_zero_when_batch_mean_hits_target():
    c = Mean(idx=0, target=1.0, scale=1.0)
    x = torch.tensor([[0.0], [2.0]])  # mean = 1.0
    assert torch.isclose(c.loss(x), torch.tensor(0.0))


def test_fraction_targets_batch_satisfaction_rate():
    c = Fraction(idx=0, target=0.0, target_fraction=0.5, direction='greater', tau=0.01, scale=1.0)
    x = torch.tensor([[5.0], [5.0], [-5.0], [-5.0]])  # ~50% satisfy v>0
    assert c.loss(x).item() < 1e-3

"""Constraint guidance for TabDiff (tabdiff-universal method).

Two channels, matching TabDiff's hybrid diffusion:
  * Numeric constraints operate on the EDM x0 estimate `denoised[:, idx]`.
  * Categorical constraints bias the MDLM unmasking logits `logits[:, col_pos, class_idx]`.

Loss math is ported from diffutabgen core/constraints.py; the spec dialect parser
is ported from diffutabgen methods/universal_guider.py (_split_constraint).
"""
import torch


# --- numeric constraints (operate on denoised[:, idx], normalized space) -----

class NumericConstraint:
    def __init__(self, idx, target, scale=1.0):
        self.idx = idx
        self.target = float(target)
        self.scale = float(scale)

    def loss(self, x):
        raise NotImplementedError


class Equality(NumericConstraint):
    def loss(self, x):
        return torch.mean((x[:, self.idx] - self.target) ** 2)


class GreaterThan(NumericConstraint):
    def loss(self, x):
        return torch.mean(torch.relu(self.target - x[:, self.idx]))


class LessThan(NumericConstraint):
    def loss(self, x):
        return torch.mean(torch.relu(x[:, self.idx] - self.target))


class NotEqual(NumericConstraint):
    def __init__(self, idx, target, margin=0.1, scale=1.0):
        super().__init__(idx, target, scale)
        self.margin = float(margin)

    def loss(self, x):
        dist = torch.abs(x[:, self.idx] - self.target)
        return torch.mean(torch.relu(self.margin - dist))


class Mean(NumericConstraint):
    def loss(self, x):
        return (torch.mean(x[:, self.idx]) - self.target) ** 2


class Fraction(NumericConstraint):
    def __init__(self, idx, target, target_fraction, direction='less', tau=0.1, scale=1.0):
        super().__init__(idx, target, scale)
        if not 0.0 <= float(target_fraction) <= 1.0:
            raise ValueError("target_fraction must be between 0 and 1")
        if direction not in {'less', 'greater'}:
            raise ValueError("direction must be 'less' or 'greater'")
        if tau <= 0:
            raise ValueError("tau must be positive")
        self.target_fraction = float(target_fraction)
        self.direction = direction
        self.tau = float(tau)

    def loss(self, x):
        v = x[:, self.idx]
        if self.direction == 'less':
            ind = torch.sigmoid((self.target - v) / self.tau)
        else:
            ind = torch.sigmoid((v - self.target) / self.tau)
        return (ind.mean() - self.target_fraction) ** 2

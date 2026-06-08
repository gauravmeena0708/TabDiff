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


# --- categorical constraints (bias logits[:, col_pos, class_idx]) ------------

class CategoricalConstraint:
    def __init__(self, col_pos, class_idx, scale=4.0, sign=1):
        if sign not in (1, -1):
            raise ValueError("sign must be +1 (equality) or -1 (not-equal)")
        self.col_pos = int(col_pos)
        self.class_idx = int(class_idx)
        self.scale = float(scale)
        self.sign = int(sign)


def compute_numeric_delta(denoised, num_constraints, m, lr):
    """m steps of plain gradient descent on Δ (init 0) minimizing
    Σ c.scale · c.loss(denoised + Δ). Returns Δ detached, same shape as denoised.

    Operates on TabDiff's native x0 estimate (`denoised`); the constraint then
    propagates through the unchanged EDM Euler step. Returns zeros when there is
    nothing to optimize or the gradient is non-finite.
    """
    if not num_constraints or m <= 0:
        return torch.zeros_like(denoised)

    base = denoised.detach()
    delta = torch.zeros_like(base)
    with torch.enable_grad():
        for _ in range(m):
            d = delta.detach().requires_grad_(True)
            total = 0
            for c in num_constraints:
                total = total + c.scale * c.loss(base + d)
            grad = torch.autograd.grad(total, d, allow_unused=True)[0]
            if grad is None:
                break
            if torch.isnan(grad).any() or torch.isinf(grad).any():
                return torch.zeros_like(base)
            delta = (d - lr * grad).detach()
    return delta


def apply_categorical_bias(logits, cat_constraints, w_t):
    """Add ±scale·w_t to logits[:, col_pos, class_idx] for each categorical
    constraint. logits shape is (bs, K, K_max) from _subs_parameterization.
    Returns a modified clone (does not mutate the input)."""
    if not cat_constraints:
        return logits
    out = logits.clone()
    for c in cat_constraints:
        out[:, c.col_pos, c.class_idx] = out[:, c.col_pos, c.class_idx] + c.sign * c.scale * w_t
    return out


def guidance_weight(schedule, i, num_timesteps):
    """Strength weight across the reverse loop. i runs T-1 (noisy) → 0 (clean).
    'none' is uniform; 'linear' ramps to full strength at the clean end."""
    if schedule == 'none':
        return 1.0
    if schedule == 'linear':
        denom = max(num_timesteps - 1, 1)
        return float((denom - i) / denom)
    raise ValueError(f"Unknown guidance schedule: {schedule!r}")

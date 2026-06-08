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


import numpy as np

_OPERATORS = [('>=', '>='), ('<=', '<='), ('!=', '!='), ('>', '>'), ('<', '<'), ('=', '=')]


def _split_constraint(spec):
    """Split 'col=val' / 'col>=val' into (col, op, val). Leftmost operator wins,
    ties broken toward the longer token ('>=' over '>'). Ported verbatim from
    diffutabgen universal_guider._split_constraint."""
    best_idx = len(spec)
    best_token = None
    best_op = None
    for token, op in _OPERATORS:
        idx = spec.find(token)
        if idx != -1 and (idx < best_idx or (idx == best_idx and len(token) > len(best_token))):
            best_idx = idx
            best_token = token
            best_op = op
    if best_token:
        col, val = spec.split(best_token, 1)
        return col, best_op, val
    raise ValueError(f"Unknown operator in constraint: {spec}")


def _cat_col_pos(info, col_idx):
    """Categorical position in num_classes order (mirrors _engine.generate_multi_conditional)."""
    if col_idx in info["cat_col_idx"]:
        pos = info["cat_col_idx"].index(col_idx)
        if info["task_type"] != "regression":
            pos += 1
        return pos
    if col_idx in info.get("target_col_idx", []):
        return 0
    raise ValueError(f"Column index {col_idx} is not categorical or target")


def _class_index(classes, value):
    classes_str = [str(c).strip() for c in classes]
    v = str(value).strip()
    if v not in classes_str:
        raise ValueError(f"Value {value!r} is not a known class. Known: {classes_str}")
    return classes_str.index(v)


def _normalize_numeric(info, X_num_train, num_idx, raw_value, num_transform, int_transform):
    """Normalize a raw numeric target into denoised space, the same way
    _engine.generate_multi_conditional does (mean row → int → num transform)."""
    base_row = X_num_train.mean(axis=0).astype(np.float32)
    row = base_row.copy()
    row[num_idx] = float(raw_value)
    row = row.reshape(1, -1)
    if int_transform is not None:
        row = int_transform.transform(row)
    if num_transform is not None:
        row = num_transform.transform(row)
    return float(row[0, num_idx])


def parse_constraint_spec(spec, info, X_num_train, num_transform, int_transform,
                          num_scale=0.1, cat_scale=4.0, mean_scale=0.1,
                          not_equal_margin=0.1, fraction_tau=0.1):
    """Parse one constraint spec into a NumericConstraint or CategoricalConstraint."""
    # @scale=N suffix
    per_spec_scale = None
    if '@scale=' in spec:
        spec, scale_str = spec.rsplit('@scale=', 1)
        per_spec_scale = float(scale_str.strip())

    # ~p fraction suffix (numeric inequalities only)
    frac_target = None
    if '~' in spec:
        spec, frac_str = spec.rsplit('~', 1)
        frac_target = float(frac_str.strip())

    col, op, val = _split_constraint(spec)
    col = col.strip()

    is_mean = False
    if col.startswith('mean(') and col.endswith(')'):
        col = col[5:-1]
        is_mean = True

    if col not in info["column_names"]:
        raise ValueError(f"Column {col!r} not found. Known: {info['column_names']}")
    col_idx = info["column_names"].index(col)
    is_numeric = col_idx in info["num_col_idx"]

    if not is_numeric:
        if is_mean:
            raise ValueError(f"mean({col}) is only valid for numeric columns")
        if op not in ('=', '!='):
            raise ValueError(f"Operator {op!r} is not valid on categorical column {col!r}")
        col_pos = _cat_col_pos(info, col_idx)
        class_idx = _class_index(info["cat_encoders"][col], val)
        scale = per_spec_scale if per_spec_scale is not None else cat_scale
        sign = 1 if op == '=' else -1
        return CategoricalConstraint(col_pos, class_idx, scale=scale, sign=sign)

    # numeric
    num_idx = info["num_col_idx"].index(col_idx)
    target = _normalize_numeric(info, X_num_train, num_idx, val, num_transform, int_transform)
    scale = per_spec_scale if per_spec_scale is not None else (mean_scale if is_mean else num_scale)

    if is_mean:
        return Mean(num_idx, target, scale=scale)
    if frac_target is not None:
        if op not in ('>', '>=', '<', '<='):
            raise ValueError("~fraction targets are only valid on inequalities")
        direction = 'greater' if op in ('>', '>=') else 'less'
        return Fraction(num_idx, target, frac_target, direction=direction, tau=fraction_tau, scale=scale)
    if op == '=':
        return Equality(num_idx, target, scale=scale)
    if op in ('>', '>='):
        return GreaterThan(num_idx, target, scale=scale)
    if op in ('<', '<='):
        return LessThan(num_idx, target, scale=scale)
    if op == '!=':
        return NotEqual(num_idx, target, margin=not_equal_margin, scale=scale)
    raise ValueError(f"Unhandled operator {op!r}")


def snap_categorical(x_cat, cat_constraints):
    """Hard-set equality-constrained categorical columns to their target class.
    Returns a modified clone; not-equal constraints (sign=-1) are left alone."""
    if not cat_constraints:
        return x_cat
    out = x_cat.clone()
    for c in cat_constraints:
        if c.sign == 1:
            out[:, c.col_pos] = c.class_idx
    return out

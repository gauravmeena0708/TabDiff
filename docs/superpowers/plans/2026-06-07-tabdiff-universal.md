# tabdiff-universal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `tabdiff-universal` — a training-free constraint sampler that enforces the native universal dialect (`=`, `!=`, `>`, `<`, `>=`, `<=`, `mean()`, `~p` fractions; numeric + categorical) on TabDiff's strong model, exposed through the `sdcontract` plugin.

**Architecture:** Approach C — native two-channel guidance. Numeric constraints guide the EDM **x0 estimate** (`denoised`); categorical constraints bias the MDLM **unmasking logits**. All testable logic lives in a new `tabdiff/guidance.py`; the model gets two **appended** methods (`_edm_update_guided`, `sample_guided`) that are thin copies calling those helpers, leaving `edm_update`/`sample`/`sample_impute`/`sample_all` byte-for-byte unchanged. The engine gets a `generate_guided` path; the contract registers a `tabdiff-universal` variant.

**Tech Stack:** Python, PyTorch, pytest. Repo: `sub/TabDiff`, branch `feat/tabdiff-universal`. Spec: `docs/superpowers/specs/2026-06-07-tabdiff-universal-design.md`. Conda env: `diffutabgen`.

---

## File Structure

**New files**
- `tabdiff/guidance.py` — constraint classes, spec parser, and pure guidance helpers (`compute_numeric_delta`, `apply_categorical_bias`, `guidance_weight`).
- `tests/conftest.py` — adds repo root to `sys.path` (TabDiff has no test infra yet).
- `tests/test_guidance_constraints.py` — numeric/categorical constraint loss math.
- `tests/test_guidance_helpers.py` — `compute_numeric_delta`, `apply_categorical_bias`, `guidance_weight`.
- `tests/test_guidance_parser.py` — `parse_constraint_spec` against a fake `info`.
- `tests/test_sample_guided.py` — tiny constructed model, end-to-end behavior.

**Edited files (additive only; existing code paths unchanged)**
- `tabdiff/models/unified_ctime_diffusion.py` — append `_edm_update_guided` + `sample_guided`.
- `sdcontract/_engine.py` — append `generate_guided`; add one dispatch branch in `generate()`.
- `sdcontract/generate` — add `"universal"` to a passthrough-modes set.
- `sdcontract/meta.json`, `sdcontract/capabilities.json` — register the variant.

**Default guidance constants (calibration starting points; §"Open items" in spec)**
`num_scale=0.1`, `mean_scale=0.1`, `cat_scale (logit β)=4.0`, `backward_steps m=10`, `backward_lr=1.0`, `guidance_schedule='none'`, `not_equal_margin=0.1`, `fraction_tau=0.1`.

---

## Task 1: Test scaffolding

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/test_smoke.py`

- [ ] **Step 1: Write conftest that puts the repo root on sys.path**

```python
# tests/conftest.py
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
```

- [ ] **Step 2: Write a trivial smoke test**

```python
# tests/test_smoke.py
def test_import_tabdiff_package():
    import tabdiff  # noqa: F401
```

- [ ] **Step 3: Run it**

Run: `conda activate diffutabgen && cd sub/TabDiff && python -m pytest tests/test_smoke.py -v`
Expected: PASS (1 passed).

- [ ] **Step 4: Commit**

```bash
git add tests/conftest.py tests/test_smoke.py
git commit -m "test: add pytest scaffolding for TabDiff"
```

---

## Task 2: Numeric constraint classes

**Files:**
- Create: `tabdiff/guidance.py`
- Test: `tests/test_guidance_constraints.py`

Loss math is ported from `sub/diffutabgen/core/constraints.py` (verbatim semantics), specialized to a single column index into `denoised`. Categorical equality's "extremize margin" logic is NOT ported — categoricals use logit-space guidance (Task 4), not these classes.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_guidance_constraints.py
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
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_guidance_constraints.py -v`
Expected: FAIL (ImportError: cannot import name 'Equality').

- [ ] **Step 3: Implement the constraint classes**

```python
# tabdiff/guidance.py
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
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_guidance_constraints.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add tabdiff/guidance.py tests/test_guidance_constraints.py
git commit -m "feat(guidance): numeric constraint loss classes"
```

---

## Task 3: Categorical constraint class

**Files:**
- Modify: `tabdiff/guidance.py`
- Test: `tests/test_guidance_constraints.py`

A categorical constraint is applied by adding a bias to one logit entry (not via a differentiable loss). `col_pos` is the categorical column's position in `num_classes` order; `class_idx` is the local class index; `sign` is `+1` for equality (boost) or `-1` for not-equal (suppress).

- [ ] **Step 1: Add failing tests**

```python
# append to tests/test_guidance_constraints.py
from tabdiff.guidance import CategoricalConstraint


def test_categorical_constraint_fields():
    c = CategoricalConstraint(col_pos=1, class_idx=2, scale=4.0, sign=1)
    assert c.col_pos == 1 and c.class_idx == 2
    assert c.scale == 4.0 and c.sign == 1


def test_categorical_constraint_rejects_bad_sign():
    import pytest
    with pytest.raises(ValueError):
        CategoricalConstraint(col_pos=0, class_idx=0, scale=1.0, sign=0)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_guidance_constraints.py -k categorical -v`
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement**

```python
# append to tabdiff/guidance.py

# --- categorical constraints (bias logits[:, col_pos, class_idx]) ------------

class CategoricalConstraint:
    def __init__(self, col_pos, class_idx, scale=4.0, sign=1):
        if sign not in (1, -1):
            raise ValueError("sign must be +1 (equality) or -1 (not-equal)")
        self.col_pos = int(col_pos)
        self.class_idx = int(class_idx)
        self.scale = float(scale)
        self.sign = int(sign)
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_guidance_constraints.py -v`
Expected: PASS (8 passed).

- [ ] **Step 5: Commit**

```bash
git add tabdiff/guidance.py tests/test_guidance_constraints.py
git commit -m "feat(guidance): categorical constraint class"
```

---

## Task 4: Guidance helpers (pure functions)

**Files:**
- Modify: `tabdiff/guidance.py`
- Test: `tests/test_guidance_helpers.py`

Three pure helpers the model methods call. `compute_numeric_delta` mirrors `universal_guider._compute_backward_delta` (m steps of GD on Δ over the x0 estimate). `apply_categorical_bias` adds the logit bias. `guidance_weight` anneals strength across the reverse loop (i = T-1 noisy … 0 clean).

- [ ] **Step 1: Write failing tests**

```python
# tests/test_guidance_helpers.py
import torch
from tabdiff.guidance import (
    GreaterThan, CategoricalConstraint,
    compute_numeric_delta, apply_categorical_bias, guidance_weight,
)


def test_compute_numeric_delta_moves_toward_satisfaction():
    denoised = torch.zeros(4, 2)
    c = GreaterThan(idx=0, target=1.0, scale=0.1)
    delta = compute_numeric_delta(denoised, [c], m=10, lr=1.0)
    # column 0 should be pushed up; column 1 untouched
    assert delta[:, 0].mean().item() > 0.0
    assert torch.allclose(delta[:, 1], torch.zeros(4))


def test_compute_numeric_delta_empty_is_zero():
    denoised = torch.randn(3, 2)
    delta = compute_numeric_delta(denoised, [], m=10, lr=1.0)
    assert torch.allclose(delta, torch.zeros_like(denoised))


def test_apply_categorical_bias_boosts_target_logit():
    logits = torch.zeros(2, 3, 5)  # (bs, K=3 cols, K_max=5)
    c = CategoricalConstraint(col_pos=1, class_idx=2, scale=4.0, sign=1)
    out = apply_categorical_bias(logits, [c], w_t=1.0)
    assert out[0, 1, 2].item() == 4.0
    assert out[0, 0, 0].item() == 0.0  # other entries untouched


def test_apply_categorical_bias_suppresses_for_not_equal():
    logits = torch.zeros(2, 3, 5)
    c = CategoricalConstraint(col_pos=0, class_idx=1, scale=4.0, sign=-1)
    out = apply_categorical_bias(logits, [c], w_t=0.5)
    assert out[0, 0, 1].item() == -2.0


def test_guidance_weight_modes():
    assert guidance_weight('none', i=5, num_timesteps=10) == 1.0
    assert guidance_weight('linear', i=0, num_timesteps=10) == 1.0   # cleanest step
    assert guidance_weight('linear', i=9, num_timesteps=10) == 0.0   # noisiest step
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_guidance_helpers.py -v`
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement the helpers**

```python
# append to tabdiff/guidance.py

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
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_guidance_helpers.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add tabdiff/guidance.py tests/test_guidance_helpers.py
git commit -m "feat(guidance): numeric delta + categorical bias + schedule helpers"
```

---

## Task 5: Spec parser

**Files:**
- Modify: `tabdiff/guidance.py`
- Test: `tests/test_guidance_parser.py`

`_split_constraint` is ported verbatim from `sub/diffutabgen/methods/universal_guider.py` (lines 31-67). `parse_constraint_spec` resolves a column against TabDiff `info`, normalizes numeric targets the same way `_engine.generate_multi_conditional` does (mean-row substitution → `int_transform` → `num_transform`), and resolves categorical targets to `(col_pos, class_idx)`.

`col_pos` mapping (mirrors `_engine.generate_multi_conditional` lines 209-218): for a categorical column, `col_pos = cat_col_idx.index(col_idx)`, plus 1 when `task_type != "regression"` (the target occupies categorical position 0); a target column maps to `col_pos = 0`. `num_idx` for a numeric column is `num_col_idx.index(col_idx)`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_guidance_parser.py
import numpy as np
import pytest
from tabdiff.guidance import parse_constraint_spec, GreaterThan, Mean, Equality, CategoricalConstraint


class _FakeTransform:
    """Identity transform with the sklearn-style .transform(2d) API."""
    def transform(self, row):
        return row


def _fake_info():
    # 2 numeric (age, balance), 1 categorical (education) + a target (income).
    return {
        "column_names": ["age", "balance", "education", "income"],
        "num_col_idx": [0, 1],
        "cat_col_idx": [2],
        "target_col_idx": [3],
        "task_type": "binclass",
        "cat_encoders": {
            "education": ["HS", "Bachelors", "Masters"],
            "income": ["<=50K", ">50K"],
        },
    }


def _parse(spec, **kw):
    info = _fake_info()
    X_num_train = np.zeros((10, 2), dtype=np.float32)
    return parse_constraint_spec(
        spec, info, X_num_train,
        num_transform=_FakeTransform(), int_transform=_FakeTransform(),
        **kw,
    )


def test_parses_numeric_greater_than():
    c = _parse("age>30")
    assert isinstance(c, GreaterThan)
    assert c.idx == 0 and c.target == 30.0


def test_parses_mean_aggregate():
    c = _parse("mean(balance)=1000")
    assert isinstance(c, Mean)
    assert c.idx == 1 and c.target == 1000.0


def test_parses_per_spec_scale_override():
    c = _parse("age>30@scale=0.5")
    assert c.scale == 0.5


def test_parses_categorical_equality_with_target_offset():
    # education is cat_col_idx[0]; task is binclass so col_pos = 0 + 1 = 1.
    c = _parse("education=Bachelors")
    assert isinstance(c, CategoricalConstraint)
    assert c.col_pos == 1 and c.class_idx == 1 and c.sign == 1


def test_parses_target_column_at_position_zero():
    c = _parse("income=>50K")
    assert isinstance(c, CategoricalConstraint)
    assert c.col_pos == 0 and c.class_idx == 1


def test_parses_categorical_not_equal():
    c = _parse("education!=HS")
    assert c.sign == -1 and c.class_idx == 0


def test_numeric_op_on_categorical_raises():
    with pytest.raises(ValueError):
        _parse("education>2")


def test_unknown_column_raises():
    with pytest.raises(ValueError):
        _parse("nope=1")
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_guidance_parser.py -v`
Expected: FAIL (ImportError: cannot import name 'parse_constraint_spec').

- [ ] **Step 3: Implement `_split_constraint` and `parse_constraint_spec`**

```python
# append to tabdiff/guidance.py
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
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_guidance_parser.py -v`
Expected: PASS (8 passed).

- [ ] **Step 5: Commit**

```bash
git add tabdiff/guidance.py tests/test_guidance_parser.py
git commit -m "feat(guidance): constraint spec parser"
```

---

## Task 6: Append `_edm_update_guided` + `sample_guided` to the model

**Files:**
- Modify: `tabdiff/models/unified_ctime_diffusion.py`

**Isolation rule:** do NOT edit `edm_update`, `sample`, `sample_impute`, or `sample_all`. Append two new methods after `sample_all` (after line 236). Verify with `git diff` that the only changes are additions.

- [ ] **Step 1: Add the import at the top of the file**

At the top of `tabdiff/models/unified_ctime_diffusion.py`, after the existing imports, add:

```python
from tabdiff.guidance import (
    compute_numeric_delta, apply_categorical_bias, guidance_weight,
)
```

(The `NumericConstraint`/`CategoricalConstraint` split happens in the engine, Task 8 — the model methods receive pre-split lists, so they are not imported here.)

- [ ] **Step 2: Append `_edm_update_guided`**

Copy the body of `edm_update` (lines 405-505) verbatim into a new method `_edm_update_guided` with this signature (adds 6 trailing params), then insert the two hook blocks at the marked anchors. Append after `sample_all` (after line 236):

```python
    def _edm_update_guided(
            self, x_num_cur, x_cat_cur, i,
            t_cur, t_next, t_hat,
            sigma_num_cur, sigma_num_next, sigma_num_hat,
            sigma_cat_cur, sigma_cat_next, sigma_cat_hat,
            cat_temperature=1.0,
            num_constraints=(), cat_constraints=(),
            backward_steps=10, backward_lr=1.0, w_t=1.0,
        ):
        """Copy of edm_update with two guidance hooks. CFG (y_only_model) path is
        unsupported here: guidance + CFG interaction is untested (spec §10)."""
        # <PASTE lines 415-432 of edm_update verbatim: through the first
        #  `denoised, raw_logits = self._denoise_fn(...)` call.>

        # === HOOK A: numeric guidance on the x0 estimate ===
        if num_constraints:
            delta = compute_numeric_delta(denoised, list(num_constraints),
                                          m=backward_steps, lr=backward_lr * w_t)
            denoised = denoised + delta
        # === end HOOK A ===

        # <PASTE the CFG block (lines 433-460) verbatim. NOTE: this method asserts
        #  cfg is off (see sample_guided), so this block is a no-op; keep it for
        #  a verbatim diff against edm_update.>

        # Euler step (verbatim lines 462-464)
        d_cur = (x_num_hat - denoised) / sigma_num_hat
        x_num_next = x_num_hat + (sigma_num_next - sigma_num_hat) * d_cur

        # Unmasking (verbatim lines 466-473), with HOOK B inserted before _mdlm_update:
        x_cat_next = x_cat_cur
        q_xs = torch.zeros_like(x_cat_cur).float()
        if has_cat:
            logits = self._subs_parameterization(raw_logits, x_cat_hat)
            # === HOOK B: categorical guidance on the unmasking logits ===
            if cat_constraints:
                logits = apply_categorical_bias(logits, list(cat_constraints), w_t=w_t)
            # === end HOOK B ===
            alpha_t = torch.exp(-sigma_cat_hat).unsqueeze(0).repeat(b, 1)
            alpha_s = torch.exp(-sigma_cat_next).unsqueeze(0).repeat(b, 1)
            x_cat_next, q_xs = self._mdlm_update(logits, x_cat_hat, alpha_t, alpha_s, temperature=cat_temperature)

        # <PASTE the 2nd-order correction block (lines 475-503) verbatim.>

        return x_num_next, x_cat_next, q_xs
```

Note: `b`, `has_cat`, `x_num_hat`, `x_cat_hat`, `x_cat_hat_oh`, `denoised`, `raw_logits` all come from the pasted verbatim prefix (lines 415-432), exactly as in `edm_update`.

- [ ] **Step 3: Append `sample_guided`**

Copy `sample` (lines 155-213) verbatim into `sample_guided`, changing only: the signature, an assert that CFG is off, and the `edm_update` call → `_edm_update_guided` with the guidance args.

```python
    def sample_guided(self, num_samples, num_constraints=(), cat_constraints=(),
                      backward_steps=10, backward_lr=1.0, guidance_schedule='none'):
        """Unconditional EDM/MDLM sampling (copy of self.sample) with per-step
        constraint guidance. Pure guidance: no known-column blending."""
        assert self.y_only_model is None, "tabdiff-universal does not support the CFG (y_only_model) path"
        # <PASTE lines 156-199 of `sample` verbatim: through the z_cat prior init.>

        pbar = tqdm(reversed(range(0, self.num_timesteps)), total=self.num_timesteps)
        pbar.set_description("Guided Sampling Progress")
        for i in pbar:
            w_t = guidance_weight(guidance_schedule, i, self.num_timesteps)
            z_norm, z_cat, q_xs = self._edm_update_guided(
                z_norm, z_cat, i,
                t[i], t[i - 1] if i > 0 else None, t_hat[i],
                sigma_num_cur[i], sigma_num_next[i], sigma_num_hat[i],
                sigma_cat_cur[i], sigma_cat_next[i], sigma_cat_hat[i],
                num_constraints=num_constraints, cat_constraints=cat_constraints,
                backward_steps=backward_steps, backward_lr=backward_lr, w_t=w_t,
            )

        assert torch.all(z_cat < self.mask_index)
        sample = torch.cat([z_norm, z_cat], dim=1).cpu()
        return sample
```

- [ ] **Step 4: Verify isolation — original methods unchanged**

Run: `git diff tabdiff/models/unified_ctime_diffusion.py | grep '^-' | grep -v '^---'`
Expected: NO output (no lines removed/changed — additions only).

- [ ] **Step 5: Verify the file imports cleanly**

Run: `python -c "from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion; print('ok')"`
Expected: prints `ok` (no ImportError / SyntaxError).

- [ ] **Step 6: Commit**

```bash
git add tabdiff/models/unified_ctime_diffusion.py
git commit -m "feat(model): append _edm_update_guided + sample_guided (isolated)"
```

---

## Task 7: End-to-end behavior test on a tiny constructed model

**Files:**
- Test: `tests/test_sample_guided.py`

Builds a minimal `UnifiedCtimeDiffusion` with a trivial denoise function (no categoricals, so the categorical channel is skipped) to prove numeric guidance raises the generated column vs. unguided `sample`. This needs no trained checkpoint.

- [ ] **Step 1: Write the test**

```python
# tests/test_sample_guided.py
import numpy as np
import torch
import pytest

from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
from tabdiff.guidance import GreaterThan


def _build_tiny_model():
    """2 numeric features, no categoricals. Denoise fn returns x_num unchanged
    (identity denoiser) so sampling is well-defined and cheap."""
    def denoise_fn(x_num, x_cat_oh, t, sigma=None):
        # returns (denoised_num, raw_logits); no categoricals → empty logits
        return x_num, torch.zeros(x_num.shape[0], 0, device=x_num.device)

    model = UnifiedCtimeDiffusion(
        num_classes=np.array([], dtype=int),
        num_numerical_features=2,
        denoise_fn=denoise_fn,
        y_only_model=None,
        num_timesteps=20,
        scheduler='power_mean',
        cat_scheduler='log_linear',
        noise_dist='uniform_t',
        edm_params={'sigma_data': 1.0},
        noise_schedule_params={},
        sampler_params={'stochastic_sampler': False, 'second_order_correction': False,
                        'S_churn': 0, 'S_min': 0, 'S_max': float('inf'), 'S_noise': 1},
        device=torch.device('cpu'),
    )
    return model


def test_sample_guided_raises_constrained_column():
    torch.manual_seed(0)
    model = _build_tiny_model()

    unguided = model.sample(64)
    torch.manual_seed(0)
    guided = model.sample_guided(
        64, num_constraints=[GreaterThan(idx=0, target=3.0, scale=0.5)],
        backward_steps=10, backward_lr=1.0,
    )
    # Guided column-0 mean should exceed the unguided mean.
    assert guided[:, 0].mean().item() > unguided[:, 0].mean().item()


def test_sample_guided_no_constraints_matches_sample():
    torch.manual_seed(0)
    a = _build_tiny_model().sample(16)
    torch.manual_seed(0)
    b = _build_tiny_model().sample_guided(16)
    assert torch.allclose(a, b, atol=1e-5)
```

- [ ] **Step 2: Run it**

Run: `python -m pytest tests/test_sample_guided.py -v`
Expected: PASS (2 passed). If the tiny-model constructor signature drifts from the real `__init__`, adjust the fixture kwargs to match `UnifiedCtimeDiffusion.__init__` (lines 19-93) — do not change the assertions.

- [ ] **Step 3: Commit**

```bash
git add tests/test_sample_guided.py
git commit -m "test(model): sample_guided raises constrained numeric column"
```

---

## Task 8: Engine `generate_guided` + dispatch

**Files:**
- Modify: `sdcontract/_engine.py`

Add `generate_guided`, parallel to `generate_multi_conditional`, and dispatch to it from `generate()` when the variant is `"universal"`. Existing engine functions are untouched.

- [ ] **Step 1: Add `generate_guided` after `generate_unconditional` (after line 287)**

```python
def generate_guided(
    dataname="adult",
    constraint_specs=None,   # list of native dialect strings, e.g. ["age>30", "education=Bachelors"]
    num_samples=100,
    ckpt_path=None,
    device="cuda",
    num_inference_steps=None,
    num_scale=0.1, cat_scale=4.0, mean_scale=0.1,
    backward_steps=10, backward_lr=1.0, guidance_schedule="none",
):
    """tabdiff-universal: training-free constraint guidance over TabDiff's model.
    Numeric constraints guide the x0 estimate; categorical constraints bias the
    unmasking logits. Falls back to nothing-to-guide → unconditional handled by caller."""
    from tabdiff.guidance import (
        NumericConstraint, CategoricalConstraint, parse_constraint_spec,
    )

    original_cwd = os.getcwd()
    os.chdir(TABDIFF_DIR)
    try:
        from generate_conditional import load_model_and_info

        device = _resolve_device(device)
        (
            diffusion, info, X_num_train, _Xc, _dn, _cats,
            num_inverse, int_inverse, cat_inverse, num_transform, int_transform,
        ) = load_model_and_info(dataname, ckpt_path=ckpt_path, device=device)

        if num_inference_steps is not None:
            diffusion.num_timesteps = num_inference_steps

        parsed = [
            parse_constraint_spec(
                s, info, X_num_train, num_transform, int_transform,
                num_scale=num_scale, cat_scale=cat_scale, mean_scale=mean_scale,
            )
            for s in (constraint_specs or [])
        ]
        num_constraints = [c for c in parsed if isinstance(c, NumericConstraint)]
        cat_constraints = [c for c in parsed if isinstance(c, CategoricalConstraint)]
        logger.info("Guided generation: %d numeric, %d categorical constraints",
                    len(num_constraints), len(cat_constraints))

        with torch.no_grad():
            syn_data = diffusion.sample_guided(
                num_samples,
                num_constraints=num_constraints, cat_constraints=cat_constraints,
                backward_steps=backward_steps, backward_lr=backward_lr,
                guidance_schedule=guidance_schedule,
            )
        return custom_decode_synthetic_data(syn_data, info, num_inverse, int_inverse, cat_inverse)
    finally:
        os.chdir(original_cwd)
```

- [ ] **Step 2: Add the dispatch branch in `generate()`**

In `generate()` (the contract entry point), the `privacy_mode` argument carries the variant. Add a branch BEFORE the existing `if use_unconditional:` block (replace lines 500-513). The new logic:

```python
    # tabdiff-universal: full native dialect via guidance.
    if privacy_mode == "universal":
        if not native_constraints:
            df = generate_unconditional(dataname=hashed, num_samples=n_samples,
                                        ckpt_path=ckpt_path, device=device,
                                        num_inference_steps=num_inference_steps)
        else:
            df = generate_guided(
                dataname=hashed, constraint_specs=list(native_constraints),
                num_samples=n_samples, ckpt_path=ckpt_path, device=device,
                num_inference_steps=num_inference_steps,
            )
    else:
        # Unconditional only when there is genuinely nothing to condition on.
        use_unconditional = (privacy_mode == "none" and not constraints)
        if use_unconditional:
            df = generate_unconditional(dataname=hashed, num_samples=n_samples,
                                        ckpt_path=ckpt_path, device=device,
                                        num_inference_steps=num_inference_steps)
        else:
            df = generate_multi_conditional(
                dataname=hashed, constraints=constraints, num_samples=n_samples,
                s_churn=s_churn, privacy_noise_scale=privacy_noise_scale,
                cat_noise_scale=cat_noise_scale, impute_condition="x_t",
                ckpt_path=ckpt_path, device=device, num_inference_steps=num_inference_steps,
            )
```

Note: `generate_guided` receives the raw native dialect strings (`native_constraints`), NOT the `(col, val)` equality tuples that `generate_multi_conditional` parses. The `s_churn`/`privacy_noise_scale`/`cat_noise_scale`/`constraints` setup above the branch (lines 486-498) stays as-is and is simply unused on the universal path.

- [ ] **Step 3: Verify the module imports**

Run: `cd sub/TabDiff/sdcontract && python -c "import _engine; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 4: Commit**

```bash
git add sdcontract/_engine.py
git commit -m "feat(engine): generate_guided path for tabdiff-universal"
```

---

## Task 9: Contract wiring

**Files:**
- Modify: `sdcontract/generate`
- Modify: `sdcontract/meta.json`
- Modify: `sdcontract/capabilities.json`

The `generate` endpoint currently round-trips ALL constraints through `to_spec(parse(s), "equality")`, which rejects non-equality. For the universal variant we must pass the dialect through raw (mirrors diffutabgen's `_PASSTHROUGH_MODES`).

- [ ] **Step 1: Edit `sdcontract/generate` to pass the universal dialect through raw**

Replace the body of `main()` (lines 25-33) with:

```python
        from sdcontract_core.constraints import parse, to_spec
        import _engine

        raw = req.get("constraints", [])
        variant = req.get("variant") or "none"
        # Universal variant consumes the shared dialect raw (>, mean(...), ~p, !=);
        # all other variants are equality-only (col=val).
        if variant == "universal":
            native = list(raw)
        else:
            native = [to_spec(parse(s), "equality") for s in raw]
        out_path = _engine.generate(req, native, variant)
        result = {"ok": True, "output_csv": str(out_path), "message": ""}
```

- [ ] **Step 2: Register the variant in `meta.json`**

In `sdcontract/meta.json`, add to the `"methods"` object:

```json
    "tabdiff-universal": { "variant": "universal", "enabled": true },
```

- [ ] **Step 3: Register the variant in `capabilities.json`**

In `sdcontract/capabilities.json`, add to the `"methods"` object:

```json
    "tabdiff-universal": { "variant": "universal", "enabled": true },
```

- [ ] **Step 4: Validate JSON**

Run: `cd sub/TabDiff && python -c "import json; json.load(open('sdcontract/meta.json')); json.load(open('sdcontract/capabilities.json')); print('ok')"`
Expected: prints `ok`.

- [ ] **Step 5: Commit**

```bash
git add sdcontract/generate sdcontract/meta.json sdcontract/capabilities.json
git commit -m "feat(contract): register tabdiff-universal variant + dialect passthrough"
```

---

## Task 10: `cat_snap_final` exactness escape hatch

**Files:**
- Modify: `tabdiff/guidance.py`
- Modify: `tabdiff/models/unified_ctime_diffusion.py`
- Modify: `sdcontract/_engine.py`
- Test: `tests/test_guidance_helpers.py`

Categorical equality under pure logit guidance is soft (spec §5.2). `cat_snap_final` hard-sets equality-constrained categorical columns to their target class on the final reverse step (i==0), giving guaranteed-exact equality when requested. Not-equal constraints are unaffected (snapping has no single target).

- [ ] **Step 1: Add a failing test for the snap helper**

```python
# append to tests/test_guidance_helpers.py
from tabdiff.guidance import snap_categorical


def test_snap_categorical_sets_equality_targets_only():
    import torch
    x_cat = torch.zeros(3, 2, dtype=torch.long)
    eq = CategoricalConstraint(col_pos=0, class_idx=2, scale=4.0, sign=1)
    ne = CategoricalConstraint(col_pos=1, class_idx=1, scale=4.0, sign=-1)
    out = snap_categorical(x_cat, [eq, ne])
    assert torch.all(out[:, 0] == 2)   # equality snapped
    assert torch.all(out[:, 1] == 0)   # not-equal column untouched
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_guidance_helpers.py -k snap -v`
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement `snap_categorical` in `tabdiff/guidance.py`**

```python
# append to tabdiff/guidance.py

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
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_guidance_helpers.py -k snap -v`
Expected: PASS.

- [ ] **Step 5: Thread `snap` through the model methods (additive)**

In `_edm_update_guided`, add `snap=False` to the signature (after `w_t=1.0`), and after the `x_cat_next, q_xs = self._mdlm_update(...)` line inside the `if has_cat:` block, add:

```python
            if snap and cat_constraints:
                x_cat_next = snap_categorical(x_cat_next, list(cat_constraints))
```

Add the import: extend the top-of-file guidance import to include `snap_categorical`.

In `sample_guided`, add `cat_snap_final=False` to the signature, and in the loop compute `snap = cat_snap_final and (i == 0)` and pass `snap=snap` to the `_edm_update_guided` call.

- [ ] **Step 6: Plumb the knob through the engine**

In `generate_guided` (`sdcontract/_engine.py`), add `cat_snap_final=False` to the signature and pass `cat_snap_final=cat_snap_final` into the `diffusion.sample_guided(...)` call. In `generate()`'s universal branch, read it from the request: pass `cat_snap_final=req.get("cat_snap_final", False)` to `generate_guided`.

- [ ] **Step 7: Verify imports + full helper suite**

Run: `cd sub/TabDiff && python -c "from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion; print('ok')" && python -m pytest tests/test_guidance_helpers.py -v`
Expected: prints `ok`; all helper tests PASS.

- [ ] **Step 8: Commit**

```bash
git add tabdiff/guidance.py tabdiff/models/unified_ctime_diffusion.py sdcontract/_engine.py tests/test_guidance_helpers.py
git commit -m "feat(guidance): cat_snap_final exactness escape hatch"
```

---

## Task 11: Full suite + isolation regression

**Files:** none (verification only)

- [ ] **Step 1: Run the whole new test suite**

Run: `cd sub/TabDiff && python -m pytest tests/ -v`
Expected: all PASS.

- [ ] **Step 2: Final isolation check on the model file**

Run: `git diff main -- tabdiff/models/unified_ctime_diffusion.py | grep '^-' | grep -v '^---'`
Expected: only the import-line change region (if any), NO removed lines inside `edm_update`/`sample`/`sample_impute`/`sample_all`.

- [ ] **Step 3 (optional, needs a trained adult checkpoint): contract smoke**

Run:
```bash
cd sub/TabDiff && echo '{"model_dir": "<path/to/adult/model_dir>", "dataset": "adult", "n_samples": 50, "variant": "universal", "constraints": ["age>30", "education=Bachelors"]}' | python sdcontract/generate
```
Expected: a single JSON line `{"ok": true, "output_csv": "...", "message": ""}`; the CSV's `age` mean visibly exceeds the unconditional baseline and `education` skews to Bachelors. Skip if no checkpoint is available locally.

---

## Notes for the implementer

- Run everything with `conda activate diffutabgen` from `sub/TabDiff`.
- Never run GraphTabGen anywhere (parent-repo rule); irrelevant here but a standing constraint.
- Calibrate `num_scale`/`cat_scale`/`mean_scale` on one `adult` sweep after Task 10; the defaults are starting points, not tuned values (spec §10).
- If `load_model_and_info` returns a model whose `y_only_model` is not None (CFG checkpoint), `sample_guided` will assert out by design — v1 does not support CFG + guidance (spec §10). Document this in the method's error message.
- **Deferred from v1: `exact_gradient` (DPS through `_denoise_fn`).** Spec §5.1/§6 list it as an optional advanced mode (default off). This plan implements only the cheaper x0-estimate guidance, which is the spec's default. Adding `exact_gradient` later means: in `_edm_update_guided` HOOK A, under `torch.enable_grad()`, set `x_num_hat.requires_grad_(True)`, re-run `_denoise_fn`, backprop the constraint loss to `x_num_hat`, and use that gradient instead of `compute_numeric_delta`. No interface here blocks it.

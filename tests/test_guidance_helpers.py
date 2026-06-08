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


def test_compute_numeric_delta_is_batch_size_independent():
    # Per-row constraints must produce the same per-row delta regardless of N.
    c = GreaterThan(idx=0, target=1.0, scale=0.1)
    small = compute_numeric_delta(torch.zeros(4, 1), [c], m=5, lr=1.0)
    large = compute_numeric_delta(torch.zeros(400, 1), [c], m=5, lr=1.0)
    assert small[0, 0].item() > 0.0
    assert torch.allclose(small[0, 0], large[0, 0], atol=1e-6)


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


from tabdiff.guidance import snap_categorical


def test_snap_categorical_sets_equality_targets_only():
    x_cat = torch.zeros(3, 2, dtype=torch.long)
    eq = CategoricalConstraint(col_pos=0, class_idx=2, scale=4.0, sign=1)
    ne = CategoricalConstraint(col_pos=1, class_idx=1, scale=4.0, sign=-1)
    out = snap_categorical(x_cat, [eq, ne])
    assert torch.all(out[:, 0] == 2)   # equality snapped
    assert torch.all(out[:, 1] == 0)   # not-equal column untouched

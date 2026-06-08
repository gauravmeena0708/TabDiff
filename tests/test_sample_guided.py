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

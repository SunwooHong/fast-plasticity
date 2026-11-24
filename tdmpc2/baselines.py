"""Reference utilities for the paper's gain controls.

These functions are not wired into scripts/evaluate.py. The runnable methods
are frozen, keyed, bias, and constant. Do not treat this module as an
executable reproduction of the gain-estimator results.
"""
import torch

from mwm_plastic import HebbianLatentResidual

# Online scalar-gain grid used in the paper. Not tuned on evaluation returns.
GAIN_GRID = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
GAIN_WINDOW = 25


class ConstantKey(HebbianLatentResidual):
    def __init__(self, *args, key, **kwargs):
        super().__init__(*args, **kwargs)
        self.key = key.detach().float().view(-1)

    def to(self, device):
        super().to(device)
        self.key = self.key.to(device)
        return self

    def phi(self, z, a):
        return self.key.to(dtype=z.dtype, device=z.device).view(1, -1).expand(z.shape[0], -1)


def known_gain(action, gain):
    """Action shown to the dynamics when the gain is known. The planner still commands the unscaled action."""
    return action * gain


def estimate_gain(model, history):
    if not history:
        return 1.0
    recent = history[-GAIN_WINDOW:]
    best_g, best_err = 1.0, None
    with torch.no_grad():
        for g in GAIN_GRID:
            total = 0.0
            for z, a, z_obs in recent:
                pred = model._dynamics(torch.cat([z, a * g], dim=-1))
                total += float((pred - z_obs).pow(2).mean())
            err = total / len(recent)
            if best_err is None or err < best_err:
                best_g, best_err = g, err
    return best_g

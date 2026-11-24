import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tdmpc2"))
from mwm_plastic import BiasCorrection, HebbianLatentResidual, simnorm


class PlasticityTests(unittest.TestCase):
    def test_zero_matches_dynamics(self):
        check_zero_matches_dynamics()

    def test_shapes(self):
        check_shapes()

    def test_write_after_score(self):
        check_write_after_score()

    def test_imagined_read_does_not_write(self):
        check_imagined_read_does_not_write()

    def test_reset(self):
        check_reset()

    def test_deterministic_update(self):
        check_deterministic_update()


def transition(plastic, z, a, z_hat, z_obs):
    pred = plastic.correct(z, a, z_hat)
    plastic.update(z, a, pred, z_obs)
    return pred


def check_zero_matches_dynamics():
    keyed = HebbianLatentResidual(16, 2, 8, feature_dim=4)
    z = torch.rand(1, 16)
    a = torch.rand(1, 2)
    z_hat = simnorm(torch.randn(1, 16), 8)
    assert torch.allclose(keyed.correct(z, a, z_hat), z_hat, atol=1e-5)


def check_shapes():
    keyed = HebbianLatentResidual(512, 6, 8)
    bias = BiasCorrection(512, 8)
    assert tuple(keyed.P.shape) == (512, 64)
    assert tuple(bias.P.shape) == (512, 1)


def check_write_after_score():
    plastic = BiasCorrection(16, 8)
    z = torch.rand(1, 16)
    a = torch.rand(1, 2)
    z_hat = simnorm(torch.randn(1, 16), 8)
    z_obs = simnorm(torch.randn(1, 16), 8)
    pred = plastic.correct(z, a, z_hat).clone()
    before = plastic.P.clone()
    assert torch.equal(plastic.P, before)
    scored = pred.clone()
    plastic.update(z, a, pred, z_obs)
    assert torch.equal(pred, scored)
    assert not torch.equal(plastic.P, before)


def check_imagined_read_does_not_write():
    plastic = HebbianLatentResidual(16, 2, 8, feature_dim=4)
    z = torch.rand(4, 16)
    a = torch.rand(4, 2)
    z_hat = simnorm(torch.randn(4, 16), 8)
    before = plastic.P.clone()
    steps = plastic.steps
    plastic.correct(z, a, z_hat)
    assert torch.equal(plastic.P, before)
    assert plastic.steps == steps


def check_reset():
    plastic = BiasCorrection(16, 8)
    z = torch.rand(1, 16)
    a = torch.rand(1, 2)
    z_hat = simnorm(torch.randn(1, 16), 8)
    z_obs = simnorm(torch.randn(1, 16), 8)
    transition(plastic, z, a, z_hat, z_obs)
    plastic.P.zero_()
    plastic.steps = 0
    assert torch.count_nonzero(plastic.P) == 0
    assert plastic.steps == 0


def check_deterministic_update():
    def once():
        torch.manual_seed(0)
        plastic = HebbianLatentResidual(16, 2, 8, feature_dim=4, seed=0)
        z = torch.rand(1, 16)
        a = torch.rand(1, 2)
        z_hat = simnorm(torch.randn(1, 16), 8)
        z_obs = simnorm(torch.randn(1, 16), 8)
        transition(plastic, z, a, z_hat, z_obs)
        return plastic.P.clone()

    assert torch.equal(once(), once())

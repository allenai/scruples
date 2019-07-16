"""Tests for scruples.baselines.loss."""

import unittest

import numpy as np
import torch as th

from scruples.baselines import loss


class TestSoftCrossEntropyLoss(unittest.TestCase):
    """Test SoftCrossEntropyLoss."""

    def test_forward(self):
        f = loss.SoftCrossEntropyLoss()

        self.assertAlmostEqual(
            f(
                th.Tensor([[0., 0.]]),
                th.Tensor([[0.5, 0.5]])
            ).item(),
            0.6931471805599453)
        self.assertAlmostEqual(
            f(
                th.Tensor([[0., 0.], [-1.0986123, 1.0986123]]),
                th.Tensor([[0.3, 0.7], [0.6, 0.4]])
            ).item(),
            1.0584212213097515)


class TestDirichletMultinomialLoss(unittest.TestCase):
    """Test DirichletMultinomialLoss."""

    def test_forward(self):
        f = loss.DirichletMultinomialLoss()

        # when [a, b] goes to [0, 0], the dirichlet prior becomes a
        # Bernoulli distribution with p = 0.5, where there's a 50/50
        # chance that the label always comes up 0 or always comes up 1
        self.assertAlmostEqual(
            f(
                th.Tensor([[np.log(1e-20), np.log(1e-20)]]),
                th.Tensor([[3, 0]])
            ).item(),
            0.6931471805599453,
            places=4)
        # test other cases
        self.assertAlmostEqual(
            f(
                th.Tensor([[np.log(1.), np.log(1.)]]),
                th.Tensor([[0.5, 0.5]])
            ).item(),
            0.9347116558304358)
        self.assertAlmostEqual(
            f(
                th.Tensor(
                    [[np.log(1.), np.log(1.)],
                     [np.log(0.5), np.log(2.)]]
                ),
                th.Tensor([[0.3, 0.7], [0.6, 0.4]])
            ).item(),
            1.1093992405423625)

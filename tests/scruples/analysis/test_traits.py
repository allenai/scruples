"""Tests for scruples.analysis.traits."""

import unittest

import numpy as np
import pytest
from scipy.linalg import orthogonal_procrustes
import torch as th

from scruples.analysis import traits


class LatentTraitModelTestCase(unittest.TestCase):
    """Test the LatentTraitModel class."""

    def _check_fit_on_synthetic_example(
            self,
            latent_dim: int,
            n_samples: int,
            n_variables: int,
            lr: float,
            n_batch_size: int,
            patience: int,
            n_epochs: int
    ) -> None:
        # create a random ground-truth for W and b
        W = np.random.randn(latent_dim, n_variables)
        b = np.random.randn(n_variables)

        # generate data from the ground-truth model

        # step 1: sample the latent variables
        zs = np.random.randn(n_samples, latent_dim)
        # step 2: use the latent variables to generate the observations
        data = (
            # evaluate the probability vectors
            1. / (1. + np.exp(-(np.matmul(zs, W) + b)))
            # compare the probabilities to uniform random noise to
            # create the observations
            > np.random.rand(n_samples, n_variables)
        ).astype(int)

        # create and fit the model
        model = traits.LatentTraitModel(latent_dim=latent_dim)
        model.fit(
            data,
            lr=lr,
            n_batch_size=n_batch_size,
            patience=patience,
            n_epochs=n_epochs,
            device=th.device('cuda')
              if th.cuda.is_available()
              else th.device('cpu'))

        # evaluate the parameter estimates

        W_hat = model.W_.detach().cpu().numpy()
        # Since the parameters in a latent trait / factor analysis model
        # are only determined up to a rotation, we must align the
        # learned parameters with the ground-truth with a rotation.
        R, _ = orthogonal_procrustes(W_hat, W)
        W_hat = np.matmul(W_hat, R)

        b_hat = model.b_.detach().cpu().numpy()

        # check that self.n_samples_ and self.n_variables_ were set
        # correctly
        self.assertEqual(model.n_samples_, n_samples)
        self.assertEqual(model.n_variables_, n_variables)

        # sanity-check that the learned weights give a better estimate
        # than the origin
        if latent_dim > 0:
            self.assertLess(
                np.mean((W - W_hat)**2),
                np.mean(W**2))
        self.assertLess(
            np.mean((b - b_hat)**2),
            np.mean(b**2))

        # check that the learned weights give good estimates for this
        # easy problem
        if latent_dim > 0:
            self.assertLess(
                np.mean((W - W_hat)**2),
                7.5e-2)
        self.assertLess(
            np.mean((b - b_hat)**2),
            7.5e-2)

    def test___init__(self):
        # test that __init__ sets latent_dim
        model = traits.LatentTraitModel(latent_dim=10)

        self.assertEqual(model.latent_dim, 10)

    @pytest.mark.slow
    def test_fit_learns_no_trait_model(self):
        self._check_fit_on_synthetic_example(
            latent_dim=0,
            n_samples=1024,
            n_variables=20,
            lr=1e1,
            n_batch_size=1024,
            patience=5,
            n_epochs=100)

    @pytest.mark.slow
    def test_fit_learns_one_trait_model(self):
        self._check_fit_on_synthetic_example(
            latent_dim=1,
            n_samples=4096,
            n_variables=20,
            lr=1e1,
            n_batch_size=1024,
            patience=4,
            n_epochs=50)

    @pytest.mark.slow
    def test_fit_learns_two_trait_model(self):
        self._check_fit_on_synthetic_example(
            latent_dim=2,
            n_samples=32768,
            n_variables=20,
            lr=1e1,
            n_batch_size=1024,
            patience=3,
            n_epochs=25)

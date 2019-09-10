"""Latent trait analysis."""

import numpy as np
import torch as th


class LatentTraitModel:
    """A latent trait model.

    Attributes
    ----------
    latent_dim : int
        The number of latent traits in the model.
    n_samples_ : int
        The number of samples used during fitting.
    n_variables_ : int
        The number of observed variables in the latent trait model.
    W_ : th.Tensor
        The weight matrix for the latent trait model.
    b_ : th.Tensor
        The bias vector for the latent trait model.

    Parameters
    ----------
    latent_dim : int
        The number of latent traits in the model.
    """

    def __init__(
            self,
            latent_dim: int
    ) -> None:
        self.latent_dim = latent_dim

        self.n_samples_ = None
        self.n_variables_ = None

        self.W_ = None
        self.b_ = None

    def fit(
            self,
            data: th.Tensor,
            n_latent_samples: int = 10000,
            lr: float = 1e1,
            n_batch_size: int = 128,
            patience: int = 5,
            n_epochs: int = 100,
            device: th.device = th.device('cpu')
    ) -> None:
        """Fit the latent trait model.

        The latent trait model is fit via maximum likelihood. The latent
        variable is marginalized out using Monte Carlo integration, and
        the optimization is performed via SGD.

        Parameters
        ----------
        data : th.Tensor
            An n x k tensor of ints representing the binary responses of
            each of the n annotators to all of the k questions.
        n_latent_samples : int, optional (default=10000)
            The number of samples to use in the Monte Carlo integration.
        lr : float, optional (default=1e1)
            The learning rate for SGD.
        n_batch_size : int, optional (default=128)
            The mini-batch size for SGD.
        patience : int, optional (default=5)
            The number of epochs to wait for the loss to reduce before
            reducing the learning rate.
        n_epochs : int, optional (default=100)
            The number of epochs to run SGD.
        device : th.device, optional (default=th.device('cpu'))
            The torch device to use.

        Returns
        -------
        None
        """
        # move the data to the device
        data = th.tensor(data, device=device)

        # compute constants
        n_samples, n_variables = data.shape

        # initialize the parameters
        # (use a Xavier-like initialization)

        # shape: self.latent_dim x n_variables
        W = th.nn.Parameter(
            (2. / (self.latent_dim + n_variables))**0.5
            * th.randn(
                self.latent_dim,
                n_variables,
                requires_grad=True,
                device=device))
        # shape: n_variables
        b = th.nn.Parameter(
            th.zeros(
                n_variables,
                requires_grad=True,
                device=device))

        # optimization
        optimizer = th.optim.SGD([W, b], lr=lr)
        scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode='min',
            factor=1e-1,
            patience=patience)

        # run training
        for epoch in range(n_epochs):
            epoch_nlls = []
            epoch_perm = np.random.permutation(len(data))
            for i in range(0, len(data), n_batch_size):
                optimizer.zero_grad()

                # assemble the mini-batch
                # shape: n_latent_samples x self.latent_dim
                mb_latents = th.randn(
                    n_latent_samples,
                    self.latent_dim,
                    device=device)
                # shape: n_batch_size x n_variables
                mb_labels = data[epoch_perm[i:i+n_batch_size]]

                # shape: n_latent_samples x n_variables
                if self.latent_dim > 0:
                    mb_latent_probs = th.sigmoid(
                        th.matmul(mb_latents, W) + b)
                else:
                    mb_latent_probs = th.sigmoid(
                        b.repeat(n_latent_samples, 1))

                # shape: n_batch_size
                mb_likelihoods = th.mean(
                    th.prod(
                        mb_labels.float().unsqueeze(-1)
                          * mb_latent_probs.t()
                        + (1 - mb_labels.float()).unsqueeze(-1)
                          * (1 - mb_latent_probs.t()),
                        dim=1),
                    dim=-1)

                # shape: scalar
                mb_nll = - th.mean(th.log(mb_likelihoods))

                mb_nll.backward()
                optimizer.step()

                # update statistics
                epoch_nlls.append(mb_nll.cpu().item())

            # update the learning rate
            scheduler.step(np.mean(epoch_nlls))

        # update fitted attributes

        self.n_samples_ = n_samples
        self.n_variables_ = n_variables

        self.W_ = W.detach().cpu()
        self.b_ = b.detach().cpu()

    def project(
            self,
            data: th.Tensor,
            lr: float = 1e1,
            n_batch_size: int = 128,
            patience: int = 5,
            n_epochs: int = 250,
            device: th.device = th.device('cpu')
    ) -> th.Tensor:
        """Project ``data`` into the latent space.

        Project ``data`` into the latent space using the maximum a
        posteriori (MAP) estimate.

        Parameters
        ----------
        data : th.Tensor
            The data matrix to project into the latent space. The tensor
            should have the shape: # samples x # observed variables.
        lr : float, optional (default=1e1)
            The learning rate for SGD.
        n_batch_size : int, optional (default=128)
            The mini-batch size for SGD.
        patience : int, optional (default=5)
            The number of epochs to wait for the loss to reduce before
            reducing the learning rate.
        n_epochs : int, optional (default=250)
            The number of epochs to run SGD.
        device : th.device, optional (default=th.device('cpu'))
            The torch device to use.

        Returns
        -------
        th.Tensor
            ``data`` projected into the latent space. This tensor has
            the shape: # samples x # latent traits.
        """
        if self.latent_dim == 0:
            raise ValueError(
                'Cannot project the observations into the latent space'
                ' when the latent dimension is 0.')

        # By Bayes' Rule, we have:
        #
        #                P(Y | Z) P(Z)
        #     P(Z | Y) = -------------
        #                    P(Y)
        #
        # Since P(Y) is a constant, we can then obtain the MAP estimate
        # by optimizing Z such that P(Y | Z) P(Z) is maximized, or
        # equivalently such that - (log P(Y | Z) + log P(Z)) is
        # minimized.
        #
        # Applying this to the likelihood yields:
        #
        #     - log \prod_n P(y_n | z_n) P(z_n)
        #     = - \sum_n \log P(y_n | z_n) + log P(z_n)
        #     = - \sum_n \sum_i y_{ni} \log \sigma(w_i z_n + b_i)
        #                       + (1 - y_{ni}) \log (1 - \sigma(w_i z_n + b_i))
        #                - \frac{|| z_n ||_2^2}{2}
        #                + C
        #
        # Where C is a constant.

        # move the data to the device
        data = th.tensor(data, device=device)

        # compute constants
        n_samples, n_variables = data.shape

        if n_variables != self.n_variables_:
            raise ValueError(
                'The data matrix has a different number of observed'
                ' variables than the model was trained with.')

        W = self.W_.to(device)
        b = self.b_.to(device)

        # initialize latent variable estimates
        # shape: n_samples x self.latent_dim
        zs = th.nn.Parameter(
            th.zeros(
                n_samples,
                self.latent_dim,
                requires_grad=True,
                device=device))

        # optimization
        optimizer = th.optim.SGD([zs], lr=lr)
        scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode='min',
            factor=1e-1,
            patience=patience)

        # run training
        for epoch in range(n_epochs):
            # nlp stands for negative log-probability
            epoch_nlps = []
            epoch_perm = np.random.permutation(len(data))
            for i in range(0, len(data), n_batch_size):
                optimizer.zero_grad()

                # assemble the mini-batch
                # shape: n_batch_size x self.latent_dim
                mb_latents = zs[epoch_perm[i:i+n_batch_size]]
                # shape: n_batch_size x n_variables
                mb_labels = data[epoch_perm[i:i+n_batch_size]]

                # shape: n_batch_size x n_variables
                mb_probs = th.sigmoid(th.matmul(mb_latents, W) + b)

                # shape: scalar
                mb_nlp = - th.mean(
                    th.sum(
                        mb_labels.float() * th.log(mb_probs)
                        + (1 - mb_labels.float()) * th.log(1 - mb_probs),
                        dim=-1)
                    - th.matmul(mb_latents, mb_latents.t()) / 2)

                mb_nlp.backward()
                optimizer.step()

                # update statistics
                epoch_nlps.append(mb_nlp.cpu().item())

            # update the learning rate
            scheduler.step(np.mean(epoch_nlps))

        return zs.detach().cpu()

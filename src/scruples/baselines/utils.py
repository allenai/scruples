"""Utilities for baselines on scruples."""

from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy.special import gammaln
from sklearn.base import (
    BaseEstimator,
    TransformerMixin)
from sklearn.utils.validation import check_is_fitted


# functions

def concat_title_and_text(features: pd.DataFrame) -> np.ndarray:
    """Return the concatenation of the title and text features.

    Parameters
    ----------
    features : pd.DataFrame
        The features for the scruples dataset.

    Returns
    -------
    np.ndarray
        The concatentation of the title and text strings separated by a
        newline character, in a numpy array.
    """
    return (features['title'] + '\n' + features['text']).values


def dirichlet_multinomial(log_alphas: np.ndarray) -> np.ndarray:
    """Return class probabilities from a dirichlet-multinomial model.

    Parameters
    ----------
    log_alphas : np.ndarray
        An n x k dimensional numpy array where n is the number of
        samples and k is the number of classes. The values of the array
        should correspond to the log of the alpha parameters for the
        predicted dirichlet distribution corresponding to each instance.

    Returns
    -------
    np.ndarray
        An n x k dimensional array giving the class probabilities
        corresponding to ``log_alphas`` for each sample.
    """
    alphas = np.exp(log_alphas)
    return alphas / np.expand_dims(np.sum(alphas, axis=-1), -1)


# classes

class ResourceTransformer(BaseEstimator, TransformerMixin):
    """Featurize the action pairs from the scruples resource.

    ``ResourceTransformer`` applies the same featurization pipeline
    (``self.transformer``) to both actions in an instance from the
    scruples resource and then takes the difference of their
    features.

    You can set parameters on the ``self.transformer`` attribute by
    prefixing parameters to ``ResourceTransformer`` with
    ``transformer__``.

    ``ResourceTransformer`` is particularly useful in front of linear
    models like logistic regression, since applying the model to the
    difference of the features is the same as taking the difference of
    the final scores.

    Attributes
    ----------
    See `Parameters`_.

    Parameters
    ----------
    transformer : Transformer
        The transformer to apply to the actions.
    """
    def __init__(
            self,
            transformer: TransformerMixin
    ) -> None:
        self.transformer = transformer

    def set_params(
            self,
            **params: Dict[str, Any]
    ) -> 'ResourceTransformer':
        self_params = {}
        transformer_params = {}
        for param, value in params.items():
            if param.startswith('transformer__'):
                transformer_params[param[13:]] = value
            else:
                self_params[param] = value
        # set the parameters on this instance
        super().set_params(**self_params)
        # set the parameters on the transformer attribute
        self.transformer.set_params(**transformer_params)

        return self

    def fit(
            self,
            X: pd.DataFrame,
            y: np.ndarray = None
    ) -> 'ResourceTransformer':
        """Fit the instance to ``X``.

        Fitting an instance of ``ResourceTransformer`` fits its
        ``self.transformer`` attribute to the data. The ``y`` argument
        is ignored.

        Parameters
        ----------
        X : pd.DataFrame
            The data to fit.
        y : None
            An ignored argument.

        Returns
        -------
        self : object
            The instance.
        """
        X_ = pd.concat([X['action0'], X['action1']])

        self.transformer.fit(X_)

        self._fitted = True

        return self

    def transform(
            self,
            X: pd.DataFrame
    ) -> Any:
        """Transform ``X``.

        Parameters
        ----------
        X : pd.DataFrame
            The data to transform.

        Returns
        -------
        Any
            The difference of the features for the actions derived by
            applying ``self.transformer`` to them.
        """
        check_is_fitted(self, '_fitted')

        return (
            self.transformer.transform(X['action1'])
            - self.transformer.transform(X['action0'])
        )

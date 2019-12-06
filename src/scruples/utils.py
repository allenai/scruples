"""Utilities."""

import collections
import logging
import os
import random
import string
from typing import (
    Callable,
    List,
    Optional,
    Tuple)

from autograd import grad
import autograd.numpy as np
from autograd.scipy.special import gammaln
import regex
from scipy.optimize import minimize, minimize_scalar
from scipy.special import softmax
from sklearn import metrics

from . import settings


def configure_logging(verbose: bool = False) -> logging.Handler:
    """Configure logging and return the log handler.

    This function is useful in scripts when logging should be set up
    with a basic configuration.

    Parameters
    ----------
    verbose : bool, optional (default=False)
        If ``True``, set the log level to DEBUG, else set the log level
        to INFO.

    Returns
    -------
    logging.Handler
        The log handler set up by this function to handle basic logging.
    """
    # unset the log level from root (defaults to WARNING)
    logging.root.setLevel(logging.NOTSET)

    # set up the log handler
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    handler.setFormatter(logging.Formatter(settings.LOG_FORMAT))

    # attach the log handler to root
    logging.root.addHandler(handler)

    return handler


_character_filter_regex = regex.compile(r'[^\w\s]')
_whitespace_regex = regex.compile(r'\s+')

def count_words(text: str):
    """Return the number of words in ``text``.

    Return the number of words in ``text`` where a word is defined as a
    string of non-whitespace characters. Non-alphanumeric characters are
    ignored.

    Parameters
    ----------
    text : str
        The text to count words from.

    Returns
    -------
    int
        The number of words in ``text``.

    Examples
    --------
    The basic operation of ``count_words`` splits on whitespace and
    counts the number of tokens:

    >>> count_words("two words")
    2

    Successive whitespace characters are replaced by a single space, and
    leading and trailing whitespace does not affect the token count:

    >>> count_words("three    words here ")
    3

    Similarly, puntuaction and non-alphanumeric characters are stripped
    and ignored in the computation:

    >>> count_words("hello, world !!!")
    2
    """
    # strip non-alphanumeric characters
    text = _character_filter_regex.sub('', text)

    # replace strings of whitespace with a single space
    text = _whitespace_regex.sub(' ', text)

    return len(text.strip().split())


def xentropy(
        y_true: np.ndarray,
        y_pred: np.ndarray
) -> np.float64:
    """Return the xentropy of ``y_pred`` with respect to ``y_true``.

    Parameters
    ----------
    y_true : np.ndarray, required
        An ``n_samples`` by ``n_classes`` array for the class
        probabilities given to each sample.
    y_pred : np.ndarray, required
        An ``n_samples`` by ``n_classes`` array for the predicted class
        probabilities given to each sample.

    Returns
    -------
    np.float64
        The xentropy of ``y_pred` with respect to ``y_true``.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(- np.sum(np.log(y_pred ** y_true), axis=1))


def make_id() -> str:
    """Return a random ID string."""
    return ''.join(
        random.choices(
            string.ascii_letters + string.digits,
            k=32))


def make_confusion_matrix_str(
        y_true: List[str],
        y_pred: List[str],
        labels: Optional[List[str]] = None
) -> str:
    """Return the confusion matrix as a pretty printed string.

    Parameters
    ----------
    y_true : List[str], required
        The true labels.
    y_pred : List[str], required
        The predicted labels.
    labels : Optional[List[str]], optional (default=None)
        The list of label names. If ``None`` then the list of possible
        labels is computed from ``y_true`` and ``y_pred``.

    Returns
    -------
    str
        A pretty printed string for the confusion matrix.
    """
    labels = labels or [
        str(label)
        for label in sorted(list(set(y_true).union(set(y_pred))))
    ]
    confusion_matrix = metrics.confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels)

    # The cell length should be the longest of:
    #
    #   1. 6 characters
    #   2. The length of the longest label string
    #   3. The maximum number of digits of any number in the confusion
    #      matrix
    #
    cell_length = max(
        6,
        max(len(label) for label in labels),
        np.ceil(
            np.log(np.max(np.abs(confusion_matrix)))
            / np.log(10)))

    header_separator = (
        '+'
        + '+'.join('=' * (cell_length + 2) for _ in range(len(labels) + 1))
        + '+')

    body_separator = (
        '+'
        + '+'.join('-' * (cell_length + 2) for _ in range(len(labels) + 1))
        + '+')

    header = (
        f'| {"": <{cell_length}} |'
        + '|'.join(f' {label: <{cell_length}} ' for label in labels)
        + '| predicted')

    body = (
        '|'
        + '|\n|'.join(
            f' {label: <{cell_length}} |' + '|'.join(
                f' {x: >{cell_length}} ' for x in row)
            for label, row in zip(labels, confusion_matrix))
        + '|')

    return (
        f'{header_separator}\n'
        f'{header}\n'
        f'{header_separator}\n'
        f'{body}\n'
        f'{body_separator}\n'
        f' true'
    )


def make_label_distribution_str(
        y_true: List[str],
        labels: Optional[List[str]] = None
) -> str:
    """Return a pretty printed table with the label distribution.

    Parameters
    ----------
    y_true : List[str], required
        The true labels.
    labels : Optional[List[str]], optional (default=None)
        The list of labels. If ``None`` then the labels are computed
        from ``y_true``.

    Returns
    -------
    str
        A pretty printed table of the label distribution.
    """
    if len(y_true) == 0:
        return "[no data]"

    labels = labels or [
        str(label)
        for label in sorted(list(set(y_true)))
    ]
    label_counts = collections.defaultdict(int)
    for label in y_true:
        label_counts[label] += 1

    total_n_labels = len(y_true)

    # The cell length should be the longest of:
    #
    #   1. 8 characters (the length of the word "fraction")
    #   2. The length of the longest label string
    #   3. The maximum number of digits of any number in the list of
    #      label counts
    #
    cell_length = max(
        8,
        max(len(label) for label in labels),
        np.ceil(
            np.log(np.max(np.abs(list(label_counts.values()))))
            / np.log(10)))

    header_separator = (
        '+'
        + '+'.join('=' * (cell_length + 2) for _ in range(len(labels) + 1))
        + '+')

    body_separator = (
        '+'
        + '+'.join('-' * (cell_length + 2) for _ in range(len(labels) + 1))
        + '+')

    header = (
        f'| {"": <{cell_length}} |'
        + '|'.join(f' {label: <{cell_length}} ' for label in labels)
        + '|')

    body = (
        f'| {"fraction": <{cell_length}} |'
        + '|'.join(
            f' {label_counts[label] / total_n_labels:>{cell_length}.4f} '
            for label in labels)
        + '|\n'
        + body_separator + '\n'
        + f'| {"total": <{cell_length}} |'
        + '|'.join(
            f' {label_counts[label]: >{cell_length}} '
            for label in labels)
        + '|')

    return (
        f'{header_separator}\n'
        f'{header}\n'
        f'{header_separator}\n'
        f'{body}\n'
        f'{body_separator}\n'
    )


def next_unique_path(path: str):
    """Return a unique path derived from ``path``.

    If ``path`` does not exist, return ``path``, else find the smallest
    natural number such that appending an underscore followed by it to
    ``path`` creates a unique path, then return ``path`` with an
    underscore and that natural number appended.

    Parameters
    ----------
    path : str
        The path to generate a unique version of.

    Returns
    -------
    str
        ``path`` with ``_{number}`` appended where ``{number}`` is
        the smallest natural number that makes the path unique.
    """
    if not os.path.exists(path):
        return path

    number = 1
    new_path = f'{path}_{number}'
    while os.path.exists(new_path):
        number += 1
        new_path = f'{path}_{number}'

    return new_path


def estimate_beta_binomial_parameters(
        successes: np.ndarray,
        failures: np.ndarray
) -> Tuple[float, float]:
    """Return the MLE for the beta-binomial distribution.

    Given success counts, ``successes``, and failure counts,
    ``failures``, return the MLE for the parameters of a beta-binomial
    model of that data.

    Parameters
    ----------
    successes : np.ndarray
        An array of integers giving the observed number of successes for
        each trial.
    failures : np.ndarray
        An array of integers giving the observed number of failures for
        each trail.

    Returns
    -------
    a : float, b : float
        A tuple, ``(a, b)``, giving the MLE of the alpha and beta
        parameters for the beta-binomial model.
    """
    # This function computes the negative log-likelihood for the
    # beta-binomial model with parameters a and b on the provided data
    # (successes and failures)
    def nll(params):
        a, b = params
        # N.B. the likelihood must be expressed as a sum of log-gamma
        # factors in order to avoid overflow / underflow for large
        # numbers of successes or failures in an observation.
        return - np.sum(
            gammaln(successes + failures + 1)
            + gammaln(successes + a)
            + gammaln(failures + b)
            + gammaln(a + b)
            - gammaln(successes + 1)
            - gammaln(failures + 1)
            - gammaln(successes + failures + a + b)
            - gammaln(a)
            - gammaln(b))

    # minimize the negative log-likelihood
    a, b = minimize(
        fun=nll,
        x0=[1., 1.],
        jac=grad(nll),
        bounds=[(1e-10, None), (1e-10, None)]
    ).x

    return a, b


def estimate_dirichlet_multinomial_parameters(
        observations: np.ndarray
) -> Tuple[float]:
    """Return the MLE for the dirichlet-multinomial distribution.

    Given observations, ``observations``, return the MLE for the
    parameters of a dirichlet-multinomial model of that data.

    Parameters
    ----------
    observations : np.ndarray
        An n x k array of integers giving the observed number of counts
        for each class in each trial, where n is the number of trials
        and k is the number of classes.

    Returns
    -------
    Tuple[float]
        A tuple giving the MLE of the parameters for the
        dirichlet-multinomial model.
    """
    # This function computes the negative log-likelihood for the
    # dirichlet-multinomial model with parameters given by params and on
    # the provided data.
    def nll(params):
        # N.B. the likelihood must be expressed as a sum of log-gamma
        # factors in order to avoid overflow / underflow for large
        # numbers of successes or failures in an observation.
        return - np.sum(
            gammaln(np.sum(params))
            - gammaln(
                np.sum(params)
                + np.sum(observations, axis=1))
            + np.sum(
                gammaln(observations + params)
                - gammaln(params),
              axis=1))

    # minimize the negative log-likelihood
    params = minimize(
        fun=nll,
        x0=[1. for _ in range(observations.shape[-1])],
        jac=grad(nll),
        bounds=[(1e-10, None) for _ in range(observations.shape[-1])]
    ).x

    return params


def calibration_factor(
        logits: np.ndarray,
        targets: np.ndarray
) -> np.float64:
    """Return the calibrating temperature for the model.

    Parameters
    ----------
    logits : np.ndarray, required
        The logits from the model to calibrate.
    targets : np.ndarray, required
        The targets on which to calibrate. The targets should be probabilities.

    Returns
    -------
    np.float64
        The temperature to use when calibrating the model. Divide the logits by
        this number to calibrate them.
    """
    def loss(t):
        return xentropy(y_true=targets, y_pred=softmax(logits / t, axis=-1))

    return minimize_scalar(
        fun=loss,
        bounds=(1e-10, 1e10),
    ).x


def oracle_performance(
        ys: np.ndarray,
        metric: Callable,
        make_predictions: Optional[Callable] = None,
        n_samples: int = 625
) -> float:
    """Estimate the oracle performance.

    Estimate the oracle performance for ``metric_func`` on a dataset
    with labels given by ``ys``.

    Parameters
    ----------
    ys : np.ndarray, required
        An N x K array where N is the number of data points, K is the
        number of classes, and the ij'th entry gives the number of
        annotators that said example i is from class j.
    metric : Callable, required
        A function taking in ``y_pred`` and ``y_true`` arguments and
        returning a float representing the performance.
    make_predictions : Optional[Callable], optional (default=None)
        An optional function to apply to the predictions and true labels
        (i.e. class probabilities) before passing them to
        ``metric``. Typically, this transformation would be an
        argmax. If ``None``, then the identity transformation is used.
    n_samples : int, optional (default=625)
        The number of samples to use for Monte-carlo integration.

    Returns
    -------
    float
        An estimate of the oracle performance.
    float
        An estimate of the sampling error. The estimator uses a Monte-carlo
        integration step, so this value represents the standard error of that
        integration.
    """
    y_true = (
        ys / np.expand_dims(np.sum(ys, axis=-1), axis=-1)
    )
    if make_predictions:
        y_true = make_predictions(y_true)

    estimated_alphas = estimate_dirichlet_multinomial_parameters(ys)
    posterior = estimated_alphas + ys

    scores = []
    for _ in range(n_samples):
        # efficiently sample from dirichlet distributions in a vectorized way
        # using the fact that the dirichlet distribution is equivalent to a
        # normalized vector of independent gamma distributed random variables.
        y_pred = np.random.gamma(posterior)
        y_pred = (
            y_pred / np.expand_dims(np.sum(y_pred, axis=-1), axis=-1)
        )
        if make_predictions:
            y_pred = make_predictions(y_pred)

        scores.append(metric(y_pred=y_pred, y_true=y_true))

    return np.mean(scores), np.std(scores) / n_samples**0.5

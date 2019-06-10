"""Utilities."""

import collections
import logging
import random
import string
from typing import (
    List,
    Optional)

import numpy as np
import regex
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
    return np.mean(- np.sum(y_true * np.log(y_pred), axis=1))


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

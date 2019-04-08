"""Utilities."""

from functools import wraps
from typing import (
    Callable,
    List,
    Optional)

import numpy as np
import regex
from sklearn import metrics


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

    return len(text.strip().split(' '))


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
    labels : Optional[List[str]], optional (default = None)
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
        + '|')

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
        f'{body_separator}'
    )


def cached_property(method: Callable):
    """Decorate a method to act as a cached property.

    This decorator converts a method into a cached property. It is
    intended to only be used on the methods of classes decorated with
    ``@attr.s`` where ``frozen=True``. This decorator works analogously
    to ``@property`` except it caches the computed value.

    Parameters
    ----------
    method : Callable, required
        The method to decorate. ``method`` should take only one
        argument: ``self``.

    Returns
    -------
    Callable
        The decoratored method.

    Notes
    -----
    When used on a frozen attrs class, values for the property may
    safely be cached because the object is intended to be
    immutable. Additionally, the best place to store these cached values
    is on the object itself, so that they can be garbage collected when
    the object is.
    """
    @wraps(method)
    def wrapper(self):
        cached_name = f'_{method.__name__}'
        if not hasattr(self, cached_name):
            value = method(self)

            # To get around the immutability of the instance, we have to
            # use __setattr__ from object.
            object.__setattr__(self, cached_name, value)

        return getattr(self, cached_name)
    return property(wrapper)

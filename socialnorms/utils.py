"""Utilities."""

import regex


_character_filter_regex = regex.compile(r'[^\w\s]')
_whitespace_regex = regex.compile(r'\s+')

def count_words(text):
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

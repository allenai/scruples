"""Filter functions."""

import regex

from . import base
from .. import utils


class EmptyStringFilter(base.LoggedCallable):
    """Filter out the empty string."""

    def apply(
            self,
            s: str
    ) -> bool:
        """Return ``True`` if ``s`` is the empty string.

        Parameters
        ----------
        s : str, required
            The string to test.

        Returns
        -------
        bool
            ``True`` if ``s`` is the empty string, otherwise ``False``.
        """
        return s == ''


class TooFewCharactersFilter(base.LoggedCallable):
    """Filter strings with too few characters.

    Parameters
    ----------
    min_chars : int, required
        The minimum number of characters to allow.
    """

    def __init__(
            self,
            min_chars: int,
            *args,
            **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.min_chars = min_chars

    def apply(
            self,
            s: str
    ) -> bool:
        """Return ``True`` if ``s`` has too few characters.

        Parameters
        ----------
        s : str, required
            The string to test.

        Returns
        -------
        bool
            ``True`` if ``s`` has fewer than ``self.min_chars``
            characters, otherwise ``False``.
        """
        if not isinstance(s, str):
            raise ValueError(f's ({s}) must be a string.')

        return len(s) < self.min_chars


class TooFewWordsFilter(base.LoggedCallable):
    """Filter out strings with too few words.

    Parameters
    ----------
    min_words : int, required
        The minimum number of words to allow.
    """

    def __init__(
            self,
            min_words: int,
            *args,
            **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.min_words = min_words

    def apply(
            self,
            s: str
    ) -> bool:
        """Return ``True`` if ``s`` has too few words.

        Parameters
        ----------
        s : str, required
            The string to test.

        Returns
        -------
        bool
            ``True`` if ``s`` has fewer than ``self.min_words`` words,
            otherwise ``False``.
        """
        if not isinstance(s, str):
            raise ValueError(f's ({s}) must be a string.')

        return utils.count_words(s) < self.min_words


class PrefixFilter(base.LoggedCallable):
    """Filter out strings with a certain prefix.

    Parameters
    ----------
    prefix : str, required
        The prefix to look for when filtering strings.
    case_sensitive : bool, optional (default=False)
        Whether or not to match the prefix in a case-sensitive fashion.
    """

    def __init__(
            self,
            prefix: str,
            case_sensitive: bool = False,
            *args,
            **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.prefix = prefix
        self.case_sensitive = case_sensitive

    def apply(
            self,
            s: str
    ) -> bool:
        """Return ``True`` if ``s`` begins with ``prefix``.

        Parameters
        ----------
        s : str, required
            The string to test.

        Returns
        -------
        bool
            ``True`` if ``s`` begins with ``self.prefix``, otherwise
            ``False``.
        """
        if not isinstance(s, str):
            raise ValueError(f's ({s}) must be a string.')

        if not self.case_sensitive:
            prefix, s = self.prefix.lower(), s.lower()
        else:
            prefix = self.prefix

        return s.startswith(prefix)


class StartsWithGerundFilter(base.LoggedCallable):
    """Filter strings which have no gerunds in the first few words."""

    _pattern = regex.compile(
        r'^(?:[\p{Ps}\p{Pi}"\']?\m[^\s]+\M[\p{Pe}\p{Pf}"\']? ){0,2}'
        r'[\p{Pi}"\']?\m[^\s]+ing\M[\p{Pf}"\']?')

    def apply(
            self,
            s: str
    ) -> bool:
        """Return ``True`` if ``s`` has no gerunds in the first 3 words.

        Parameters
        ----------
        s : str, required
            The string to test.

        Returns
        -------
        bool
            ``True`` if ``s`` has no gerunds in the first three words,
            otherwise ``False``.
        """
        return self._pattern.match(s) is None


class WhWordFilter(base.LoggedCallable):
    """Filter strings which start with a wh-word."""

    _wh_word_patterns = [
        'why',
        'who',
        'which',
        'what',
        'where',
        'when',
        'how'
    ]
    # compile the _wh_word_patterns to regexes
    _wh_word_patterns = [
        regex.compile(r'(?i:[\p{{Pi}}"\']?\m{}\M[\p{{Pf}}"\']?)'.format(w))
        for w in _wh_word_patterns
    ]

    def apply(
            self,
            s: str
    ) -> bool:
        """Return ``True`` if ``s`` starts with a wh-word.

        Parameters
        ----------
        s : str, required
            The string to test.

        Returns
        -------
        bool
            ``True`` if ``s`` begins with a wh-word, otherwise
            ``False``.
        """
        return any(pattern.match(s) for pattern in self._wh_word_patterns)

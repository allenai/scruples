"""Text normalizers."""

from typing import (
    Callable,
    List)

import ftfy
import regex
import spacy

from . import base


class ComposedNormalizer(base.LoggedCallable):
    """Compose several normalizers.

    Parameters
    ----------
    normalizers : List[Callable], required
         The list of normalizers to compose.
    """

    def __init__(
            self,
            normalizers: List[Callable],
            *args,
            **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.normalizers = normalizers

    def apply(
            self,
            s: str
    ) -> str:
        """Return ``s`` transformed by ``self.normalizers``.

        Parameters
        ----------
        s : str, required
            The input string to normalize.

        Returns
        -------
        str
            ``s`` transformed by each normalizer in ``self.normalizers``
            in the order in which they appear in the list.
        """
        for normalizer in self.normalizers:
            s = normalizer(s)

        return s


class FixTextNormalizer(base.LoggedCallable):
    """Fix issues in and normalize the character encoding."""

    def apply(
            self,
            s: str
    ) -> str:
        """Return ``s`` with the character encoding normalized.

        Parameters
        ----------
        s : str, required
            The input string to normalize.

        Returns
        -------
        str
            ``s`` with common issues fixed and the character encoding
            normalized.
        """
        return ftfy.fix_text(s)


class GonnaGottaWannaNormalizer(base.LoggedCallable):
    """Normalize words like gonna, gotta, and wanna."""

    _pattern_replacements = [
        ('gonna', 'going to'),
        ('gotta', 'got to'),
        ('wanna', 'want to'),
        ('coulda', 'could have'),
        ('woulda', 'would have'),
        ('shoulda', 'should have')
    ]
    # compile _patterns with regular expressions to match the lexical
    # items
    _pattern_replacements = [
        (regex.compile(r'\m{}\M'.format(x)), y)
        for x, y in _pattern_replacements
    ]

    def apply(
            self,
            s: str
    ) -> str:
        """Return ``s`` with words like gonna / wanna normalized.

        Parameters
        ----------
        s : str, required
            The input string to normalize.

        Returns
        -------
        str
            ``s`` with all words like gonna / wanna normalized.
        """
        for pattern, replacement in self._pattern_replacements:
            s = pattern.sub(replacement, s)

        return s


class RemoveAgeGenderMarkersNormalizer(base.LoggedCallable):
    """Remove age-gender markers from titles, like "(23m)"."""

    _pattern = regex.compile(
        # age followed by gender : (27M), (27, M), ...
        r' ?\p{Ps}\d{1,3}[,.:\/]? ?(?i:m|f)\p{Pe}'
        # gender followed by age : [M27], (M, 27), ...
        r'| ?\p{Ps}(?i:m|f)[,.:\/]? ?\d{1,3}\p{Pe}'
        # gender only            : {m}, (f), ...
        r'| ?\p{Ps}(?i:m|f)\p{Pe}'
        # age only               : [27], (8), ...
        r'| ?\p{Ps}\d{1,3}\p{Pe}'
        # no brackets, age then gender : 27M, 29f, ...
        r'| ?\d{1,3}(?i:m|f)'
        # no brackets, gender then age : m27, f29, ...
        r'| ?(?i:m|f)\d{1,3}')

    def apply(
            self,
            s: str
    ) -> str:
        """Return ``s`` with age / gender markers removed.

        Parameters
        ----------
        s : str, required
            The input string to normalize.

        Returns
        -------
        str
            ``s`` with all age / gender markers removed.
        """
        return self._pattern.sub('', s)


class StripWhitespaceAndPunctuationNormalizer(base.LoggedCallable):
    """Strip leading and trailing whitespace and punctuation.

    Strip leading and trailing whitespace and punctuation, except for
    matched punctuation characters such as quotes and brackets.
    """

    _leading_pattern = regex.compile(
        r'^(?:[\p{Pc}\p{Pd}\p{S}\s]|[^\P{Po}\'"])*')
    _trailing_pattern = regex.compile(
        r'(?:[\p{Pc}\p{Pd}\p{S}\s]|[^\P{Po}\'"])*$')

    def apply(
            self,
            s: str
    ) -> str:
        """Return ``s`` with whitespace and punctuation stripped.

        Parameters
        ----------
        s : str, required
            The input string to normalize.

        Returns
        -------
        str
           ``s`` with leading and trailing whitespace and punctuation
           stripped except for matched characters like quotes and
           brackets.
        """
        s = self._leading_pattern.sub('', s)
        s = self._trailing_pattern.sub('', s)

        return s


class StripMatchedPunctuationNormalizer(base.LoggedCallable):
    """Remove matched punctuation that wraps the text.

    Remove matched punctuation characters wrapping the entire string
    such as quotes or brackets.
    """

    _bracket_pattern = regex.compile(r'^\p{Ps}(.*)\p{Pe}$')
    _quote_pattern = regex.compile(r'^[\p{Pi}"\'](.*)[\p{Pf}"\']$')

    def apply(
            self,
            s: str
    ) -> str:
        """Return ``s`` with matched punctuation characters stripped.

        Parameters
        ----------
        s : str, required
            The input string to normalize.

        Returns
        -------
        str
            ``s`` with matched punctuation characters, such as quotes or
            brackets, that wrap the entire string stripped.
        """
        maybe_matches = True
        while maybe_matches:
            match = (
                self._bracket_pattern.match(s)
                or self._quote_pattern.match(s)
            )
            if match is not None:
                s = match.group(1)
            else:
                maybe_matches = False

        return s


class StripLeadingAndTrailingParentheticalsNormalizer(base.LoggedCallable):
    """Remove leading and trailing parenthetical phrases.

    Parameters
    ----------
    strip_leading : bool, optional (default=True)
        Whether or not to strip leading parenthetical phrases.
    strip_trailing : bool, optional (default=True)
        Whether or not to strip trailing parenthetical phrases.
    """

    _leading_pattern = regex.compile(r'^\p{Ps}.*?\p{Pe}(.*)$')
    _trailing_pattern = regex.compile(r'^(.*)\p{Ps}.*?\p{Pe}$')

    def __init__(
            self,
            strip_leading: bool = True,
            strip_trailing: bool = True,
            *args,
            **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.strip_leading = strip_leading
        self.strip_trailing = strip_trailing

    def apply(
            self,
            s: str
    ) -> str:
        """Return ``s`` with leading / trailing parentheticals removed.

        Parameters
        ----------
        s : str, required
            The input string to normalize.

        Returns
        -------
        str
            ``s`` with leading and trailing parenthetical phrases
            removed.
        """
        maybe_matches = True
        while maybe_matches:
            match = None

            if self.strip_leading:
                match = self._leading_pattern.match(s)
                if match is not None:
                    s = match.group(1).lstrip()

            if self.strip_trailing and match is None:
                match = self._trailing_pattern.match(s)
                if match is not None:
                    s = match.group(1).rstrip()

            maybe_matches = match is not None

        return s


class RemovePostTypeNormalizer(base.LoggedCallable):
    """Remove the post type initialism (AITA / WIBTA)."""

    _pattern = regex.compile(r'^(?i:AITA|WIBTA){e<=1}\M')

    def apply(
            self,
            s: str
    ) -> str:
        """Return ``s`` with the post type initialism removed.

        Parameters
        ----------
        s : str, required
            The input string to normalize.

        Returns
        -------
        str
            ``s`` with the post type initialism removed (AITA / WIBTA).
        """
        return self._pattern.sub('', s)


class RemoveExpandedPostTypeNormalizer(base.LoggedCallable):
    """Remove expansions of the post type initialisms (AITA / WIBTA)."""

    _pattern = regex.compile(
        # forms of "am I the a-hole"
        r'^(?i:(?:am|are|was|were) (?:i|we) (?:a|an|the) (?:assholes?|a-?holes?)){e<=1}\s*'
        r'|^(?i:(?:i|we) (?:a|an|the) (?:assholes?|a-?holes?)){e<=1}\s*'
        # forms of "would I be the a-hole"
        r'|^(?i:(?:would) (?:i|we) (?:be|have been) (?:a|an|the) (?:assholes?|a-?holes?)){e<=1}\s*'
        r'|^(?i:(?:i|we) (?:be|have been) (?:a|an|the) (?:assholes?|a-?holes?)){e<=1}\s*'
        r'|^(?i:(?:be|have been) (?:a|an|the) (?:assholes?|a-?holes?)){e<=1}\s*'
        # shared suffixes
        r'|^(?i:(?:a|an|the) (?:assholes?|a-?holes?)){e<=1}\s*'
        r'|^(?i:assholes?|a-?holes?){e<=1}\s*'
    )

    def apply(
            self,
            s: str
    ) -> str:
        """Return ``s`` with post type indicators removed.

        Return ``s`` with expansions or partial expansions of the post
        type initialisms removed.

        Parameters
        ----------
        s : str, required
            The input string to normalize.

        Returns
        -------
        str
            ``s`` with expansions of the post type initialisms removed.
        """
        return self._pattern.sub('', s)


class WhitespaceNormalizer(base.LoggedCallable):
    """Normalize whitespace."""

    _pattern = regex.compile(r'\s+', flags=regex.M)

    def apply(
            self,
            s: str
    ) -> str:
        """Return ``s`` with all whitespace normalized.

        Return ``s`` with leading and trailing whitespace stripped, and
        consecutive whitespace characters (including new lines) replaced
        by a single space.

        Parameters
        ----------
        s : str, required
            The input string to normalize.

        Returns
        -------
        str
            ``s`` with consecutive whitespace characters replaced by a
            single space.
        """
        s = s.strip()
        s = self._pattern.sub(' ', s)

        return s


class CapitalizationNormalizer(base.LoggedCallable):
    """Normalize capitalization."""

    _nlp = spacy.load('en', disable=['ner', 'parser'])

    @classmethod
    def _normalize_token(
            cls,
            t: spacy.tokens.token.Token
    ) -> str:
        if t.tag_ in ['NNP', 'NNPS']:
            s = t.text_with_ws
        elif t.tag_ == 'PRP' and t.text.lower() == 'i':
            s = t.text_with_ws.upper()
        else:
            s = t.text_with_ws.lower()

        return s

    def apply(
            self,
            s: str
    ) -> str:
        """Return ``s`` with capitalization normalized.

        Return ``s`` with the text lowercased except for the personal
        pronoun "I", and proper nouns. The first letter of sentences
        will be lowercased.

        Parameters
        ----------
        s : str, required
            The input string to normalize.

        Returns
        -------
        str
            ``s`` with capitalization normalized, i.e. only (possibly)
            the first letter of sentences and proper nouns.
        """
        return ''.join(
            self._normalize_token(t)
            for t in self._nlp(s))

"""Code for extracting and representing actions.

Actions are textual descriptions of actions people can take for which
they can be considered culpable / responsible.
"""

from typing import Optional, Tuple

import attr
import regex

from . import utils as data_utils
from ..extraction import filters
from ..extraction import normalizers
from ..extraction import transformers
from ..extraction.base import Case


# action extraction helper code

class GerundPhraseCase(Case):
    """Extract action descriptions from gerund phrases."""

    _pattern = regex.compile(
        r'^(?i:(?:if )?I\'?m )?((?i:not )?[-\w]*?ing\M.*)$')

    def match(
            self,
            x: str
    ) -> Tuple[Optional[str], bool]:
        match = self._pattern.match(x)
        if match is None:
            return (None, False)

        return (match.group(1), True)

    def transform(
            self,
            x: str
    ) -> str:
        return x

    def filter(
            self,
            x: str
    ) -> bool:
        return False


class PrepositionalPhraseCase(Case):
    """Extract action descriptions from prepositional phrases."""

    _pattern = regex.compile(r'^(?i:for|by|after) (.*)$')

    def match(
            self,
            x: str
    ) -> Tuple[Optional[str], bool]:
        match = self._pattern.match(x)
        if match is None:
            return (None, False)

        return (match.group(1), True)

    def transform(
            self,
            x: str
    ) -> str:
        return x

    def filter(
            self,
            x: str
    ) -> bool:
        return False


class IPhraseCase(Case):
    """Extract action descriptions from "I" phrases."""

    _pattern = regex.compile(
        r"^(?i:(?:if|cause|because|that|when|since) )?"
        r"((?i:I(?:'?m)?|we(?:'re)?) .*)$")
    _transformer = transformers.GerundifyingTransformer()

    def match(
            self,
            x: str
    ) -> Tuple[Optional[str], bool]:
        match = self._pattern.match(x)
        if match is None:
            return (None, False)

        return (match.group(1), True)

    def transform(
            self,
            x: str
    ) -> str:
        return self._transformer(x)

    def filter(
            self,
            x: str
    ) -> bool:
        return False


class InfinitivePhraseCase(Case):
    """Extract action descriptions from infinitive phrases."""

    _pattern = regex.compile(r'^((?i:to) .*)$')
    _transformer = transformers.GerundifyingTransformer()

    def match(
            self,
            x: str
    ) -> Tuple[Optional[str], bool]:
        match = self._pattern.match(x)
        if match is None:
            return (None, False)

        return (match.group(1), True)

    def transform(
            self,
            x: str
    ) -> str:
        return self._transformer(x)

    def filter(
            self,
            x: str
    ) -> bool:
        return False


# main class

@attr.s(frozen=True, kw_only=True)
class Action:
    """A class representing an action.

    An action is something that a person can do for which they may be
    considered culpable / responsible.

    Attributes
    ----------
    normativity : float
       A float between zero and one describing the proportion of people
       that would rate the action as not norm-violating. Closer to zero
       means more people view it as violating a norm.
    is_good : bool
       ``True`` if the action is considered a good candidate for
       creating an instance in the resource, otherwise ``False``.

    See `Parameters`_ for additional attributes.

    Parameters
    ----------
    description : str, required
        A textual description of the action.
    pronormative_score : int, required
        A score describing the pronormativity of the action. The
        pronormative score should be the number of people observed
        saying that the action is not norm-violating.
    contranormative_score : int, required
        A score describing the contranormativity of the action. The
        contranormative score should be the number of people observed
        saying that the action is norm-violating.
    """
    # hidden class attributes
    _TITLE_NORMALIZERS = [
        normalizers.FixTextNormalizer(),
        normalizers.GonnaGottaWannaNormalizer(),
        normalizers.RemoveAgeGenderMarkersNormalizer(),
        normalizers.ComposedNormalizer(
            normalizers=[
                normalizers.StripWhitespaceAndPunctuationNormalizer(),
                normalizers.StripMatchedPunctuationNormalizer(),
                normalizers.StripLeadingAndTrailingParentheticalsNormalizer(
                    strip_trailing=False),
                normalizers.WhitespaceNormalizer()
            ]
        ),
        normalizers.RemovePostTypeNormalizer(),
        normalizers.ComposedNormalizer(
            normalizers=[
                normalizers.StripWhitespaceAndPunctuationNormalizer(),
                normalizers.StripMatchedPunctuationNormalizer(),
                normalizers.StripLeadingAndTrailingParentheticalsNormalizer(
                    strip_trailing=False),
                normalizers.WhitespaceNormalizer()
            ]
        ),
        normalizers.RemoveExpandedPostTypeNormalizer(),
        normalizers.ComposedNormalizer(
            normalizers=[
                normalizers.StripWhitespaceAndPunctuationNormalizer(),
                normalizers.StripMatchedPunctuationNormalizer(),
                normalizers.StripLeadingAndTrailingParentheticalsNormalizer(),
                normalizers.WhitespaceNormalizer()
            ]
        ),
        normalizers.CapitalizationNormalizer()
    ]
    _TITLE_PRE_FILTERS = [
        filters.EmptyStringFilter(),
        filters.TooFewCharactersFilter(min_chars=10),
        filters.TooFewWordsFilter(min_words=2),
        filters.PrefixFilter(prefix='apparently', case_sensitive=False)
    ]
    _TITLE_CASES = [
        GerundPhraseCase(),
        PrepositionalPhraseCase(),
        IPhraseCase(),
        InfinitivePhraseCase()
    ]
    _TITLE_POST_FILTERS = [
        filters.EmptyStringFilter(),
        filters.TooFewCharactersFilter(min_chars=10),
        filters.TooFewWordsFilter(min_words=2),
        filters.PrefixFilter(prefix='apparently', case_sensitive=False),
        filters.StartsWithGerundFilter(),
        filters.WhWordFilter()
    ]

    # content
    description: str = attr.ib(
        validator=attr.validators.instance_of(str),
        converter=str)

    # normativity scores
    pronormative_score: int = attr.ib(
        validator=attr.validators.instance_of(int),
        converter=int)
    contranormative_score: int = attr.ib(
        validator=attr.validators.instance_of(int),
        converter=int)

    # computed properties

    @data_utils.cached_property
    def normativity(self) -> float:
        if self.pronormative_score + self.contranormative_score == 0:
            return float('nan')

        return (
            self.pronormative_score
            / (self.pronormative_score + self.contranormative_score)
        )

    @data_utils.cached_property
    def is_good(self) -> bool:
        return (
            self.pronormative_score > 0
            or self.contranormative_score > 0
        )

    # methods

    @classmethod
    def extract_description_from_title(
            cls,
            title: str
    ) -> Optional[str]:
        text = title

        # normalize the text
        for normalizer in cls._TITLE_NORMALIZERS:
            # iterate the normalizer on text until it converges to a
            # fixed point
            prev_text = None
            while prev_text != text:
                prev_text = text
                text = normalizer(prev_text)

        # see if any pre-filters apply to the text
        if any(does_filter(text) for does_filter in cls._TITLE_PRE_FILTERS):
            return None

        # run case-by-case processing
        for case in cls._TITLE_CASES:
            case_text, success = case(text)
            if success:
                text = case_text
                break
        else:
            # no case matched text
            return None

        # see if any post-filters apply to the text
        if any(does_filter(text) for does_filter in cls._TITLE_POST_FILTERS):
            return None

        return text

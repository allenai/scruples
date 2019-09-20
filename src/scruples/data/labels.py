"""Definitions of labels and code for extracting them."""

import enum
from typing import (
    List,
    Optional,
    Tuple)

import regex


# constants

# N.B. this pattern cannot be included as a class attribute on the Label
# class because then the class would try and turn it into one of the
# values in the enumeration.
_CONTRADICTION_PATTERN = regex.compile(
    r'\m(?i:but|however|although|yet){e<=1}\M')


# classes

@enum.unique
class Label(enum.Enum):
    """A label for the anecdote.

    Labels are extracted from the comments for a given post using a set
    of patterns. The labels are an enumeration with the following
    possible values:

      1. **AUTHOR**  : The author of the anecdote is in the wrong.
      2. **OTHER**  : The other person in the anecdote is in the wrong.
      3. **EVERYBODY**  : Everyone in the anecdote is in the wrong.
      4. **NOBODY**  : No one in the anecdote is in the wrong.
      5. **INFO** : More information is required to make a judgment.

    Attributes
    ----------
    See `Parameters`_.

    Parameters
    ----------
    index : int
        A (unique) numerical index assigned to the label.
    reddit_name : str
        The name for the label used on the subreddit from which the data
        originates.
    patterns : List[str]
        A list of strings, each representing a regular expression
        pattern used to extract that label from a comment's body
        text. Note that the patterns are compiled to regular expressions
        when they're bound to the ``Label`` instance as an attribute.
    """
    AUTHOR = (0, 'YTA', [
        r'\m(?i:YTAH?)\M',
        r"(?e)(?i:"
          r"you(?:'re|r| are)? "
          r"(?:(?:kind|sort) of |really |indeed |just )?"
          r"(?:an? |the )?"
          r"(?:huge |big |giant )?"
          r"(?:asshole|a-?hole)"
        r"){e<=1}",
        r"(?e)(?i:"
          r"you "
          r"(?:(?:kind|sort) of |really |indeed |just )?"
          r"are (?:an? |the )?"
          r"(?:huge |big |giant )?"
          r"(?:asshole|a-?hole)"
        r"){e<=1}"
    ])
    OTHER = (1, 'NTA', [
        r'\m(?i:Y?NTAH?)\M',
        r'(?e)(?i:'
          r'not '
          r'(?:really )?'
          r'(an? |the )?'
          r'(asshole|a-?hole)\M'
        r'){e<=1}'
    ])
    EVERYBODY = (2, 'ESH', [
        r'\m(?i:ESH)\M',
        r'(?e)(?i:every(?:one|body) sucks here){e<=1}',
        r'(?e)(?i:you both suck){e<=1}'
    ])
    NOBODY = (3, 'NAH', [
        r'\m(?i:NAH?H)\M',
        r'(?e)(?i:no (?:assholes|a-?holes) here){e<=1}',
        r'(?e)(?i:no one is the (?:asshole|a-?hole)){e<=1}'
    ])
    INFO = (4, 'INFO', [
        r'\m(?i:INFO)\M',
        r'(?e)(?i:not enough info){e<=1}',
        r'(?e)(?i:needs? more info){e<=1}',
        r"(?e)(?i:more info(?:'s| is)? required){e<=1}"
    ])

    @classmethod
    def extract_from_text(
            cls,
            text: str
    ) -> Optional['Label']:
        """Return a label extracted from ``text`` or ``None``.

        If a label can be unambiguously extracted from text, return it;
        otherwise, return ``None``.

        Parameters
        ----------
        text : str
            The text from which to extract the label.

        Returns
        -------
        Optional[Label]
            The extracted label.
        """
        found_labels = {}
        for label in cls:
            # span is either a tuple containing (start_idx, end_idx) or
            # is None
            span = label.find(text)
            if span is not None:
                found_labels[label] = span

        # return the label now if no conflict resolution is required

        if len(found_labels) == 0:
            return None

        if len(found_labels) == 1:
            return next(iter(found_labels.keys()))

        # multiple labels were matched, so resolve which label is the
        # correct one
        label0_span0, label1_span1 = sorted(
            found_labels.items(),
            # sort by the start of each span
            key=lambda x: x[1][0])[:2]
        label0, (start0, end0) = label0_span0
        label1, (start1, end1) = label1_span1

        # if the utterances for the labels overlap, pick the earlier
        # label
        if end0 > start1:
            return label0

        between_text = text[end0:start1]
        if _CONTRADICTION_PATTERN.search(between_text) is None:
            # no contradiction / signs of switching opinion, return the
            # earlier label
            return label0
        else:
            # there's indication of a change of opinion, so return the
            # later label
            return label1

    def __init__(
            self,
            index: int,
            reddit_name: str,
            patterns: List[str]
    ) -> None:
        self.index = index
        self.reddit_name = reddit_name
        self.patterns = [regex.compile(pattern) for pattern in patterns]

    def find(
            self,
            text: str
    ) -> Optional[Tuple[int, int]]:
        """Return the first span denotating the label in ``text``.

        If ``text`` contains a span denoting this label, then return
        the first one (in order of the starting index of each
        span). Otherwise, return ``None``.

        Parameters
        ----------
        text : str
            The text to check for the label.

        Returns
        -------
        Optional[Tuple[int, int]]
            If ``text`` contains a span denoting this label, a tuple
            providing the start and end indices for the first such span
            (in order of the spans' starting indices) otherwise
            ``None``.
        """
        span = None
        for pattern in self.patterns:
            match = pattern.search(text)
            if match is None:
                continue

            if span is None:
                span = (match.start(), match.end())
            elif match.start() < span[0]:
                span = (match.start(), match.end())
            else:
                pass

        return span


@enum.unique
class BinarizedLabel(enum.Enum):
    """A binary label saying whether the author was in the wrong.

    The binarized labels group together the full labels in the following
    way:

      1. **RIGHT** : The full label is ``OTHER`` or ``NOBODY``.
      2. **WRONG** : The full label is ``AUTHOR`` or ``EVERYBODY``.

    The label ``INFO`` is dropped during binarization.

    Attributes
    ----------
    See `Parameters`_.

    Parameters
    ----------
    index : int
        A (unique) numerical index assigned to the binarized label.
    """
    RIGHT = 0
    WRONG = 1

    @classmethod
    def binarize(
            cls,
            label: Label
    ) -> Optional['BinarizedLabel']:
        """Return ``label`` binarized into a ``BinarizedLabel``.

        Parameters
        ----------
        label : Label
            The label to binarize.

        Returns
        -------
        Optional[BinarizedLabel]
            ``BinarizedLabel.RIGHT`` if ``label`` is ``Label.OTHER`` or
            ``Label.NOBODY``, ``BinarizedLabel.WRONG`` if ``label`` is
            ``Label.AUTHOR`` or ``Label.EVERYBODY``. ``None`` if
            ``label`` is ``Label.INFO``.
        """
        if not isinstance(label, Label):
            raise ValueError(f'label ({label}) is not of type Label.')

        if label == Label.OTHER or label == Label.NOBODY:
            return cls.RIGHT
        elif label == Label.AUTHOR or label == Label.EVERYBODY:
            return cls.WRONG
        elif label == Label.INFO:
            return None
        else:
            raise ValueError(f'Unrecognized label: {label}.')

    def __init__(
            self,
            index: int
    ) -> None:
        self.index = index

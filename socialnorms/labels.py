"""Definitions of labels and code for extracting them."""

import enum
from typing import (
    List,
    Optional)

import regex


@enum.unique
class Label(enum.Enum):
    """A label for the anecdote.

    Labels are extracted from the comments for a given post using a set
    of patterns. The labels are an enumeration with the following
    possible values:

      1. **YTA**  : The author of the anecdote is in the wrong.
      2. **NTA**  : The other person in the anecdote is in the wrong.
      3. **ESH**  : Everyone in the anecdote is in the wrong.
      4. **NAH**  : No one in the anecdote is in the wrong.
      5. **INFO** : More information is required to make a judgment.

    Attributes
    ----------
    See `Parameters`_.

    Parameters
    ----------
    index : int
        A (unique) numerical index assigned to the label.
    patterns : List[str]
        A list of strings, each representing a regular expression
        pattern used to extract that label from a comment's body
        text. Note that the patterns are compiled to regular expressions
        when they're bound to the ``Label`` instance as an attribute.
    """
    YTA = (0, [
        r'\m(?i:YTAH?)\M',
        r"(?i:you(?:'re|r| are)? (?:(?:kind|sort) of |really )?(?:an? |the )?(?:asshole|a-?hole)){e<=1}"
        r"(?i:you (?:(?:kind|sort) of |really )?are (?:an? |the )?(?:asshole|a-?hole)){e<=1}"
    ])
    NTA = (1, [
        r'\m(?i:Y?NTAH?)\M',
        r'(?i:not (?:really )?(an? |the )?(asshole|a-?hole)){e<=1}',
    ])
    ESH = (2, [
        r'\m(?i:ESH)\M',
        r'(?i:every(?:one|body) sucks here){e<=1}',
    ])
    NAH = (3, [
        r'\m(?i:NAH?H)\M',
        r'(?i:no (?:assholes|a-?holes) here){e<=1}'
    ])
    INFO = (4, [
        r'\m(?i:INFO)\M',
        r'(?i:not enough info){e<=1}'
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
        found_labels = set()
        for label in cls:
            if label.in_(text):
                found_labels.add(label)

        return found_labels.pop() if len(found_labels) == 1 else None

    def __init__(
            self,
            index: int,
            patterns: List[str]
    ) -> None:
        self.index = index
        self.patterns = [regex.compile(pattern) for pattern in patterns]

    def in_(
            self,
            text: str
    ) -> bool:
        """Return ``True`` if the label can be found in ``text``.

        Return ``True`` if label has any pattern that matches a
        substring from ``text``.

        Parameters
        ----------
        text : str
            The text to check for the label.

        Returns
        -------
        bool
            Whether or not the label is in ``text``.
        """
        return any(pattern.search(text) for pattern in self.patterns)

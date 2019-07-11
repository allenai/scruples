"""Definitions of post types and code for extracting them."""

import enum
from typing import (
    List,
    Optional)

import regex


@enum.unique
class PostType(enum.Enum):
    """A post type.

    Posts are categorized into three types:

      1. **HISTORICAL**  : The author is asking if they're in the wrong, based
         on an event that has actually happened.
      2. **HYPOTHETICAL** : The author is asking if they would be in the wrong,
         if they were to perform some action.
      3. **META**  : The post is discussing the subreddit itself.

    Attributes
    ----------
    See `Parameters`_.

    Parameters
    ----------
    index : int
        A (unique) numerical index assigned to the post type.
    reddit_name : str
        The name for the post type used on the subreddit from which the
        data originates.
    patterns : List[str]
        A list of strings, each representing a regular expression
        pattern used to extract that post type from a post's title. Note
        that patterns are compiled to regular expressions when they're
        bound to the ``PostType`` instance as an attribute.
    """
    HISTORICAL = (0, 'AITA', [
        r'\m(?i:AITAH?)\M',
        r'(?i:Am I the asshole){e<=2}'
    ])
    HYPOTHETICAL = (1, 'WIBTA', [
        r'\m(?i:WIBTAH?)\M',
        r'(?i:Would I be the asshole){e<=2}'
    ])
    META = (2, 'META', [
        r'\mMETA\M',
        r'\[(?i:META)\]'
    ])

    @classmethod
    def extract_from_title(
            cls,
            title: str
    ) -> Optional['PostType']:
        """Return a post type extracted from ``title`` or ``None``.

        If a post type can be unambiguously extracted from the title,
        return it; otherwise, return ``None``.

        Parameters
        ----------
        title : str
            The title string from which to extract the post type.

        Returns
        -------
        Optional[PostType]
            The extracted post type.
        """
        found_post_types = set()
        for post_type in cls:
            if post_type.in_(title):
                found_post_types.add(post_type)

        return found_post_types.pop() if len(found_post_types) == 1 else None

    def __init__(
            self,
            index: int,
            reddit_name: str,
            patterns: List[str]
    ) -> None:
        self.index = index
        self.reddit_name = reddit_name
        self.patterns = [regex.compile(pattern) for pattern in patterns]

    def in_(
            self,
            title: str
    ) -> bool:
        """Return ``True`` if ``title`` expresses the post type.

        Return ``True`` if the post type has any pattern that matches a
        substring from ``title``.

        Parameters
        ----------
        title : str
            The title string to check for the post type.

        Returns
        -------
        bool
            Whether or not the post type is in ``title``.
        """
        return any(pattern.search(title) for pattern in self.patterns)

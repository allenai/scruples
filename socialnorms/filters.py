"""Filters to remove bad data.

This module contains filters for removing bad comments, posts, and
labels from the final dataset. A filter is a predicate that takes the
proper data type and returns ``True`` if the data should be kept and
``False`` if it should be removed.
"""

from typing import Dict

from .data import Comment, Post
from .labels import Label
from .utils import count_words


# filters for comments

def comment_is_top_level(
        comment: Comment
) -> bool:
    """Return ``True`` if ``comment`` is a top-level comment."""
    return comment.parent_id == comment.link_id


# N.B. place cheaper filters earlier so code can short-circuit during
# filtering to avoid evaluating more expensive filters.
COMMENT_FILTERS = [
    comment_is_top_level
]
"""The list of comment filters used for creating the dataset."""


# filters for posts

def post_not_deleted(
        post: Post
) -> bool:
    """Return ``True`` if the post not been deleted."""
    return not (
        post.selftext == '[deleted]'
        or post.selftext == '[removed]'
    )


def post_is_self(
        post: Post
) -> bool:
    """Return ``True`` if the post is a self-post."""
    return post.is_self


def post_selftext_is_not_empty(
        post: Post
) -> bool:
    """Return ``True`` if the post selftext is a non-empty string."""
    return post.selftext != ""


def post_has_enough_content(
        post: Post
) -> bool:
    """Return ``True`` if the post has enough content tokens.

    Return ``True`` if the post has more content tokens (tokens in the
    title plus in the selftext) than a certain threshold.
    """
    content = post.title + ' ' + post.selftext

    return count_words(content) >= 16


# N.B. place cheaper filters earlier so code can short-circuit during
# filtering to avoid evaluating more expensive filters.
POST_FILTERS = [
    post_is_self,
    post_selftext_is_not_empty,
    post_not_deleted,
    post_has_enough_content
]
"""The list of post filters used for creating the dataset."""


# filters for label scores

def label_scores_has_nonzero_elements(
        label_scores: Dict[Label, int]
) ->  bool:
    """Return ``True`` if there's any non-zero label score."""
    return any(v != 0 for v in label_scores.values())


def label_scores_has_one_highest_scoring_label(
        label_scores: Dict[Label, int]
) -> bool:
    """Return ``True`` if one label score is higher than all others."""
    max_score = max(label_scores.values())
    return sum(v == max_score for v in label_scores.values()) == 1


# N.B. place cheaper filters earlier so code can short-circuit during
# filtering to avoid evaluating more expensive filters.
LABEL_SCORES_FILTERS = [
    label_scores_has_nonzero_elements,
    label_scores_has_one_highest_scoring_label
]
"""The list of label score filters used for creating the dataset."""

"""A class representing a comment."""

from typing import Optional

import attr

from .. import settings
from . import utils
from .labels import Label


@attr.s(frozen=True, kw_only=True)
class Comment:
    """A class representing a comment.

    Attributes
    ----------
    label : Optional[Label]
        The label expressed by the comment. If no label can be extracted
        or multiple labels are extracted from the comment, then this
        attribute is ``None``.
    is_top_level : bool
        ``True`` if the comment is a top-level comment (i.e., a direct
        response to a link and not another comment).
    has_empty_body : bool
        ``True`` if the body text of the comment is empty.
    is_deleted : bool
        ``True`` if the comment is deleted.
    is_by_automoderator : bool
        ``True`` if the comment is by the AutoModerator.
    is_good : bool
        ``True`` if the comment is a good candidate for contributing a
        label.

    See `Parameters`_ for additional attributes.

    Parameters
    ----------
    id : str
        A unique ID for the comment.
    subreddit_id : str
        The ID of the subreddit the comment was posted in. The ID has
        ``"t5_"`` prepended to it to represent the fact that it is a
        *subreddit* ID.
    subreddit : str
        The name of the subreddit the comment was posted in.
    link_id : str
        The ID of the post that the comment was made on. The ID has
        ``"t3_"`` prepended to it to represent the fact that it is a
        *post* ID.
    parent_id : str
        The ID of the parent object (either a comment or a post). If the
        parent object is a post, the ID will begin with ``"t3_"``. If
        the parent object is a comment, the ID will begin with
        ``"t1_"``.
    created_utc : int
        The time that the comment was created in seconds since the
        epoch.
    author : str
        The username of the comment's author.
    body : str
        The body text of the comment.
    score : int
        The score (upvotes minus downvotes) of the comment.
    controversiality : int
        The controversiality score for the comment.
    gilded : int
        The number of times the comment has been gilded.
    """
    # identifying information
    id: str = attr.ib(
        validator=attr.validators.instance_of(str),
        converter=str)

    # reddit location
    subreddit_id: str = attr.ib(
        validator=attr.validators.instance_of(str),
        converter=str)
    subreddit: str = attr.ib(
        validator=attr.validators.instance_of(str),
        converter=str)
    link_id: str = attr.ib(
        validator=attr.validators.instance_of(str),
        converter=str)
    parent_id: str = attr.ib(
        validator=attr.validators.instance_of(str),
        converter=str)

    # creation
    created_utc: int = attr.ib(
        validator=attr.validators.instance_of(int),
        converter=int)
    author: str = attr.ib(
        validator=attr.validators.instance_of(str),
        converter=str)

    # content
    body: str = attr.ib(
        validator=attr.validators.instance_of(str),
        converter=str)

    # user interactions
    score: int = attr.ib(
        validator=attr.validators.instance_of(int),
        converter=int)
    controversiality: int = attr.ib(
        validator=attr.validators.instance_of(int),
        converter=int)
    gilded: int = attr.ib(
        validator=attr.validators.instance_of(int),
        converter=int)

    # computed content properties

    @utils.cached_property
    def label(self) -> Optional[Label]:
        return Label.extract_from_text(self.body)

    # computed properties for identifying comments to count in label
    # scores

    @utils.cached_property
    def is_top_level(self) -> bool:
        return self.parent_id == self.link_id

    @utils.cached_property
    def has_empty_body(self) -> bool:
        return self.body == ""

    @utils.cached_property
    def is_deleted(self) -> bool:
        return (
            self.body == '[deleted]'
            or self.body == '[removed]'
        )

    @utils.cached_property
    def is_by_automoderator(self) -> bool:
        return self.author == settings.AUTO_MODERATOR_NAME

    @utils.cached_property
    def is_good(self) -> bool:
        # N.B. place cheaper predicates earlier so short-circuiting can
        # avoid evaluating more expensive predicates.
        return (
            self.is_top_level
            and not self.has_empty_body
            and not self.is_deleted
            and not self.is_by_automoderator
        )

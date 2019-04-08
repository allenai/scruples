"""A class representing a post."""

from typing import (
    List,
    Optional)

import attr

from .. import utils
from . import utils as data_utils
from .comment import Comment
from .label_scores import LabelScores
from .labels import Label
from .post_types import PostType


@attr.s(frozen=True, kw_only=True)
class Post:
    """A class representing a post.

    Attributes
    ----------
    label_scores : LabelScores
        The label scores for the post. Scores are computed by summing
        one vote for each comment expressing a particular label.
    original_text : Optional[str]
        The original text of the post or ``None``. The post in it's
        original form is usually captured by the AutoModerator for the
        subreddit. If the original text can be found in the comments,
        then it will captured in this attribute; otherwise, the
        attribute is ``None``.
    post_type : Optional[PostType]
        The type of post the post is. If no post type can be extracted
        or multiple post types are extracted for the post, then this
        attribute is ``None``.
    has_empty_selftext : bool
        ``True`` if the post's selftext is empty.
    is_deleted : bool
        ``True`` if the post is deleted.
    has_post_type : bool
        ``True`` if the post's post type is not ``None``.
    is_meta : bool
        ``True`` if the post has the 'META' post type.
    has_original_text : bool
        ``True`` if the post's original text attribute is not ``None``,
        i.e. if the original post text was successfully found and
        extracted from the comments.
    has_enough_content : bool
        ``True`` if the post has more content tokens (tokens in the
        title plus in the original text or selftext if it's not
        available) than a certain threshold.
    has_good_label_scores : bool
       ``True`` if the post's label scores object is good, in other
       words, if the label scores object is considered a good set of
       label scores for a dataset instance.
    is_good : bool
        ``True`` if the post is considered a good candidate for creating
        an instance for the dataset.

    See `Parameters`_ for additional attributes.

    Parameters
    ----------
    id : str
        A unique ID for the post.
    subreddit_id : str
        The ID of the subreddit the post was posted in. The ID has
        ``"t5_"`` prepended to it to represent the fact that it is a
        *subreddit* ID.
    subreddit : str
        The name of the subreddit the post was posted in.
    permalink : str
        The permanent URL for the post relative to reddit's site.
    domain : str
        The domain that the post's link points to.
    url : str
        The URL that the post's link points to.
    created_utc : int
        The time that the post was created in seconds since the epoch.
    author : str
        The username of the author of the post.
    title : str
        The title of the post.
    selftext : str
        The text of the post.
    thumbnail : str
        Full URL to the thumbnail for the post or "default" for the
        default thumbnail (non-image posts), "self" if it is a
        self-post, or "image" if the post is an image but has no
        thumbnail available.
    comments : List[Comment]
        A list of comments made on the post.
    score : int
        The score of the post (upvotes minus downvotes).
    num_comments : int
        The number of comments made on the post.
    gilded : int
        The number of times the post has been gilded.
    retrieved_on : int
        The time the post was retrieved in seconds since the epoch.
    archived : bool
        Whether or not the post has been archived.
    is_self : bool
        Whether or not the post is a self-post.
    over_18 : bool
        Whether or not the post has been marked as NSFW (not safe for
        work).
    stickied : bool
        Whether or not the post has been stickied to the top of the
        subreddit.
    """
    # hidden class attributes
    _ORIGINAL_COMMENT_PREFIX_AND_SUFFIX = (
        # prefix for the comment archiving the original post
        '^^^^AUTOMOD  ***This is a copy of the above post. It is a'
        ' record of the post as originally written, in case the post is'
        ' deleted or edited.***\n\n',
        # (optional) suffix for the comment archiving the original post
        '\n\n*I am a bot, and this action was performed'
        ' automatically. Please [contact the moderators of this'
        ' subreddit](/message/compose/?to=/r/AmItheAsshole) if you have'
        ' any questions or concerns.*'
    )

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

    # location
    permalink: str = attr.ib(
        validator=attr.validators.instance_of(str),
        converter=str)

    # link information
    domain: str = attr.ib(
        validator=attr.validators.instance_of(str),
        converter=str)
    url: str = attr.ib(
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
    title: str = attr.ib(
        validator=attr.validators.instance_of(str),
        converter=str)
    selftext: str = attr.ib(
        validator=attr.validators.instance_of(str),
        converter=str)
    thumbnail: str = attr.ib(
        validator=attr.validators.instance_of(str),
        converter=str)
    comments: List[Comment] = attr.ib(
        validator=attr.validators.deep_iterable(
            attr.validators.instance_of(Comment)))

    # user interactions
    score: int = attr.ib(
        validator=attr.validators.instance_of(int),
        converter=int)
    num_comments: int = attr.ib(
        validator=attr.validators.instance_of(int),
        converter=int)
    gilded: int = attr.ib(
        validator=attr.validators.instance_of(int),
        converter=int)

    #  miscellaneous
    retrieved_on: int = attr.ib(
        validator=attr.validators.instance_of(int),
        converter=int)
    archived: bool = attr.ib(
        validator=attr.validators.instance_of(bool),
        converter=bool)
    is_self: bool = attr.ib(
        validator=attr.validators.instance_of(bool),
        converter=bool)
    over_18: bool = attr.ib(
        validator=attr.validators.instance_of(bool),
        converter=bool)
    stickied: bool = attr.ib(
        validator=attr.validators.instance_of(bool),
        converter=bool)

    # computed content properties

    @data_utils.cached_property
    def label_scores(self) -> LabelScores:
        label_to_score = {label: 0 for label in Label}
        for comment in self.comments:
            if not comment.is_good:
                continue

            if comment.label:
                label_to_score[comment.label] += 1

        return LabelScores(label_to_score=label_to_score)

    @data_utils.cached_property
    def original_text(self) -> Optional[str]:
        prefix, suffix = self._ORIGINAL_COMMENT_PREFIX_AND_SUFFIX
        original_text = None
        for comment in self.comments:
            if (
                    comment.is_by_automoderator
                    and comment.body.startswith(prefix)
            ):
                if comment.body.endswith(suffix):
                    original_text = comment.body[len(prefix):-len(suffix)]
                else:
                    original_text = comment.body[len(prefix):]

        return original_text

    @data_utils.cached_property
    def post_type(self) -> Optional[PostType]:
        return PostType.extract_from_title(self.title)

    # computed properties for identifying good instance candidates

    @data_utils.cached_property
    def has_empty_selftext(self) -> bool:
        return self.selftext == ""

    @data_utils.cached_property
    def is_deleted(self) -> bool:
        return (
            self.selftext == '[deleted]'
            or self.selftext == '[removed]'
        )

    @data_utils.cached_property
    def has_post_type(self) -> bool:
        return self.post_type is not None

    @data_utils.cached_property
    def is_meta(self) -> bool:
        return self.post_type == PostType.META

    @data_utils.cached_property
    def has_original_text(self) -> bool:
        return self.original_text is not None

    @data_utils.cached_property
    def has_enough_content(self) -> bool:
        content = self.title + ' ' + (self.original_text or self.selftext)

        return utils.count_words(content) >= 16

    @data_utils.cached_property
    def has_good_label_scores(self) -> bool:
        return self.label_scores.is_good

    @data_utils.cached_property
    def is_good(self) -> bool:
        # N.B. place cheaper predicates earlier so short-circuiting can
        # avoid evaluating more expensive predicates.
        return (
            self.is_self
            and not self.has_empty_selftext
            and not self.is_deleted
            and self.has_post_type
            and not self.is_meta
            and self.has_original_text
            and self.has_enough_content
            and self.has_good_label_scores
        )

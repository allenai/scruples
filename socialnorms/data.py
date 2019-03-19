"""Data models and utilities."""

from typing import (
    Any,
    Dict,
    List)

import attr


# utility functions

def instantiate_attrs_with_extra_kwargs(
        cls: Any,
        **kwargs: Dict[str, Any]
):
    """Return ``cls`` instantiated with ``kwargs`` ignoring extra kwargs.

    Parameters
    ----------
    cls : Object
        An object that has been decorated with ``@attr.s``.
    **kwargs : Dict[str, Any]
        Any keyword arguments to use when instantiating ``cls``. Extra
        keyword arguments will be ignored.
    """
    if not attr.has(cls):
        raise ValueError(f'{cls} must be decorated with @attr.s')

    attr_names = attr.fields_dict(cls).keys()
    return cls(**{
        k: kwargs[k]
        for k in attr_names
    })


# data models

@attr.s(frozen=True, slots=True, kw_only=True)
class Comment:
    """A class representing a comment.

    Attributes
    ----------
    See `Parameters`_.

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


@attr.s(frozen=True, slots=True, kw_only=True)
class Post:
    """A class representing a post.

    Attributes
    ----------
    See `Parameters`_.

    Parameters
    ----------
    id : str
        A unique ID for the post.
    subreddit_id : str
        The ID of the subreddit the comment was posted in. The ID has
        ``"t5_"`` prepended to it to represent the fact that it is a
        *subreddit* ID.
    subreddit : str
        The name of the subreddit the comment was posted in.
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

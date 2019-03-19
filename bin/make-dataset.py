"""Make the social norms dataset from raw reddit data.

This script takes in posts and comments from the reddit API and creates
the social norms dataset.
"""

import collections
import json
import logging
import random
from typing import (
    Dict,
    List)

import click
import tqdm

from socialnorms import settings
from socialnorms.data import (
    Comment,
    Post,
    instantiate_attrs_with_extra_kwargs)
from socialnorms.filters import (
    COMMENT_FILTERS,
    POST_FILTERS,
    LABEL_SCORES_FILTERS)
from socialnorms.labels import Label


logger = logging.getLogger(__name__)


# constants and helper functions

def _extract_label_scores_from_comments(
        comments: List[Comment]
) -> Dict[Label, float]:
    label_to_score = {label: 0. for label in Label}
    for comment in comments:
        label = Label.extract_from_text(comment.body)
        if label:
            label_to_score[label] += comment.score

    return label_to_score


# main function

@click.command()
@click.argument(
    'comments_path',
    type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument(
    'posts_path',
    type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument(
    'output_path',
    type=click.Path(exists=False, file_okay=True, dir_okay=False))
def make_dataset(
        comments_path: str,
        posts_path: str,
        output_path: str
) -> None:
    """Create the social norms dataset and write it to OUTPUT_PATH.

    Read in the reddit posts from POSTS_PATH and comments from
    COMMENTS_PATH, create the social norms dataset, and write it to
    OUTPUT_PATH.
    """
    # Step 1: Read in the comments, filter out bad ones, and index the
    # rest by their link ids.
    link_id_to_comments = collections.defaultdict(list)
    with click.open_file(comments_path, 'r') as comments_file:
        for ln in tqdm.tqdm(comments_file.readlines(), **settings.TQDM_KWARGS):
            kwargs = json.loads(ln)
            comment = instantiate_attrs_with_extra_kwargs(
                Comment,
                **kwargs)

            if all(check(comment) for check in COMMENT_FILTERS):
                # IDs are usually prefixed with something like "t1_",
                # "t2_", etc. to denote what type of object it is. Slice
                # off the first 3 characters to remove this prefix from
                # the link id because it will not be on the posts' IDs
                # when we join the comments to them.
                link_id_to_comments[comment.link_id[3:]].append(comment)

    # Step 2: Read in the posts, filter out the bad ones, and join them
    # with their comments.
    posts = []
    with click.open_file(posts_path, 'r') as posts_file:
        for ln in tqdm.tqdm(posts_file.readlines(), **settings.TQDM_KWARGS):
            kwargs = json.loads(ln)
            post = instantiate_attrs_with_extra_kwargs(
                Post,
                comments=link_id_to_comments[kwargs['id']],
                **kwargs)

            if all(check(post) for check in POST_FILTERS):
                posts.append(post)

    # Step 3: Extract labels with scores from the comments for each
    # post, filter out posts with bad labels, then write the dataset
    # instances to disk.
    with click.open_file(output_path, 'w') as out_file:
        for post in tqdm.tqdm(
                random.sample(posts, len(posts)),
                **settings.TQDM_KWARGS
        ):
            label_scores = _extract_label_scores_from_comments(post.comments)

            if all(check(label_scores) for check in LABEL_SCORES_FILTERS):
                instance = {
                    'id': post.id,
                    'title': post.title,
                    'text': post.selftext,
                    'label_scores': {
                        label.name: score
                        for label, score in label_scores.items()
                    },
                    'label': max(
                        label_scores.items(),
                        key=lambda t: t[1])[0].name
                }

                out_file.write(json.dumps(instance) + '\n')


if __name__ == '__main__':
    make_dataset()

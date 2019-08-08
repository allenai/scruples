"""Estimate human performance for the scruples corpus."""

import collections
import json
import logging
import random

import click
import tqdm

from ... import settings
from ...baselines.metrics import METRICS
from ...data.comment import Comment
from ...data.post import Post
from ...data.labels import Label
from ...data.utils import instantiate_attrs_with_extra_kwargs


logger = logging.getLogger(__name__)


# main function

@click.command()
@click.argument(
    'comments_path',
    type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument(
    'posts_path',
    type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument(
    'split_path',
    type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument(
    'output_path',
    type=click.Path(exists=False, file_okay=True, dir_okay=False))
def corpus_human_performance(
        comments_path: str,
        posts_path: str,
        split_path: str,
        output_path: str
) -> None:
    """Estimate human performance on the scruples corpus.

    Read in the comments from COMMENTS_PATH, posts from POSTS_PATH, and
    the split from SPLIT_PATH, then estimate human performance metrics
    and write them to OUTPUT_PATH.
    """
    logger.info('Reading in comments.')

    link_id_to_comments = collections.defaultdict(list)
    with click.open_file(comments_path, 'r') as comments_file:
        for ln in tqdm.tqdm(comments_file.readlines(), **settings.TQDM_KWARGS):
            comment = instantiate_attrs_with_extra_kwargs(
                Comment,
                **json.loads(ln))

            # IDs are usually prefixed with something like "t1_",
            # "t2_", etc. to denote what type of object it is. Slice
            # off the first 3 characters to remove this prefix from
            # the link id because it will not be on the posts' IDs
            # when we join the comments to them.
            link_id_to_comments[comment.link_id[3:]].append(comment)

    logger.info('Reading in posts.')

    split_post_ids = set()
    with click.open_file(split_path, 'r') as split_file:
        for ln in split_file:
            split_post_ids.add(json.loads(ln)['post_id'])

    posts = []
    with click.open_file(posts_path, 'r') as posts_file:
        for ln in tqdm.tqdm(posts_file.readlines(), **settings.TQDM_KWARGS):
            kwargs = json.loads(ln)
            post = instantiate_attrs_with_extra_kwargs(
                Post,
                comments=link_id_to_comments[kwargs['id']],
                **kwargs)

            if post.id in split_post_ids:
                posts.append(post)

    logger.info('Computing human performance.')

    human_preds = []
    gold_labels = []
    for post in tqdm.tqdm(posts, **settings.TQDM_KWARGS):
        post_labels = [
            comment.label.index
            for comment in post.comments
            if comment.is_good and comment.label is not None
        ]
        random.shuffle(post_labels)

        if len(post_labels) > 1:
            human_preds.append(post_labels[-1])
            gold_labels.append(collections.Counter(
                post_labels[:-1]).most_common(1)[0][0])
        elif len(post_labels) == 1:
            # predict with the majority label
            human_preds.append(Label.OTHER.index)
            gold_labels.append(post_labels[0])
        else:
            raise ValueError('Found a post without a label.')

    with open(output_path, 'w') as metrics_file:
        json.dump({
            key: metric(
                y_true=gold_labels,
                y_pred=human_preds)
            for key, (_, metric, scorer_kwargs) in METRICS.items()
            if not scorer_kwargs['needs_proba']
        }, metrics_file)

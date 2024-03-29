"""Make the scruples corpus from raw reddit data.

This script takes in posts and comments from the reddit API and creates
the scruples dataset.
"""

import collections
import json
import logging
import os
import random

import attr
import click
import tqdm

from ... import settings, utils
from ...data.comment import Comment
from ...data.post import Post
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
    'corpus_dir',
    type=click.Path(exists=False, file_okay=False, dir_okay=True))
def corpus(
        comments_path: str,
        posts_path: str,
        corpus_dir: str
) -> None:
    """Create the scruples corpus and write it to CORPUS_DIR.

    Read in the reddit posts from POSTS_PATH and comments from
    COMMENTS_PATH, create the scruples corpus, and write it to
    CORPUS_DIR.
    """
    # Create the output directory.
    os.makedirs(corpus_dir)

    # Step 1: Read in the comments and index them by their link ids.
    logger.info('Reading the comments.')

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

    # Step 2: Read in the posts and join them with their comments.
    logger.info('Reading the posts.')

    posts = []
    with click.open_file(posts_path, 'r') as posts_file:
        for ln in tqdm.tqdm(posts_file.readlines(), **settings.TQDM_KWARGS):
            kwargs = json.loads(ln)
            post = instantiate_attrs_with_extra_kwargs(
                Post,
                comments=link_id_to_comments[kwargs['id']],
                **kwargs)

            posts.append(post)

    # Step 3: Write the posts to disk.
    logger.info('Writing the posts to disk.')

    processed_posts_path = os.path.join(corpus_dir, settings.POSTS_FILENAME)
    with open(processed_posts_path, 'w') as processed_posts_file:
        for post in posts:
            processed_posts_file.write(json.dumps(attr.asdict(post)) + '\n')

    # Step 4: Filter out bad posts.
    logger.info('Filtering out bad posts.')

    dataset_posts = [
        post
        for post in tqdm.tqdm(posts, **settings.TQDM_KWARGS)
        if post.is_good
    ]

    # Step 5: Create the splits then write them to disk.
    logger.info('Creating splits and writing them to disk.')

    # Shuffle dataset_posts so that the splits will be random.
    random.shuffle(dataset_posts)

    if [split['size'] for split in settings.SPLITS].count(None) > 1:
        raise ValueError(
            'The settings.SPLITS constant should have at most ONE split'
            ' with a size of None.')

    # Make sure that the split with a size of ``None`` will be processed
    # last.
    splits = [
        split
        for split in settings.SPLITS
        if split['size'] is not None
    ] + [
        split
        for split in settings.SPLITS
        if split['size'] is None
    ]
    for split in splits:
        split_path = os.path.join(
            corpus_dir,
            settings.CORPUS_FILENAME_TEMPLATE.format(split=split['name']))
        with open(split_path, 'w') as split_file:
            if split['size'] is None:
                split_posts = dataset_posts
                dataset_posts = []
            else:
                split_posts = dataset_posts[:split['size']]
                dataset_posts = dataset_posts[split['size']:]
            for post in tqdm.tqdm(split_posts, **settings.TQDM_KWARGS):
                instance = {
                    'id': utils.make_id(),
                    'post_id': post.id,
                    'action':
                        attr.asdict(post.action)
                        if post.action is not None else
                        None,
                    'title': post.title,
                    'text': post.original_text,
                    'post_type': post.post_type.name,
                    'label_scores': {
                        label.name: score
                        for label, score
                        in post.label_scores.label_to_score.items()
                    },
                    'label': post.label_scores.best_label.name,
                    'binarized_label_scores': {
                        binarized_label.name: score
                        for binarized_label, score
                        in post.label_scores.binarized_label_to_score.items()
                    },
                    'binarized_label': post.label_scores\
                        .best_binarized_label.name
                }

                split_file.write(json.dumps(instance) + '\n')

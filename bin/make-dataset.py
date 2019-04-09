"""Make the social norms dataset from raw reddit data.

This script takes in posts and comments from the reddit API and creates
the social norms dataset.
"""

import collections
import json
import logging
import random

import click
import tqdm

from socialnorms import settings, utils
from socialnorms.data.comment import Comment
from socialnorms.data.post import Post
from socialnorms.data.utils import instantiate_attrs_with_extra_kwargs


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
    'output_path',
    type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option(
    '--verbose',
    is_flag=True,
    help='Set the log level to DEBUG.')
def make_dataset(
        comments_path: str,
        posts_path: str,
        output_path: str,
        verbose: bool
) -> None:
    """Create the social norms dataset and write it to OUTPUT_PATH.

    Read in the reddit posts from POSTS_PATH and comments from
    COMMENTS_PATH, create the social norms dataset, and write it to
    OUTPUT_PATH.
    """
    utils.configure_logging(verbose=verbose)

    # Step 1: Read in the comments and index them by their link ids.
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
    posts = []
    with click.open_file(posts_path, 'r') as posts_file:
        for ln in tqdm.tqdm(posts_file.readlines(), **settings.TQDM_KWARGS):
            kwargs = json.loads(ln)
            post = instantiate_attrs_with_extra_kwargs(
                Post,
                comments=link_id_to_comments[kwargs['id']],
                **kwargs)

            posts.append(post)

    # Step 3: Extract labels with scores from the comments for each
    # post, filter out bad posts, try and retrieve the original post
    # text from the comments, and then write the dataset instances to
    # disk.
    with click.open_file(output_path, 'w') as output_file:
        for post in tqdm.tqdm(
                # shuffle the posts
                random.sample(posts, len(posts)),
                **settings.TQDM_KWARGS
        ):
            if not post.is_good:
                continue

            instance = {
                'id': post.id,
                'post_type': post.post_type.name,
                'title': post.title,
                'text': post.original_text or post.selftext,
                'label_scores': {
                    label.name: score
                    for label, score
                    in post.label_scores.label_to_score.items()
                },
                'label': post.label_scores.best_label.name
            }

            output_file.write(json.dumps(instance) + '\n')


if __name__ == '__main__':
    make_dataset()

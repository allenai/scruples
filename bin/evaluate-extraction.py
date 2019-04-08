"""Evaluate extraction.

This script evaluates various elements of the project's extraction
process, and writes the analysis to a report.
"""

import collections
import json
import logging
import os

import attr
import click
from sklearn import metrics
import tqdm

from socialnorms import (
    settings,
    utils)
from socialnorms.data.comment import Comment
from socialnorms.data.post import Post
from socialnorms.data.utils import instantiate_attrs_with_extra_kwargs
from socialnorms.data.labels import Label
from socialnorms.data.post_types import PostType


logger = logging.getLogger(__name__)


# constants

REPORT_TEMPLATE =\
"""Social Norms Extraction Performance Report
==========================================
Evaluation of extraction performance for socialnorms.

The annotations are taken as the ground truth, while the extracted
labels are taken as predictions.


Comments
--------
Information related to comment data.

Spam comments are filtered out and not reported on in all non-spam
results.

### Statistics

#### Spam / Ham

{comment_spam_stats}

#### Implicit / Explicit

{comment_implicit_stats}

#### Labels

**implicit**

{comment_implicit_label_stats}

**explicit**

{comment_explicit_label_stats}

**all**

{comment_all_label_stats}

### Performance Metrics

#### Spam / Ham

**classification report**
{comment_spam_classification_report}

**confusion matrix**
{comment_spam_confusion_matrix}

#### Labels

**explicit classification report**
{comment_explicit_labels_classification_report}

**explicit confusion matrix**
{comment_explicit_labels_confusion_matrix}

**all classification report**
{comment_all_labels_classification_report}

**all confusion matrix**
{comment_all_labels_confusion_matrix}


Posts
-----
Information related to post data.

Spam posts are filtered out and not reported on in all non-spam results.

### Statistics

#### Spam / Ham

{post_spam_stats}

#### Implicit / Explicit

{post_implicit_stats}

#### Post Type

**implicit**
{post_implicit_post_type_stats}

**explicit**
{post_explicit_post_type_stats}

**all**
{post_all_post_type_stats}

### Performance Metrics

#### Spam / Ham

**classification report**
{post_spam_classification_report}

**confusion matrix**
{post_spam_confusion_matrix}

#### Post Type

**explicit classification report**
{post_explicit_post_type_classification_report}

**explicit confusion matrix**
{post_explicit_post_type_confusion_matrix}

**all classification report**
{post_all_post_type_classification_report}

**all confusion matrix**
{post_all_post_type_confusion_matrix}

"""


# main function

@click.command()
@click.argument(
    'comments_path',
    type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument(
    'comment_annotations_path',
    type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument(
    'posts_path',
    type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument(
    'post_annotations_path',
    type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument(
    'output_dir',
    type=click.Path(exists=False, file_okay=False, dir_okay=True))
def evaluate_extraction(
        comments_path: str,
        comment_annotations_path: str,
        posts_path: str,
        post_annotations_path: str,
        output_dir: str
) -> None:
    """Evaluate extractions and write a report.

    Read in the comments from COMMENTS_PATH, comment annotations from
    COMMENT_ANNOTATIONS_PATH, posts from POSTS_PATH, post annotations
    from POST_ANNOTATIONS_PATH, analyze how well the system extracts
    dataset instances using these ground truth annotations, and then
    write several files to OUTPUT_DIR, including:

      - report.md: a report analyzing extraction performance.
      - comment-spam-misclassifications.jsonl: all the comments that
        were misclassified during spam filtering.
      - comment-label-misclassifications.jsonl: all the comments whose
        label was misclassified.
      - post-spam-misclassifications.jsonl: all the posts that were
        misclassified during spam filtering.
      - post-type-misclassifications.jsonl: all the posts whose post
        type was misclassified.

    COMMENTS_PATH and POSTS_PATH should point to files with comments and
    posts from the AmItheAsshole subreddit in JSON Lines format (as
    returned by the reddit API).

    COMMENT_ANNOTATIONS_PATH and POST_ANNOTATIONS_PATH should point to
    files constructed according to the Annotation Guidelines
    (annotation-guidelines.md) document.
    """
    # Step 0: Construct important paths.
    os.makedirs(output_dir)
    report_path = os.path.join(output_dir, 'report.md')
    comment_spam_misclassifications_path = os.path.join(
        output_dir, 'comment-spam-misclassifications.jsonl')
    comment_label_misclassifications_path = os.path.join(
        output_dir, 'comment-label-misclassifications.jsonl')
    post_spam_misclassifications_path = os.path.join(
        output_dir, 'post-spam-misclassifications.jsonl')
    post_type_misclassifications_path = os.path.join(
        output_dir, 'post-type-misclassifications.jsonl')

    # Step 1: Read in the comments, indexed by comment ID so we can join
    # them to the comment annotations, and indexed by link ID so we can
    # use them to construct the posts.
    comment_id_to_comment = {}
    link_id_to_comments = collections.defaultdict(list)
    with click.open_file(comments_path, 'r') as comments_file:
        for ln in tqdm.tqdm(comments_file.readlines(), **settings.TQDM_KWARGS):
            comment = instantiate_attrs_with_extra_kwargs(
                Comment,
                **json.loads(ln))

            if comment.id in comment_id_to_comment:
                raise ValueError(
                    f'Found multiple comments with id: {comment.id}.')

            comment_id_to_comment[comment.id] = comment

            # IDs are usually prefixed with something like "t1_",
            # "t2_", etc. to denote what type of object it is. Slice
            # off the first 3 characters to remove this prefix from
            # the link id because it will not be on the posts' IDs
            # when we join the comments to them.
            link_id_to_comments[comment.link_id[3:]].append(comment)

    # Step 2: Read in the comment annotations and join them to the
    # comments.
    comment_and_annotations = []
    with click.open_file(comment_annotations_path, 'r') \
         as comment_annotations_file:
        for ln in comment_annotations_file:
            comment_annotation = json.loads(ln)
            if comment_annotation['id'] not in comment_id_to_comment:
                raise ValueError(
                    f'Could not find a comment corresponding to the comment'
                    f' annotation with id: {comment_annotation["id"]}.')

            comment_and_annotations.append(
                (
                    comment_id_to_comment[comment_annotation['id']],
                    comment_annotation
                ))

    # Step 3: Read in the posts, indexed by post ID so we can join them
    # to the post annotations.
    post_id_to_post = {}
    with click.open_file(posts_path, 'r') as posts_file:
        for ln in tqdm.tqdm(posts_file.readlines(), **settings.TQDM_KWARGS):
            kwargs = json.loads(ln)
            post = instantiate_attrs_with_extra_kwargs(
                Post,
                comments=link_id_to_comments[kwargs['id']],
                **kwargs)

            if post.id in post_id_to_post:
                raise ValueError(
                    f'Found multiple posts with id: {post.id}.')

            post_id_to_post[post.id] = post

    # Step 4: Read in the post annotations and join them to the posts.
    post_and_annotations = []
    with click.open_file(post_annotations_path, 'r') \
         as post_annotations_file:
        for ln in post_annotations_file:
            post_annotation = json.loads(ln)
            if post_annotation['id'] not in post_id_to_post:
                raise ValueError(
                    f'Could not find a post corresponding to the post'
                    f' annotation with id: {post_annotation["id"]}.')

            post_and_annotations.append(
                (
                    post_id_to_post[post_annotation['id']],
                    post_annotation
                ))

    # Step 5: Write the report.
    label_names = [label.name for label in Label]
    post_types = [post_type.name for post_type in PostType]
    # compute various labels from comments
    comment_spam = [
        'spam'
        if (not comment.is_good) or comment.label is None
        else 'ham'
        for comment, _ in comment_and_annotations
    ]
    comment_annotation_spam = [
        'spam' if annotation['spam'] else 'ham'
        for _, annotation in comment_and_annotations
    ]
    comment_implicit_label = [
        getattr(comment.label, 'name', 'null')
        for comment, annotation in comment_and_annotations
        if (not annotation['spam']) and annotation['implicit']
    ]
    comment_annotation_implicit_label = [
        annotation['label'] or 'null'
        for _, annotation in comment_and_annotations
        if (not annotation['spam']) and annotation['implicit']
    ]
    comment_explicit_label = [
        getattr(comment.label, 'name', 'null')
        for comment, annotation in comment_and_annotations
        if (not annotation['spam']) and (not annotation['implicit'])
    ]
    comment_annotation_explicit_label = [
        annotation['label'] or 'null'
        for _, annotation in comment_and_annotations
        if (not annotation['spam']) and (not annotation['implicit'])
    ]
    comment_all_label = [
        getattr(comment.label, 'name', 'null')
        for comment, annotation in comment_and_annotations
        if not annotation['spam']
    ]
    comment_annotation_all_label = [
        annotation['label'] or 'null'
        for _, annotation in comment_and_annotations
        if not annotation['spam']
    ]
    # compute various labels from posts
    post_spam = [
        'spam'
        if (not post.is_good) or post.post_type is None
        else 'ham'
        for post, _ in post_and_annotations
    ]
    post_annotation_spam = [
        'spam' if annotation['spam'] else 'ham'
        for _, annotation in post_and_annotations
    ]
    post_implicit_post_type = [
        getattr(post.post_type, 'name', 'null')
        for post, annotation in post_and_annotations
        if (not annotation['spam']) and annotation['implicit']
    ]
    post_annotation_implicit_post_type = [
        annotation['post_type'] or 'null'
        for _, annotation in post_and_annotations
        if (not annotation['spam']) and annotation['implicit']
    ]
    post_explicit_post_type = [
        getattr(post.post_type, 'name', 'null')
        for post, annotation in post_and_annotations
        if (not annotation['spam']) and (not annotation['implicit'])
    ]
    post_annotation_explicit_post_type = [
        annotation['post_type'] or 'null'
        for _, annotation in post_and_annotations
        if (not annotation['spam']) and (not annotation['implicit'])
    ]
    post_all_post_type = [
        getattr(post.post_type, 'name', 'null')
        for post, annotation in post_and_annotations
        if not annotation['spam']
    ]
    post_annotation_all_post_type = [
        annotation['post_type'] or 'null'
        for _, annotation in post_and_annotations
        if not annotation['spam']
    ]

    report_kwargs = {
        # comment statistics
        'comment_spam_stats': utils.make_label_distribution_str(
            y_true=comment_annotation_spam,
            labels=['spam', 'ham']),
        'comment_implicit_stats': utils.make_label_distribution_str(
            y_true=[
                'implicit' if annotation['implicit'] else 'explicit'
                for _, annotation in comment_and_annotations
                if not annotation['spam']
            ],
            labels=['implicit', 'explicit']),
        'comment_implicit_label_stats': utils.make_label_distribution_str(
            y_true=comment_annotation_implicit_label,
            labels=label_names),
        'comment_explicit_label_stats': utils.make_label_distribution_str(
            y_true=comment_annotation_explicit_label,
            labels=label_names),
        'comment_all_label_stats': utils.make_label_distribution_str(
            y_true=comment_annotation_all_label,
            labels=label_names),
        # comment extraction performance metrics
        'comment_spam_classification_report': metrics.classification_report(
            y_true=comment_annotation_spam,
            y_pred=comment_spam,
            labels=['spam', 'ham']),
        'comment_spam_confusion_matrix': utils.make_confusion_matrix_str(
            y_true=comment_annotation_spam,
            y_pred=comment_spam),
        'comment_explicit_labels_classification_report':
            metrics.classification_report(
                y_true=comment_annotation_explicit_label,
                y_pred=comment_explicit_label,
                labels=label_names),
        'comment_explicit_labels_confusion_matrix':
            utils.make_confusion_matrix_str(
                y_true=comment_annotation_explicit_label,
                y_pred=comment_explicit_label),
        'comment_all_labels_classification_report':
            metrics.classification_report(
                y_true=comment_annotation_all_label,
                y_pred=comment_all_label,
                labels=label_names),
        'comment_all_labels_confusion_matrix':
            utils.make_confusion_matrix_str(
                y_true=comment_annotation_all_label,
                y_pred=comment_all_label),
        # post statistics
        'post_spam_stats': utils.make_label_distribution_str(
            y_true=post_annotation_spam,
            labels=['spam', 'ham']),
        'post_implicit_stats': utils.make_label_distribution_str(
            y_true=[
                'implicit' if annotation['implicit'] else 'explicit'
                for _, annotation in post_and_annotations
                if not annotation['spam']
            ],
            labels=['implicit', 'explicit']),
        'post_implicit_post_type_stats': utils.make_label_distribution_str(
            y_true=post_annotation_implicit_post_type,
            labels=post_types),
        'post_explicit_post_type_stats': utils.make_label_distribution_str(
            y_true=post_annotation_explicit_post_type,
            labels=post_types),
        'post_all_post_type_stats': utils.make_label_distribution_str(
            y_true=post_annotation_all_post_type,
            labels=post_types),
        # post extraction performance metrics
        'post_spam_classification_report': metrics.classification_report(
            y_true=post_annotation_spam,
            y_pred=post_spam,
            labels=['spam', 'ham']),
        'post_spam_confusion_matrix': utils.make_confusion_matrix_str(
            y_true=post_annotation_spam,
            y_pred=post_spam),
        'post_explicit_post_type_classification_report':
            metrics.classification_report(
                y_true=post_annotation_explicit_post_type,
                y_pred=post_explicit_post_type,
                labels=post_types),
        'post_explicit_post_type_confusion_matrix':
            utils.make_confusion_matrix_str(
                y_true=post_annotation_explicit_post_type,
                y_pred=post_explicit_post_type),
        'post_all_post_type_classification_report':
            metrics.classification_report(
                y_true=post_annotation_all_post_type,
                y_pred=post_all_post_type,
                labels=post_types),
        'post_all_post_type_confusion_matrix':
            utils.make_confusion_matrix_str(
                y_true=post_annotation_all_post_type,
                y_pred=post_all_post_type)
    }
    with click.open_file(report_path, 'w') as report_file:
        report_file.write(REPORT_TEMPLATE.format(**report_kwargs))

    # Step 6: Write the comment spam misclassifications.
    with open(comment_spam_misclassifications_path, 'w') \
         as comment_spam_misclassifications_file:
        for comment, annotation in comment_and_annotations:
            if annotation['spam'] != (
                    (not comment.is_good) or comment.label is None
            ):
                comment_spam_misclassifications_file.write(
                    json.dumps(attr.asdict(comment)) + '\n')

    # Step 7: Write the comment label misclassifications.
    with open(comment_label_misclassifications_path, 'w') \
         as comment_label_misclassifications_file:
        for comment, annotation in comment_and_annotations:
            comment_label = comment.label.name if comment.label else None
            if annotation['label'] != comment_label:
                comment_label_misclassifications_file.write(
                    json.dumps(attr.asdict(comment)) + '\n')

    # Step 8: Write the post spam misclassifications.
    with open(post_spam_misclassifications_path, 'w') \
         as post_spam_misclassifications_file:
        for post, annotation in post_and_annotations:
            if annotation['spam'] != (
                    (not post.is_good) or post.post_type is None
            ):
                post_spam_misclassifications_file.write(
                    json.dumps(attr.asdict(post)) + '\n')

    # Step 9: Write the post label misclassifications.
    with open(post_type_misclassifications_path, 'w') \
         as post_type_misclassifications_file:
        for post, annotation in post_and_annotations:
            post_type = post.post_type.name if post.post_type else None
            if annotation['post_type'] != post_type:
                post_type_misclassifications_file.write(
                    json.dumps(attr.asdict(post)) + '\n')


if __name__ == '__main__':
    evaluate_extraction()

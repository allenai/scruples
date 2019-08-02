"""Analyze the corpus statistics."""

import json
import logging
import os
from typing import (
    Any,
    Dict,
    List)

import click
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import spacy

from ... import settings


logger = logging.getLogger(__name__)


# constants

nlp = spacy.load('en', disable=['tagger', 'parser', 'ner'])


# helper functions

def _plot_token_lengths(
        split_output_dir: str,
        title_docs: List[spacy.tokens.doc.Doc],
        text_docs: List[spacy.tokens.doc.Doc],
        action_docs: List[spacy.tokens.doc.Doc]
) -> None:
    # plot the distribution of action lengths
    plt.title('Action Lengths')
    sns.distplot(
        np.array([len(doc) for doc in action_docs]),
        axlabel="# tokens",
        hist=True,
        kde=False,
        rug=False)
    plt.ylabel('# instances')

    plt.savefig(os.path.join(split_output_dir, 'action-lengths.png'))
    plt.close()

    # plot the distribution of title and text lengths
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    axes[0].set_title('titles')
    sns.distplot(
        np.array([len(doc) for doc in title_docs]),
        axlabel='# tokens',
        hist=True,
        kde=False,
        rug=False,
        ax=axes[0])
    axes[0].set_ylabel('# instances')

    axes[1].set_title('texts')
    sns.distplot(
        np.array([len(doc) for doc in text_docs]),
        axlabel='# tokens',
        hist=True,
        kde=False,
        rug=False,
        ax=axes[1])
    axes[1].set_ylabel('# instances')

    plt.savefig(os.path.join(split_output_dir, 'title-and-text-lengths.png'))
    plt.close()


def _write_statistics(
        split_output_dir: str,
        instances: List[Dict[str, Any]],
        title_docs: List[spacy.tokens.doc.Doc],
        text_docs: List[spacy.tokens.doc.Doc],
        action_docs: List[spacy.tokens.doc.Doc]
) -> None:
    title_tokenss = [
        [token.text.lower() for token in doc]
        for doc in title_docs
    ]
    text_tokenss = [
        [token.text.lower() for token in doc]
        for doc in text_docs
    ]
    action_tokenss = [
        [token.text.lower() for token in doc]
        for doc in action_docs
    ]

    # how large is the dataset in terms of...

    # ...data points collected?
    n_instances = len(instances)
    n_judgments = sum(
        judgment
        for instance in instances
        for judgment in instance['label_scores'].values())
    n_actions = len(action_docs)

    # ...tokens?
    n_title_tokens = sum(len(tokens) for tokens in title_tokenss)
    n_text_tokens = sum(len(tokens) for tokens in text_tokenss)
    n_total_tokens = n_title_tokens + n_text_tokens
    n_action_tokens = sum(len(tokens) for tokens in action_tokenss)

    # ...vocabulary?
    title_types = set(
        token
        for tokens in title_tokenss
        for token in tokens)
    text_types = set(
        token
        for tokens in text_tokenss
        for token in tokens)
    n_title_types = len(title_types)
    n_text_types = len(text_types)
    n_total_types = len(title_types.union(text_types))
    action_types = set(
        token
        for tokens in action_tokenss
        for token in tokens)
    n_action_types = len(action_types)

    # what does a typical instance look like in terms of...

    # ...number of judgments?
    instance_n_judgments = [
        sum(instance['label_scores'].values())
        for instance in instances
    ]

    # ...token length?
    instance_n_title_tokens = [
        len(tokens) for tokens in title_tokenss
    ]
    instance_n_text_tokens = [
        len(tokens) for tokens in text_tokenss
    ]
    instance_n_total_tokens = [
        len(title_tokens) + len(text_tokens)
        for title_tokens, text_tokens in zip(title_tokenss, text_tokenss)
    ]
    instance_n_action_tokens = [
        len(tokens) for tokens in action_tokenss
    ]

    # ...token-type ratio?
    instance_title_token_type_ratio = [
        len(tokens) / (len(set(tokens)) or 1)
        for tokens in title_tokenss
    ]
    instance_text_token_type_ratio = [
        len(tokens) / (len(set(tokens)) or 1)
        for tokens in text_tokenss
    ]
    instance_total_token_type_ratio = [
        (len(title_tokens) + len(text_tokens))
        / (len(set(title_tokens).union(set(text_tokens))) or 1)
        for title_tokens, text_tokens in zip(title_tokenss, text_tokenss)
    ]
    instance_action_token_type_ratio = [
        len(tokens) / (len(set(tokens)) or 1)
        for tokens in action_tokenss
    ]

    with open(os.path.join(split_output_dir, 'stats.json'), 'w') as stats_file:
        json.dump({
            'n_instances': n_instances,
            'n_judgments': n_judgments,
            'n_actions': n_actions,
            'n_title_tokens': n_title_tokens,
            'n_text_tokens': n_text_tokens,
            'n_total_tokens': n_total_tokens,
            'n_action_tokens': n_action_tokens,
            'n_title_types': n_title_types,
            'n_text_types': n_text_types,
            'n_total_types': n_total_types,
            'n_action_types': n_action_types,
            'instance_n_judgments': {
                '25th': np.quantile(
                    instance_n_judgments, 0.25),
                '50th': np.quantile(
                    instance_n_judgments, 0.50),
                '75th': np.quantile(
                    instance_n_judgments, 0.75)
            },
            'instance_n_title_tokens': {
                '25th': np.quantile(
                    instance_n_title_tokens, 0.25),
                '50th': np.quantile(
                    instance_n_title_tokens, 0.50),
                '75th': np.quantile(
                    instance_n_title_tokens, 0.75)
            },
            'instance_n_text_tokens': {
                '25th': np.quantile(
                    instance_n_text_tokens, 0.25),
                '50th': np.quantile(
                    instance_n_text_tokens, 0.50),
                '75th': np.quantile(
                    instance_n_text_tokens, 0.75)
            },
            'instance_n_total_tokens': {
                '25th': np.quantile(
                    instance_n_total_tokens, 0.25),
                '50th': np.quantile(
                    instance_n_total_tokens, 0.50),
                '75th': np.quantile(
                    instance_n_total_tokens, 0.75)
            },
            'instance_n_action_tokens': {
                '25th': np.quantile(
                    instance_n_action_tokens, 0.25),
                '50th': np.quantile(
                    instance_n_action_tokens, 0.50),
                '75th': np.quantile(
                    instance_n_action_tokens, 0.75)
            },
            'instance_title_token_type_ratio': {
                '25th': np.quantile(
                    instance_title_token_type_ratio, 0.25),
                '50th': np.quantile(
                    instance_title_token_type_ratio, 0.50),
                '75th': np.quantile(
                    instance_title_token_type_ratio, 0.75)
            },
            'instance_text_token_type_ratio': {
                '25th': np.quantile(
                    instance_text_token_type_ratio, 0.25),
                '50th': np.quantile(
                    instance_text_token_type_ratio, 0.50),
                '75th': np.quantile(
                    instance_text_token_type_ratio, 0.75)
            },
            'instance_total_token_type_ratio': {
                '25th': np.quantile(
                    instance_total_token_type_ratio, 0.25),
                '50th': np.quantile(
                    instance_total_token_type_ratio, 0.50),
                '75th': np.quantile(
                    instance_total_token_type_ratio, 0.75)
            },
            'instance_action_token_type_ratio': {
                '25th': np.quantile(
                    instance_action_token_type_ratio, 0.25),
                '50th': np.quantile(
                    instance_action_token_type_ratio, 0.50),
                '75th': np.quantile(
                    instance_action_token_type_ratio, 0.75)
            }
        }, stats_file, indent=2)


# main function

@click.command()
@click.argument(
    'corpus_dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument(
    'output_dir',
    type=click.Path(exists=False, file_okay=False, dir_okay=True))
def corpus(
        corpus_dir: str,
        output_dir: str
) -> None:
    """Write an analysis of the corpus.

    Read the corpus from CORPUS_DIR and write various aspects of an
    analysis to OUTPUT_DIR.
    """
    # store all instances so we can compute whole-dataset statistics
    all_instances = []

    # analyze each split separately
    for split in settings.SPLITS:
        split_name  = split['name']

        logger.info(f'Beginning analysis for {split_name}.')

        split_path = os.path.join(
            corpus_dir,
            settings.CORPUS_FILENAME_TEMPLATE.format(split=split_name))
        split_output_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_output_dir)

        logger.info(f'Reading {split_name} from {split_path}.')

        with open(split_path, 'r') as split_file:
            instances = [json.loads(ln) for ln in split_file]

        logger.info(
            f'Tokenizing titles, texts, and actions in {split_name}.')

        title_docs = [
            doc
            for doc
            in nlp.pipe([instance['title'] for instance in instances])
        ]
        text_docs = [
            doc
            for doc
            in nlp.pipe([instance['text'] for instance in instances])
        ]
        action_docs = [
            doc
            for doc
            in nlp.pipe([
                instance['action']['description']
                for instance in instances
                if instance['action'] is not None
            ])
        ]

        logger.info(
            f'Plotting token length distributions in {split_name}.')

        _plot_token_lengths(
            split_output_dir=split_output_dir,
            title_docs=title_docs,
            text_docs=text_docs,
            action_docs=action_docs)

        logger.info(f'Writing statistics for {split_name}.')

        _write_statistics(
            split_output_dir=split_output_dir,
            instances=instances,
            title_docs=title_docs,
            text_docs=text_docs,
            action_docs=action_docs)

        all_instances.extend(instances)

    # re-run the computation across the entire dataset
    split_output_dir = os.path.join(output_dir, 'all')
    os.makedirs(split_output_dir)

    logger.info('Tokenizing titles, texts, and actions for all splits.')

    all_title_docs = [
        doc
        for doc
        in nlp.pipe([instance['title'] for instance in all_instances])
    ]
    all_text_docs = [
        doc
        for doc
        in nlp.pipe([instance['text'] for instance in all_instances])
    ]
    all_action_docs = [
        doc
        for doc
        in nlp.pipe([
            instance['action']['description']
            for instance in all_instances
            if instance['action'] is not None
        ])
    ]

    logger.info(f'Writing whole-dataset statistics.')

    _write_statistics(
        split_output_dir=split_output_dir,
        instances=all_instances,
        title_docs=all_title_docs,
        text_docs=all_text_docs,
        action_docs=all_action_docs)

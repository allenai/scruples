"""Infer topics for the action descriptions in the resource."""

import json
import logging

import click
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline


logger = logging.getLogger(__name__)


# constants

N_COMPONENTS = 5
"""The number of topics to fit."""
# N.B. This value was chosen interactively by examining the resulting
# topics and log-likelihood the topic model produces on the resource's
# dev set for various values of n_components.


# main function

@click.command()
@click.argument(
    'split_path',
    type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument(
    'output_path',
    type=click.Path(exists=False, file_okay=True, dir_okay=False))
def topics(
        split_path: str,
        output_path: str
) -> None:
    """Create topics for the action descriptions.

    Read in the actions split from SPLIT_PATH, apply a Latent Dirichlet
    Allocation based topic model to the descriptions of all the actions,
    then write out the top words from the resulting topics to
    OUTPUT_PATH.
    """
    logger.info('Fitting the topic model to the action descriptions.')

    model = Pipeline([
        (
            'vectorizer',
            CountVectorizer(
                input='content',
                encoding='utf-8',
                decode_error='strict',
                strip_accents=None,
                lowercase=True,
                preprocessor=None,
                tokenizer=None,
                stop_words='english',
                token_pattern=r'(?u)\b\w\w+\b',
                ngram_range=(1, 1),
                analyzer='word',
                max_df=1.0,
                min_df=2,
                max_features=None,
                vocabulary=None,
                binary=False,
            )
        ),
        (
            'topic_model',
            LatentDirichletAllocation(n_components=N_COMPONENTS)
        )
    ])

    with click.open_file(split_path, 'r') as split_file:
        descriptions = [
            action['description']
            for ln in split_file
            for action in json.loads(ln)['actions']
        ]

    model.fit(descriptions)

    feature_names = model.named_steps['vectorizer'].get_feature_names()
    components = model.named_steps['topic_model'].components_

    topics_ = [
        [
            {
                'word': word,
                'alpha': alpha
            }
            for alpha, word in sorted(
                    zip(component, feature_names)
            )[-25:][::-1]
        ]
        for component in components
    ]

    logger.info("Writing out the topics' top words.")

    with click.open_file(output_path, 'w') as output_file:
        for topic in topics_:
            output_file.write(json.dumps(topic) + '\n')

    logger.info("Finished topic analysis.")

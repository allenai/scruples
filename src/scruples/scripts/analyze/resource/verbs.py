"""Analyze the root verbs used in the actions."""

import json
import logging

import click
import numpy as np
from scipy import stats
import spacy
from statsmodels.stats.multitest import multipletests
import tqdm

from .... import settings


logger = logging.getLogger(__name__)


# constants

N_SAMPLES = 100000
"""The number of samples to use for the permutation test."""


SIGNIFICANCE = 0.05
"""The level of statistical significance to use when mining verbs."""
# N.B. This value should be kept in sync with the doc string for the
# ``verbs`` script/function.


# helper functions

def _get_verb_to_counts(actions_verbs, labels, vocab):
    # Initialize the verb_to_counts dictionary.
    verb_to_counts = {
        verb: {
            'more_ethical': 0,
            'less_ethical': 0,
            'total': 0
        }
        for verb in vocab
    }
    for [verb0, verb1], label in zip(actions_verbs, labels):
        if label == 0:
            verb_to_counts[verb0]['less_ethical'] += 1
            verb_to_counts[verb1]['more_ethical'] += 1
        elif label == 1:
            verb_to_counts[verb0]['more_ethical'] += 1
            verb_to_counts[verb1]['less_ethical'] += 1
        else:
            raise ValueError(
                'Labels should only take binary values.')

        verb_to_counts[verb0]['total'] += 1
        verb_to_counts[verb1]['total'] += 1

    return verb_to_counts


def _get_verb_to_likelihood_ratio(actions_verbs, labels, vocab):
    total_n_less_ethical = len(labels)
    total_n_more_ethical = len(labels)

    verb_to_counts = _get_verb_to_counts(actions_verbs, labels, vocab)

    return {
        verb: (
            # Convert floats to numpy floats so that division by zero
            # will return infinity rather than an error. Because we only
            # care about quantiles, infinities are perfectly acceptable.
            np.float64(counts['less_ethical'] / total_n_less_ethical)
            / np.float64(counts['more_ethical'] / total_n_more_ethical)
        ).item()
        for verb, counts
        in verb_to_counts.items()
    }


def _get_verb_to_permutation_likelihood_ratios(
        actions_verbs,
        labels,
        vocab,
        n_samples
):
    verb_to_permutation_likelihood_ratios = {
        verb: []
        for verb in vocab
    }
    for _ in tqdm.tqdm(range(n_samples), **settings.TQDM_KWARGS):
        for verb, likelihood_ratio in _get_verb_to_likelihood_ratio(
            actions_verbs,
            np.random.permutation(labels),
            vocab
        ).items():
            verb_to_permutation_likelihood_ratios[verb].append(
                likelihood_ratio)

    return verb_to_permutation_likelihood_ratios


def _get_verb_analyses(actions_verbs, labels, significance, n_samples):
    # Compute the vocabulary.
    vocab = {
        verb
        for action_verbs in actions_verbs
        for verb in action_verbs
    }

    # Create the verb analyses with unadjusted significances.
    verb_to_counts = _get_verb_to_counts(actions_verbs, labels, vocab)
    verb_to_likelihood_ratio = _get_verb_to_likelihood_ratio(
        actions_verbs, labels, vocab)
    verb_to_permutation_likelihood_ratios =\
        _get_verb_to_permutation_likelihood_ratios(
            actions_verbs,
            labels,
            vocab,
            n_samples=n_samples)

    verb_analyses = [
        {
            'verb': verb,
            'counts': verb_to_counts[verb],
            'likelihood_ratio': verb_to_likelihood_ratio[verb],
            # Find the percentile of the absolute value of
            # log-likelihood ratio according to the data in the
            # permutation distribution of log-likelihood ratios.  This
            # assigns significance to the log-likelihood ratios using a
            # two-tailed permutation test against the alternative or no
            # association between the verbs and the gold label.
            'unadjusted_significance': 1. - (
                stats.percentileofscore(
                    np.abs(
                        np.log(verb_to_permutation_likelihood_ratios[verb])
                    ),
                    np.abs(
                        np.log(verb_to_likelihood_ratio[verb])
                    )
                ) / 100.
            )
        }
        for verb in vocab
    ]

    # Add corrected p-values using the Holm-Bonferroni method, to adjust
    # for multiple tests.
    unadjusted_significances = [
        verb_analysis['unadjusted_significance']
        for verb_analysis in verb_analyses
    ]

    significants, adjusted_significances, _, _ = multipletests(
        unadjusted_significances,
        alpha=significance,
        method='holm',
    )

    # Add the adjusted significance to each verb analysis.
    for significant, adjusted_significance, verb_analysis in zip(
            significants,
            adjusted_significances,
            verb_analyses
    ):
        verb_analysis['significant'] = significant.item()
        verb_analysis['adjusted_significance'] = adjusted_significance.item()

    return sorted(
        verb_analyses,
        key=lambda x: x['unadjusted_significance']
    )


# main function

@click.command()
@click.argument(
    'split_path',
    type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument(
    'output_path',
    type=click.Path(exists=False, file_okay=True, dir_okay=False))
def verbs(
        split_path: str,
        output_path: str
) -> None:
    """Analyze the main verbs used in the actions.

    Read in the split from SPLIT_PATH, then extract the root verb from
    each action. Analyze the verbs, then write the results to
    OUTPUT_PATH.

    The analysis counts how many times each verb appears in the more and
    less ethical action, computes their likelihood ratios between the
    classes, i.e. P(verb|less ethical) / P(verb|more ethical), uses a
    permutation test to assign significance scores to these likelihood
    ratios, and finally corrects the significances for multiple testing
    using the Holm-Bonferroni method, reporting whether each association
    is significant at the 0.05 level.
    """
    logger.info('Extracting verbs from the actions.')

    nlp = spacy.load('en_core_web_sm')

    actions = []
    with click.open_file(split_path, 'r') as split_file:
        actions = [json.loads(ln) for ln in split_file]

    actions_verbs = [
        [
            # Take the first sentence's root verb, then normalize it.
            next(doc0.sents).root.text.lower().strip(),
            next(doc1.sents).root.text.lower().strip()
        ]
        for doc0, doc1 in tqdm.tqdm(
                zip(
                    nlp.pipe(
                        action['actions'][0]['description']
                        for action in actions
                    ),
                    nlp.pipe(
                        action['actions'][1]['description']
                        for action in actions
                    ),
                ),
                total=len(actions),
                **settings.TQDM_KWARGS
        )
    ]

    labels = [action['gold_label'] for action in actions]

    logger.info('Analyzing verbs.')

    verb_analyses = _get_verb_analyses(
        actions_verbs,
        labels,
        significance=SIGNIFICANCE,
        n_samples=N_SAMPLES,
    )

    logger.info('Writing results.')

    with click.open_file(output_path, 'w') as output_file:
        for verb_analysis in verb_analyses:
            output_file.write(json.dumps(verb_analysis) + '\n')

    logger.info('Finished the verb analysis.')

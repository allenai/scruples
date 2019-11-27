"""Simulation experiments for the oracle performance estimator."""

import collections
import json
import logging

import click
import numpy as np
from scipy import stats
from sklearn import metrics
import tqdm

from scruples import settings, utils
from scruples.data.labels import Label
from scruples.dataset.readers import ScruplesCorpus


logger = logging.getLogger(__name__)


METRICS = [
    (
        'accuracy',
        metrics.accuracy_score,
        lambda pss: np.argmax(pss, axis=-1)
    ),
    (
        'f1 (macro)',
        lambda y_pred, y_true: metrics.f1_score(
            y_pred=y_pred, y_true=y_true,
            average='macro'),
        lambda pss: np.argmax(pss, axis=-1)
    ),
    (
        'xentropy',
        utils.xentropy,
        lambda pss: pss
    )
]
"""The metrics to run in the experiment.

Each tuple contains:

    (name, metric, make_predictions)

"""


@click.command()
@click.argument(
    'corpus_dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument(
    'output_path',
    type=click.Path(exists=False, file_okay=True, dir_okay=False))
def oracle_estimator(
        corpus_dir: str,
        output_path: str
) -> None:
    """Conduct simulation experiments for the oracle estimator.

    Read the corpus from CORPUS_DIR, conduct different simulation experiments
    to evaluate the robustness and effectiveness of the oracle performance
    estimator, and write the results to OUTPUT_PATH.
    """
    # Read in the corpus.

    logger.info(f'Reading data from {corpus_dir}.')

    _, _, _, label_scores = ScruplesCorpus(data_dir=corpus_dir).dev
    label_scores = label_scores.values

    n_total = len(label_scores)
    alphas = utils.estimate_dirichlet_multinomial_parameters(label_scores)

    # Create the scenarios to simulate.

    logger.info('Creating simulation scenarios.')

    scenarios = []
    # scenario 1: running the estimator on the corpus
    ns = np.sum(label_scores, axis=-1)
    pss = stats.dirichlet.rvs(alpha=alphas, size=n_total)
    ys = np.array([stats.multinomial.rvs(n, ps) for n, ps in zip(ns, pss)])
    scenarios.append({
        'name': 'corpus',
        'ns': ns,
        'pss': pss,
        'ys': ys
    })
    # scenario 2: running the estimator with 3 annotations per example
    ns = 3 * np.ones(n_total)
    pss = stats.dirichlet.rvs(alpha=alphas, size=n_total)
    ys = np.array([stats.multinomial.rvs(n, ps) for n, ps in zip(ns, pss)])
    scenarios.append({
        'name': '3 annotations',
        'ns': ns,
        'pss': pss,
        'ys': ys
    })
    # scenario 3: running the estimator with a non-dirichlet prior
    ns = np.sum(label_scores, axis=-1)
    pss = np.concatenate([
        stats.dirichlet.rvs(alpha=[2, 1, 1, 1, 1], size=n_total//3),
        stats.dirichlet.rvs(alpha=[1, 1, 3, 1, 1], size=n_total//3),
        stats.dirichlet.rvs(alpha=[1, 1, 1, 1, 2], size=n_total - 2 * n_total//3)
    ])
    np.random.shuffle(pss)
    ys = np.array([stats.multinomial.rvs(n, ps) for n, ps in zip(ns, pss)])
    scenarios.append({
        'name': 'non-dirichlet prior',
        'ns': ns,
        'pss': pss,
        'ys': ys
    })

    # Run the simulations.

    logger.info('Running simulations.')

    results = collections.defaultdict(dict)
    for scenario in tqdm.tqdm(scenarios, **settings.TQDM_KWARGS):
        name = scenario['name']
        ns = scenario['ns']
        pss = scenario['pss']
        ys = scenario['ys']

        results[name]['oracle'] = {
            metric_name: metric(
                y_pred=make_predictions(pss),
                y_true=make_predictions(
                    ys / np.expand_dims(np.sum(ys, axis=-1), axis=-1)))
            for metric_name, metric, make_predictions in METRICS
        }

        # estimate oracle performance
        estimated_alphas = utils.estimate_dirichlet_multinomial_parameters(ys)

        results[name]['estimate'] = {
            metric_name: utils.oracle_performance(
                ys=ys,
                metric=metric,
                make_predictions=make_predictions,
                n_samples=10000)[0]
            for metric_name, metric, make_predictions in tqdm.tqdm(
                    METRICS, **settings.TQDM_KWARGS)
        }

    # Write the results to disk.

    logger.info(f'Writing results to {output_path}.')

    with click.open_file(output_path, 'w') as output_file:
        json.dump(results, output_file)

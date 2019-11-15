"""Estimate the oracle performance on the scruples resource."""

import json
import logging

import click
import numpy as np
from scipy import stats
import tqdm

from .... import utils, settings
from ....baselines.metrics import METRICS


logger = logging.getLogger(__name__)


# constants

REPORT_TEMPLATE =\
"""Scruples Resource Oracle Performance
====================================
Oracle performance on the scruples resource.


Oracle Metrics
--------------
The following metrics provide estimates of the oracle performance, i.e. the
performance in each metric attained when you use the _true_ underlying label
distribution for that instance to predict the labels.

We obtain these estimates using an empirical Bayesian methodology: first, we
fit a dirichlet-multinomial model to the data, where each instance's label
distribution is drawn from a dirichlet prior. Then, for each instance we
condition on the observed labels and compute the expected performance of
predicting with the true distribution over the posterior.

{metrics_report}
"""


# main function

@click.command()
@click.argument(
    'dataset_path',
    type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument(
    'output_path',
    type=click.Path(exists=False, file_okay=True, dir_okay=False))
def oracle_performance(
        dataset_path: str,
        output_path: str
) -> None:
    """Estimate oracle performance and write a report.

    Read in the dataset from DATASET_PATH, estimate the oracle
    performance and write the results to OUTPUT_PATH.
    """
    # Step 1: Read in the dataset.
    with click.open_file(dataset_path, 'r') as dataset_file:
        labels = []
        label_scores = []
        label_annotations_all = []
        for ln in dataset_file:
            row = json.loads(ln)

            labels.append(row['gold_label'])
            label_scores.append(row['gold_annotations'])
            label_annotations_all.append(
                [
                    i + j
                    for i, j in zip(
                            row['gold_annotations'],
                            row['human_perf_annotations']
                    )
                ])

    labels = np.array(labels)
    label_scores = np.array(label_scores)
    label_annotations_all = np.array(label_annotations_all)

    # Step 2: Estimate the dirichlet-multinomial parameters.
    params = utils.estimate_dirichlet_multinomial_parameters(
        label_annotations_all)

    # Step 3: Estimate the expected performance.
    metric_name_to_value = {}
    for name, metric, scorer_kwargs in METRICS.values():
        logger.info(f'Computing estimate for {name}.')
        # estimate performance on the usual metrics
        value_samples = []
        for _ in tqdm.tqdm(range(10000), **settings.TQDM_KWARGS):
            true_scores = np.array([
                stats.dirichlet.rvs([
                    a + x
                    for a, x in zip(params, scores)
                ], size=1)[0]
                for scores in label_scores
            ])
            value = metric(
                y_true=labels,
                y_pred=true_scores
                  if scorer_kwargs['needs_proba']
                  else np.argmax(true_scores, axis=1))
            value_samples.append(value)
        metric_name_to_value[name] = np.mean(value_samples)
    if 'xentropy' in metric_name_to_value:
        raise ValueError(
            'METRICS should not have a key named "xentropy". This issue'
            ' is a bug in the library, please notify the maintainers.')

    # estimate performance on the xentropy, which requires soft
    # ground-truth labels
    logger.info('Computing estimate for xentropy.')
    value_samples = []
    for _ in tqdm.tqdm(range(10000), **settings.TQDM_KWARGS):
        true_scores = np.array([
            stats.dirichlet.rvs([
                a + x
                for a, x in zip(params, scores)
            ], size=1)[0]
            for scores in label_scores
        ])
        value = utils.xentropy(
            y_true=label_scores / np.sum(label_scores, axis=1).reshape(-1, 1),
            y_pred=true_scores)
        value_samples.append(value)
    metric_name_to_value['xentropy'] = np.mean(value_samples)

    # Step 4: Write the report.
    metric_name_width = 1 + max(
        len(name)
        for name in metric_name_to_value.keys())
    metrics_report = '\n'.join(
        f'{name: <{metric_name_width}}: {value:.4f}'
        for name, value in metric_name_to_value.items())
    with click.open_file(output_path, 'w') as output_file:
        output_file.write(
            REPORT_TEMPLATE.format(
                metrics_report=metrics_report))

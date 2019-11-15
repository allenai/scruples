"""Estimate human performance for the scruples resource."""

import json
import logging

import click

from ....baselines.metrics import METRICS


logger = logging.getLogger(__name__)


# main function

@click.command()
@click.argument(
    'split_path',
    type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument(
    'output_path',
    type=click.Path(exists=False, file_okay=True, dir_okay=False))
def human_performance(
        split_path: str,
        output_path: str
) -> None:
    """Estimate human performance on the scruples resource.

    Read in the split from SPLIT_PATH, then estimate human performance
    metrics and write them to OUTPUT_PATH.

    Human performance is computed by comparing the majority vote label
    of the human performance annotators to the majority vote label of
    the gold annotators.
    """
    logger.info('Computing human performance.')

    human_preds = []
    gold_labels = []
    with click.open_file(split_path, 'r') as split_file:
        for ln in split_file:
            row = json.loads(ln)
            human_preds.append(row['human_perf_label'])
            gold_labels.append(row['gold_label'])

    with open(output_path, 'w') as metrics_file:
        json.dump({
            key: metric(
                y_true=gold_labels,
                y_pred=human_preds)
            for key, (_, metric, scorer_kwargs) in METRICS.items()
            if not scorer_kwargs['needs_proba']
        }, metrics_file)

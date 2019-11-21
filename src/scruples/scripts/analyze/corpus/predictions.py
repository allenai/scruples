"""Analyze predictions on scruples."""

import json
import logging

import click
import numpy as np
from scipy.special import softmax
from sklearn import metrics

from .... import utils
from ....data.labels import Label
from ....baselines.metrics import METRICS


logger = logging.getLogger(__name__)


# constants

REPORT_TEMPLATE =\
"""Scruples Corpus Predictions Performance Report
==============================================
Analysis of predictions on the scruples corpus.


Main Metrics
------------
Note that the xentropy score, if present, is computed with respect to
the estimated true label distribution rather than the hard labels. All
other scores are standard and computed against the most frequent label.

{metrics_report}{calibration_factor_report}


Classification Report
---------------------
{classification_report}


Confusion Matrix
----------------
{confusion_matrix}

"""


# main function

@click.command()
@click.argument(
    'dataset_path',
    type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument(
    'predictions_path',
    type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument(
    'output_path',
    type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option(
    '--label-scores', is_flag=True,
    help='Compute metrics which require predictions of label'
         ' probabilities, and include them in the report. Predictions'
         ' must have "label_scores" keys to use this option.')
@click.option(
    '--calibration-factor', type=float, default=None,
    help='The calibration factor to use for computing the calibrated'
         ' xentropy. If no calibration factor is provided, then it will be'
         ' calculated from the data.')
def predictions(
        dataset_path: str,
        predictions_path: str,
        output_path: str,
        label_scores: bool,
        calibration_factor: float
) -> None:
    """Analyze classification performance and write a report.

    Read in the dataset from DATASET_PATH, as well as predictions from
    PREDICTIONS_PATH, then analyze the predictions and write the
    results to OUTPUT_PATH. PREDICTIONS_PATH should be a JSON Lines file
    in which each object has "id", "label", and optionally
    "label_scores" keys, corresponding to the ID for the instance, the
    predicted label, and the predicted probabilities for each class.
    """
    # Step 1: Read in the dataset.
    with click.open_file(dataset_path, 'r') as dataset_file:
        id_to_dataset_label_and_label_scores = {}
        for ln in dataset_file:
            row = json.loads(ln)
            id_to_dataset_label_and_label_scores[row['id']] = (
                row['label'],
                row['label_scores']
            )

    # Step 2: Read in the predictions.
    with click.open_file(predictions_path, 'r') as predictions_file:
        id_to_predicted_label_and_label_scores = {}
        for ln in predictions_file:
            row = json.loads(ln)
            id_to_predicted_label_and_label_scores[row['id']] = (
                row['label'],
                row.get('label_scores')
            )

    # Step 3: Extract the dataset and predictions on the relevant
    # subset.
    dataset_labels_and_label_scores, predicted_labels_and_label_scores = (
        *zip(*[
            (
                id_to_dataset_label_and_label_scores[id_],
                id_to_predicted_label_and_label_scores[id_]
            )
            for id_ in id_to_predicted_label_and_label_scores.keys()
            if id_ in id_to_dataset_label_and_label_scores
        ]),
    )
    dataset_labels = [
        label
        for label, _ in dataset_labels_and_label_scores
    ]
    predicted_labels = [
        label
        for label, _ in predicted_labels_and_label_scores
    ]
    if label_scores:
        dataset_label_scores = [
            [count / sum(scores.values()) for count in scores.values()]
            for _, scores in dataset_labels_and_label_scores
        ]
        predicted_label_scores = [
            [count / sum(scores.values()) for count in scores.values()]
            for _, scores in predicted_labels_and_label_scores
        ]

    # Step 4: Write the report.
    with click.open_file(output_path, 'w') as output_file:
        # create the metrics report
        metric_name_to_value = {
            name:
                metric(
                    y_true=dataset_labels,
                    y_pred=predicted_label_scores
                      if scorer_kwargs['needs_proba']
                      else predicted_labels)
            for name, metric, scorer_kwargs in METRICS.values()
            if label_scores or not scorer_kwargs['needs_proba']
        }
        if label_scores:
            if 'xentropy' in metric_name_to_value:
                raise ValueError(
                    'METRICS should not have a key named'
                    ' "xentropy". This issue is a bug in the library,'
                    ' please notify the maintainers.')

            metric_name_to_value['xentropy'] = utils.xentropy(
                y_true=dataset_label_scores,
                y_pred=predicted_label_scores)

            if 'calibrated_xentropy' in metric_name_to_value:
                raise ValueError(
                    'METRICS should not have a key named'
                    ' "calibrated_xentropy". This issue is a bug in the'
                    ' library, please notify the maintainers.')

            logits = np.log(predicted_label_scores)
            temperature = (
                calibration_factor
                if calibration_factor is not None else
                utils.calibration_factor(
                    logits=logits,
                    targets=dataset_label_scores)
            )

            logger.info(f'Calibrating temperature: {temperature}')

            metric_name_to_value['calibrated_xentropy'] = utils.xentropy(
                y_true=dataset_label_scores,
                y_pred=softmax(logits / temperature, axis=-1))

        metric_name_width = 1 + max(
            len(name)
            for name in metric_name_to_value.keys())
        metrics_report = '\n'.join(
            f'{name: <{metric_name_width}}: {value:.4f}'
            for name, value in metric_name_to_value.items())

        if label_scores:
            calibration_factor_report = (
                f'\n\nCalibration Factor: {temperature}\n'
            )
        else:
            calibration_factor_report = ''

        # create the classification report
        label_names = [label.name for label in Label]

        classification_report = metrics.classification_report(
            y_true=dataset_labels,
            y_pred=predicted_labels,
            labels=label_names)

        # create the confusion matrix
        confusion_matrix = utils.make_confusion_matrix_str(
            y_true=dataset_labels,
            y_pred=predicted_labels,
            labels=label_names)

        output_file.write(
            REPORT_TEMPLATE.format(
                metrics_report=metrics_report,
                calibration_factor_report=calibration_factor_report,
                classification_report=classification_report,
                confusion_matrix=confusion_matrix))

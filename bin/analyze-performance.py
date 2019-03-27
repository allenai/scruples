"""Analyze classification performance on socialnorms."""

import json
import logging

import click
from sklearn import metrics

from socialnorms import utils
from socialnorms.labels import Label


logger = logging.getLogger(__name__)


# constants

REPORT_TEMPLATE =\
"""Social Norms Classification Performance Report
==============================================
Analysis of classification performance on socialnorms.


Main Metrics
------------
Accuracy          : {accuracy:.4f}
Balanced Accuracy : {balanced_accuracy:.4f}
F1 (micro)        : {f1_micro:.4f}
F1 (macro)        : {f1_macro:.4f}
F1 (weighted)     : {f1_weighted:.4f}


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
def analyze_performance(
        dataset_path: str,
        predictions_path: str,
        output_path: str
) -> None:
    """Analyze classification performance and write a report.

    Read in the dataset from DATASET_PATH, as well as predictions from
    PREDICTIONS_PATH, then analyze the predictions and write the
    results to OUTPUT_PATH.
    """
    # Step 1: Read in the dataset.
    with click.open_file(dataset_path, 'r') as dataset_file:
        id_to_dataset_label = {}
        for ln in dataset_file:
            row = json.loads(ln)
            id_to_dataset_label[row['id']] = row['label']

    # Step 2: Read in the predictions.
    with click.open_file(predictions_path, 'r') as predictions_file:
        id_to_pred = {}
        for ln in predictions_file:
            row = json.loads(ln)
            id_to_pred[row['id']] = row['pred']

    # Step 3: Extract the dataset and predictions on the relevant
    # subset.
    dataset_labels, predicted_labels = (
        *zip(*[
            (id_to_dataset_label[id_], id_to_pred[id_])
            for id_ in id_to_pred.keys()
            if id_ in id_to_dataset_label
        ]),
    )

    # Step 4: Write the report.
    with click.open_file(output_path, 'w') as output_file:
        # compute main metrics
        accuracy = metrics.accuracy_score(
            y_true=dataset_labels,
            y_pred=predicted_labels)
        balanced_accuracy = metrics.balanced_accuracy_score(
            y_true=dataset_labels,
            y_pred=predicted_labels)
        f1_micro = metrics.f1_score(
            y_true=dataset_labels,
            y_pred=predicted_labels,
            average='micro')
        f1_macro = metrics.f1_score(
            y_true=dataset_labels,
            y_pred=predicted_labels,
            average='macro')
        f1_weighted = metrics.f1_score(
            y_true=dataset_labels,
            y_pred=predicted_labels,
            average='weighted')

        # create the classification report
        classification_report = metrics.classification_report(
            y_true=dataset_labels,
            y_pred=predicted_labels)

        # create the confusion matrix
        confusion_matrix = utils.make_confusion_matrix_str(
            y_true=dataset_labels,
            y_pred=predicted_labels,
            labels=[label.name for label in Label])

        output_file.write(
            REPORT_TEMPLATE.format(
                accuracy=accuracy,
                balanced_accuracy=balanced_accuracy,
                f1_micro=f1_micro,
                f1_macro=f1_macro,
                f1_weighted=f1_weighted,
                classification_report=classification_report,
                confusion_matrix=confusion_matrix))


if __name__ == '__main__':
    analyze_performance()

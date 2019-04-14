"""Analyze classification performance on socialnorms."""

import json
import logging

import click
from sklearn import metrics

from socialnorms import utils
from socialnorms.data.labels import Label


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
Cross-Entropy     : {xentropy:.4f}


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
    '--compute-xentropy', is_flag=True,
    help='Compute the cross-entropy based on label scores and include'
         ' it in the report. Predictions must have "label_scores" keys'
         ' to use this option.')
@click.option(
    '--verbose', is_flag=True,
    help='Set the log level to DEBUG.')
def analyze_performance(
        dataset_path: str,
        predictions_path: str,
        output_path: str,
        compute_xentropy: bool,
        verbose: bool
) -> None:
    """Analyze classification performance and write a report.

    Read in the dataset from DATASET_PATH, as well as predictions from
    PREDICTIONS_PATH, then analyze the predictions and write the
    results to OUTPUT_PATH. PREDICTIONS_PATH should be a JSON Lines file
    in which each object has "id", "label", and optionally
    "label_scores" keys, corresponding to the ID for the instance, the
    predicted label, and the predicted probabilities for each class.
    """
    utils.configure_logging(verbose=verbose)

    # Step 1: Read in the dataset.
    with click.open_file(dataset_path, 'r') as dataset_file:
        id_to_dataset_label_and_label_scores = {}
        for ln in dataset_file:
            row = json.loads(ln)
            id_to_dataset_label_and_label_scores[row['id']] = (
                row['label'],
                row.get('label_scores')
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
    if compute_xentropy:
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
        if compute_xentropy:
            xentropy = utils.xentropy(
                y_true=dataset_label_scores,
                y_pred=predicted_label_scores)
        else:
            xentropy = float('nan')

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
                accuracy=accuracy,
                balanced_accuracy=balanced_accuracy,
                f1_micro=f1_micro,
                f1_macro=f1_macro,
                f1_weighted=f1_weighted,
                xentropy=xentropy,
                classification_report=classification_report,
                confusion_matrix=confusion_matrix))


if __name__ == '__main__':
    analyze_performance()

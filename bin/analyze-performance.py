"""Analyze classification performance on socialnorms."""

import json
import logging

import click
from sklearn import metrics

from socialnorms.labels import Label


logger = logging.getLogger(__name__)


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
        ]),
    )

    # Step 4: Write the report.
    with click.open_file(output_path, 'w') as output_file:
        # write the header
        output_file.write(
            f'Social Norms Performance Report\n'
            f'===============================\n'
            f'Analysis of classification performance on socialnorms.\n')
        output_file.write('\n\n')

        # write the main metrics
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

        output_file.write(
            f'Main Metrics\n'
            f'------------\n'
            f'Accuracy          : {accuracy:.4f}\n'
            f'Balanced Accuracy : {balanced_accuracy:.4f}\n'
            f'F1 (micro)        : {f1_micro:.4f}\n'
            f'F1 (macro)        : {f1_macro:.4f}\n'
            f'F1 (weighted)     : {f1_weighted:.4f}\n')
        output_file.write('\n\n')

        # write a full classification report
        output_file.write(
            'Classification Report\n'
            '---------------------\n')
        output_file.write(
            metrics.classification_report(
                y_true=dataset_labels,
                y_pred=predicted_labels))
        output_file.write('\n\n')

        # write a confusion matrix
        label_names = [label.name for label in Label]
        confusion_matrix = metrics.confusion_matrix(
            y_true=dataset_labels,
            y_pred=predicted_labels,
            labels=label_names)
        confusion_matrix_header = '|'.join(
            f' {label_name: <4} '
            for label_name in label_names)
        confusion_matrix_body = '|\n|'.join(
            f' {label_name: <4} |' + '|'.join(f' {x: >4} ' for x in row)
            for label_name, row in zip(label_names, confusion_matrix))
        output_file.write(
            f'Confusion Matrix\n'
            f'----------------\n'
            f'+======+======+======+======+======+======+\n'
            f'|      |{confusion_matrix_header}| predicted\n'
            f'+======+======+======+======+======+======+\n'
            f'|{confusion_matrix_body}|\n'
            f'+------+------+------+------+------+------+\n'
            f' true\n')
        output_file.write('\n\n')


if __name__ == '__main__':
    analyze_performance()

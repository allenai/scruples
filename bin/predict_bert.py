"""Predict labels for socialnorms using a fine-tuned BERT model."""

import json
import math
import logging
import os
from typing import List

import click
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer
from scipy.special import softmax
import torch
from torch.utils.data import DataLoader
import tqdm

from socialnorms import settings, utils
from socialnorms.data.labels import Label
from socialnorms.dataset.transforms import (
    BertTransform,
    Compose)
from socialnorms.dataset.readers import SocialNormsDataset


logger = logging.getLogger(__name__)


# main function

@click.command()
@click.argument(
    'CACHE_DIR',
    type=click.Path(file_okay=False, dir_okay=True))
@click.argument(
    'DATA_DIR',
    type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument(
    'RESULTS_DIR',
    type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument(
    'OUTPUT_DIR',
    type=click.Path(exists=False, file_okay=False, dir_okay=True))
@click.argument(
    'SPLITS', type=click.Choice(SocialNormsDataset.SPLITS), nargs=-1)
@click.option(
    '--predict-batch-size', type=int, default=64,
    help='The batch size for prediction.')
@click.option(
    '--gpu-ids', type=str, default='',
    help='The GPU IDs to use for training as a comma-separated list.')
@click.option(
    '--verbose', is_flag=True,
    help='Set the log level to DEBUG.')
def predict_bert(
        cache_dir: str,
        data_dir: str,
        results_dir: str,
        output_dir: str,
        splits: List[str],
        predict_batch_size: int,
        gpu_ids: str,
        verbose: bool
) -> None:
    """Predict using BERT on socialnorms.

    Read the socialnorms dataset from DATA_DIR, load a fine-tuned BERT
    model from RESULTS_DIR, and write the metrics and predictions to
    OUTPUT_DIR, for each split provided as an argument.
    """
    # configure logging

    utils.configure_logging(verbose=verbose)


    # manage paths

    os.makedirs(output_dir)

    config_file_path = os.path.join(results_dir, 'config.json')
    checkpoint_file_path = os.path.join(
        results_dir, 'checkpoints', 'best.checkpoint.pkl')


    # read in the fine-tuning arguments

    with open(config_file_path, 'r') as config_file:
        config = json.load(config_file)

    pretrained_bert = config['pretrained_bert']
    max_sequence_length = config['max_sequence_length']
    truncation_strategy_title = config['truncation_strategy_title']
    truncation_strategy_text = config['truncation_strategy_text']


    # configure GPUs

    if gpu_ids:
        gpu_ids = [int(gpu_id) for gpu_id in gpu_ids.split(',')]

        logger.info(
            f'Configuring environment to use {len(gpu_ids)} GPUs:'
            f' {", ".join(str(gpu_id) for gpu_id in gpu_ids)}.')

        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))

        if not torch.cuda.is_available():
            raise EnvironmentError('CUDA must be available to use GPUs.')

        device = torch.device('cuda')
    else:
        logger.info('Configuring environment to use CPU.')

        device = torch.device('cpu')


    # create the model and loss

    model = torch.nn.DataParallel(
        BertForSequenceClassification.from_pretrained(
            pretrained_bert,
            cache_dir=cache_dir,
            num_labels=len(Label)))
    model.load_state_dict(torch.load(checkpoint_file_path)['model'])

    model.to(device)

    loss = torch.nn.CrossEntropyLoss()


    # create transformations for the dataset

    featurize = Compose([
        BertTransform(
            tokenizer=BertTokenizer.from_pretrained(
                pretrained_bert,
                do_lower_case=pretrained_bert.endswith('-uncased')),
            max_sequence_length=max_sequence_length,
            truncation_strategy=(
                truncation_strategy_title,
                truncation_strategy_text
            )),
        lambda d: {k: torch.tensor(v) for k, v in d.items()}
    ])
    labelize = lambda s: getattr(Label, s).index


    for split in splits:
        logger.info(f'Loading the dataset from {data_dir}.')

        dataset = SocialNormsDataset(
            data_dir=data_dir,
            split=split,
            transform=featurize,
            label_transform=labelize)

        data_loader = DataLoader(
            dataset=dataset,
            batch_size=predict_batch_size,
            shuffle=False)


        # run predictions

        n_instances = len(dataset)
        n_batches = math.ceil(n_instances / predict_batch_size)
        with torch.no_grad():
            # set the model to evaluation mode
            model.eval()

            ids_logits_and_predictions = []
            total_loss = 0
            total_accuracy = 0
            for i, (ids, features, labels) in tqdm.tqdm(
                    enumerate(data_loader),
                    total=n_batches,
                    **settings.TQDM_KWARGS):
                # move the data onto the device
                input_ids = features['input_ids'].to(device)
                input_mask = features['input_mask'].to(device)
                segment_ids = features['segment_ids'].to(device)

                labels = labels.to(device)

                # make predictions
                logits = model(
                    input_ids=input_ids,
                    attention_mask=input_mask,
                    token_type_ids=segment_ids)
                _, predictions = torch.max(logits, 1)

                batch_loss = loss(logits, labels)
                batch_accuracy = (predictions == labels).float().mean()

                # update statistics
                total_loss = (
                    batch_loss.item() + i * total_loss
                ) / (i + 1)
                total_accuracy = (
                    batch_accuracy.item() + i * total_accuracy
                ) / (i + 1)
                ids_logits_and_predictions.extend(
                    zip(ids, logits.tolist(), predictions.tolist()))

        logger.info(
            f'\n\n'
            f'  {split} results:\n'
            f'    loss     : {total_loss:.4f}\n'
            f'    accuracy : {total_accuracy:.4f}\n')

        metrics_path = os.path.join(
            output_dir, f'{split}-metrics.json')
        with open(metrics_path, 'w') as metrics_file:
            json.dump(
                {'loss': total_loss, 'accuracy': total_accuracy},
                metrics_file)

        labels = [label.name for label in Label]
        index_to_label = {label.index: label.name for label in Label}
        predictions_path = os.path.join(
            output_dir, f'{split}-predictions.jsonl')
        with open(predictions_path, 'w') as predictions_file:
            for id_, logits, prediction in ids_logits_and_predictions:
                predictions_file.write(
                    json.dumps({
                        'id': id_,
                        'label': index_to_label[prediction],
                        'label_scores': {
                            label: score
                            for label, score
                            in zip(labels, softmax(logits))
                        }
                    }) + '\n')


if __name__ == '__main__':
    predict_bert()

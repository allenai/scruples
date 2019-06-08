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
from socialnorms.baselines.metrics import METRICS
from socialnorms.data.labels import Label
from socialnorms.dataset.transforms import (
    BertTransform,
    Compose)
from socialnorms.dataset.readers import SocialnormsCorpusDataset


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
    'SPLITS', type=click.Choice(SocialnormsCorpusDataset.SPLITS), nargs=-1)
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
    metrics_paths = {}
    predictions_paths = {}
    for split in splits:
        os.makedirs(os.path.join(output_dir, split))
        metrics_paths[split] = os.path.join(
            output_dir, split, 'metrics.json')
        predictions_paths[split] = os.path.join(
            output_dir, split, 'predictions.jsonl')

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


    # load the model

    model = torch.nn.DataParallel(
        BertForSequenceClassification.from_pretrained(
            pretrained_bert,
            cache_dir=cache_dir,
            num_labels=len(Label)))
    model.load_state_dict(torch.load(checkpoint_file_path)['model'])

    model.to(device)


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
    delabelize = lambda idx: [l.name for l in Label if l.index == idx][0]

    for split in splits:
        logger.info(f'Loading the dataset from {data_dir}.')

        dataset = SocialnormsCorpusDataset(
            data_dir=data_dir,
            split=split,
            transform=featurize,
            label_transform=labelize)

        data_loader = DataLoader(
            dataset=dataset,
            batch_size=predict_batch_size,
            shuffle=False)


        # run predictions

        ids = []
        predictions = []
        probabilities = []
        labels = []

        n_instances = len(dataset)
        n_batches = math.ceil(n_instances / predict_batch_size)
        with torch.no_grad():
            # set the model to evaluation mode
            model.eval()

            for i, (mb_ids, mb_features, mb_labels) in tqdm.tqdm(
                    enumerate(data_loader),
                    total=n_batches,
                    **settings.TQDM_KWARGS):
                # move the data onto the device
                mb_input_ids = mb_features['input_ids'].to(device)
                mb_input_mask = mb_features['input_mask'].to(device)
                mb_segment_ids = mb_features['segment_ids'].to(device)

                mb_labels = mb_labels.to(device)

                # make predictions
                mb_logits = model(
                    input_ids=mb_input_ids,
                    attention_mask=mb_input_mask,
                    token_type_ids=mb_segment_ids)
                _, mb_predictions = torch.max(mb_logits, 1)

                ids.extend(mb_ids)
                predictions.extend(mb_predictions.cpu().numpy().tolist())
                probabilities.extend(
                    softmax(mb_logits.cpu().numpy(), axis=1).tolist())
                labels.extend(mb_labels.cpu().numpy().tolist())

        with open(metrics_paths[split], 'w') as metrics_file:
            json.dump(
                {
                    key: metric(
                        y_true=labels,
                        y_pred=probabilities
                          if scorer_kwargs['needs_proba']
                          else predictions)
                    for key, (_, metric, scorer_kwargs) in METRICS.items()
                },
                metrics_file)

        with open(predictions_paths[split], 'w') as predictions_file:
            for id_, probs, prediction in zip(
                    ids, probabilities, predictions
            ):
                predictions_file.write(
                    json.dumps({
                        'id': id_,
                        'label': delabelize(prediction),
                        'label_scores': {
                            label.name: prob
                            for label, prob
                            in zip(Label, probs)
                        }
                    }) + '\n')


if __name__ == '__main__':
    predict_bert()

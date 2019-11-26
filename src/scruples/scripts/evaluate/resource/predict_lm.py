"""Predict labels for the resource with a fine-tuned language model."""

import json
import math
import logging
import os
from typing import List, Optional

import click
from scipy.special import softmax
import torch
from torch.utils.data import DataLoader
import tqdm

from .... import settings, baselines
from ....baselines.metrics import METRICS
from ....dataset.readers import ScruplesResourceDataset
from ....baselines.utils import dirichlet_multinomial


logger = logging.getLogger(__name__)


# main function

@click.command()
@click.argument(
    'data_dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument(
    'model_dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument(
    'results_dir',
    type=click.Path(exists=False, file_okay=False, dir_okay=True))
@click.argument(
    'splits', type=click.Choice(ScruplesResourceDataset.SPLITS), nargs=-1)
@click.option(
    '--predict-batch-size', type=int, default=64,
    help='The batch size for prediction.')
@click.option(
    '--gpu-ids', type=str, default=None,
    help='The GPU IDs to use for training as a comma-separated list.')
def predict_lm(
        data_dir: str,
        model_dir: str,
        results_dir: str,
        splits: List[str],
        predict_batch_size: int,
        gpu_ids: Optional[str]
) -> None:
    """Predict using a fine-tuned LM baseline on the resource.

    Read the resource dataset from DATA_DIR, load the fine-tuned LM
    baseline model from MODEL_DIR, and write the metrics and predictions
    to RESULTS_DIR, for each split provided as an argument.
    """
    # Step 1: Manage and construct paths.

    logger.info('Creating the results directory.')

    os.makedirs(results_dir)
    metrics_paths = {}
    predictions_paths = {}
    for split in splits:
        os.makedirs(os.path.join(results_dir, split))
        metrics_paths[split] = os.path.join(
            results_dir, split, 'metrics.json')
        predictions_paths[split] = os.path.join(
            results_dir, split, 'predictions.jsonl')

    config_file_path = os.path.join(model_dir, 'config.json')
    checkpoint_file_path = os.path.join(
        model_dir, 'checkpoints', 'best.checkpoint.pkl')

    # Step 2: Read in the training arguments.

    logger.info('Reading in the training arguments.')

    with open(config_file_path, 'r') as config_file:
        config = json.load(config_file)

    Model, baseline_config, _, make_transform =\
        baselines.resource.FINE_TUNE_LM_BASELINES[config['baseline']]

    # Step 3: Configure GPUs.

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

    # Step 4: Load the model.

    logger.info('Loading the fine-tuned model.')

    model = torch.nn.DataParallel(Model(**baseline_config['model']))
    model.load_state_dict(torch.load(checkpoint_file_path)['model'])

    model.to(device)

    # Step 5: Create transformations for the dataset.

    featurize = make_transform(**baseline_config['transform'])

    # Step 6: Make predictions for the splits.

    for split in splits:
        # load the split

        logger.info(f'Loading the dataset from {data_dir}.')

        dataset = ScruplesResourceDataset(
            data_dir=data_dir,
            split=split,
            transform=featurize)

        data_loader = DataLoader(
            dataset=dataset,
            batch_size=predict_batch_size,
            shuffle=False)

        # run predictions

        logger.info(f'Running predictions for {split}.')

        ids = []
        predictions = []
        probabilities = []
        labels = []

        n_instances = len(dataset)
        n_batches = math.ceil(n_instances / predict_batch_size)
        with torch.no_grad():
            # set the model to evaluation mode
            model.eval()

            for i, (mb_ids, mb_features, mb_labels, _) in tqdm.tqdm(
                    enumerate(data_loader),
                    total=n_batches,
                    **settings.TQDM_KWARGS
            ):
                # move the data onto the device
                mb_features = {k: v.to(device) for k, v in mb_features.items()}
                mb_labels = mb_labels.to(device)

                # make predictions
                mb_logits = model(**mb_features)[0]
                _, mb_predictions = torch.max(mb_logits, 1)

                ids.extend(mb_ids)
                predictions.extend(mb_predictions.cpu().numpy().tolist())
                if (
                        config['loss_type'] == 'xentropy-hard'
                        or config['loss_type'] == 'xentropy-soft'
                        or config['loss_type'] == 'xentropy-full'
                ):
                    probabilities.extend(
                        softmax(mb_logits.cpu().numpy(), axis=1).tolist())
                elif config['loss_type'] == 'dirichlet-multinomial':
                    probabilities.extend(dirichlet_multinomial(
                        mb_logits.cpu().numpy()).tolist())
                labels.extend(mb_labels.cpu().numpy().tolist())

        # write metrics to disk

        logger.info(f'Writing metrics for {split} to disk.')

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

        # write predictions to disk

        logger.info(f'Writing predictions for {split} to disk.')

        with open(predictions_paths[split], 'w') as predictions_file:
            for id_, probs, prediction in zip(
                    ids, probabilities, predictions
            ):
                predictions_file.write(
                    json.dumps({
                        'id': id_,
                        'label': prediction,
                        'label_scores': probs
                    }) + '\n')

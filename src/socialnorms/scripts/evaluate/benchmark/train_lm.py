"""Fine-tune pre-trained LMs on the socialnorms benchmark."""

import logging
from typing import Optional

import click

from .... import baselines


logger = logging.getLogger(__name__)


# main function

@click.command()
@click.argument(
    'data_dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument(
    'model_dir',
    type=click.Path(exists=False, file_okay=False, dir_okay=True))
@click.option(
    '--baseline',
    type=click.Choice(baselines.benchmark.FINE_TUNE_LM_BASELINES.keys()),
    default='bert',
    help='The model to train. Defaults to "bert".')
@click.option(
    '--train-batch-size', type=int, default=32,
    help='The batch size for training. Defaults to 32.')
@click.option(
    '--predict-batch-size', type=int, default=64,
    help='The batch size for prediction. Defaults to 64.')
@click.option(
    '--n-epochs', type=int, default=3,
    help='The number of epochs to train the model. Defaults to 3.')
@click.option(
    '--n-gradient-accumulation', type=int, default=8,
    help='The number of gradient accumulation steps. Defaults to 8.')
@click.option(
    '--mixed-precision', is_flag=True,
    help='Use mixed precision training.')
@click.option(
    '--gpu-ids', type=str, default=None,
    help='The GPU IDs to use for training as a comma-separated list.')
def train_lm(
        data_dir: str,
        model_dir: str,
        baseline: str,
        train_batch_size: int,
        predict_batch_size: int,
        n_epochs: int,
        n_gradient_accumulation: int,
        mixed_precision: bool,
        gpu_ids: Optional[str]
) -> None:
    """Fine-tune a pre-trained LM baseline on the socialnorms benchmark.

    Fine-tune a pre-trained language model on the socialnorms benchmark,
    reading the dataset from DATA_DIR, and writing checkpoint files,
    logs, and other results to MODEL_DIR.
    """
    # parse the --gpu-ids option
    gpu_ids = [int(gpu_id) for gpu_id in gpu_ids.split(',')]

    # train the baseline
    baselines.train.train_lm(
        data_dir=data_dir,
        model_dir=model_dir,
        dataset='benchmark',
        baseline=baseline,
        train_batch_size=train_batch_size,
        predict_batch_size=predict_batch_size,
        n_epochs=n_epochs,
        n_gradient_accumulation=n_gradient_accumulation,
        mixed_precision=mixed_precision,
        gpu_ids=gpu_ids,
        logger=logger)

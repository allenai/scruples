"""Tune hyper-parameters for pre-trained LMs on the corpus."""

import json
import logging
import os
from typing import Optional

import click
import skopt

from .... import baselines, settings, utils
from ....vendor.skopt import CheckpointSaver


logger = logging.getLogger(__name__)


# constants

COMPRESSION_LEVEL = 9


# main function

@click.command()
@click.argument(
    'data_dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument(
    'models_dir',
    type=click.Path(file_okay=False, dir_okay=True))
@click.option(
    '--baseline',
    type=click.Choice(baselines.corpus.FINE_TUNE_LM_BASELINES.keys()),
    default='bert',
    help='The model to train. Defaults to "bert".')
@click.option(
    '--loss-type',
    type=click.Choice(settings.LOSS_TYPES),
    default='xentropy-hard',
    help='The loss type to use for training.')
@click.option(
    '--n-iter', type=int, default=256,
    help='The number of iterations of Bayesian optimization to run when'
         ' tuning the hyper-parameters. Defaults to 256.')
@click.option(
    '--compute-train-batch-size', type=int, default=4,
    help='The largest batch size that can fit on the hardware during'
         ' training. Gradient accumulation will be used to make sure'
         ' the actual size of the batch on the hardware respects this'
         ' limit. Defaults to 4.')
@click.option(
    '--predict-batch-size', type=int, default=64,
    help='The batch size for prediction. Defaults to 64.')
@click.option(
    '--mixed-precision', is_flag=True,
    help='Use mixed precision training.')
@click.option(
    '--gpu-ids', type=str, default=None,
    help='The GPU IDs to use for training as a comma-separated list.')
def tune_lm(
        data_dir: str,
        models_dir: str,
        baseline: str,
        loss_type: str,
        n_iter: int,
        compute_train_batch_size: int,
        predict_batch_size: int,
        mixed_precision: bool,
        gpu_ids: Optional[str]
) -> None:
    """Tune hyper-parameters for pre-trained LMs on the corpus.

    Tune hyper-parameters for a pre-trained language model on the
    scruples corpus, reading the dataset from DATA_DIR, and writing
    all artifacts to MODELS_DIR. Tuning is performed with Bayesian
    optimization.
    """
    # Step 1: Manage and construct paths.

    os.makedirs(models_dir, exist_ok=True)

    best_model_hyper_parameters_path = os.path.join(
        models_dir, 'best-model-hyper-parameters.jsonl')
    optimization_checkpoint_path = os.path.join(
        models_dir, 'optimization-checkpoint.pkl.gz')
    optimization_result_path = os.path.join(
        models_dir, 'optimization-results.pkl.gz')

    # Step 2: Fetch the hyper-parameter space.

    _, _, space, _ = baselines.corpus.FINE_TUNE_LM_BASELINES[baseline]

    # Step 3: Create the objective function.

    # parse the --gpu-ids option
    gpu_ids = [int(gpu_id) for gpu_id in gpu_ids.split(',')]

    @skopt.utils.use_named_args(space)
    def _objective(**hyper_params):
        # normalize the hyper_params dict
        hyper_params['train_batch_size'] = 2 ** hyper_params.pop(
            'log_train_batch_size')
        hyper_params = {k: v.tolist() for k, v in hyper_params.items()}

        # run the training
        best_dev_accuracy, diverged = baselines.train.train_lm(
            data_dir=data_dir,
            model_dir=utils.next_unique_path(
                os.path.join(
                    models_dir,
                    '_'.join(
                        f'{k}-{v}'
                        for k, v in hyper_params.items()))),
            dataset='corpus',
            baseline=baseline,
            hyper_params=hyper_params,
            loss_type=loss_type,
            compute_train_batch_size=compute_train_batch_size,
            predict_batch_size=predict_batch_size,
            mixed_precision=mixed_precision,
            gpu_ids=gpu_ids,
            logger=logger)

        return - (best_dev_accuracy if not diverged else -1)

    # Step 4: Check if there is a checkpoint.

    if os.path.exists(optimization_checkpoint_path):
        logger.info(
            f'Loading checkpoint {optimization_checkpoint_path}.')
        checkpoint_result = skopt.load(optimization_checkpoint_path)
        x0 = checkpoint_result.x_iters
        y0 = checkpoint_result.func_vals
    else:
        logger.info(
            'No pre-existing checkpoint found. Starting new optimization.')
        x0 = None
        y0 = None

    # Step 5: Tune the hyper-parameters.

    logger.info('Beginning optimization.')

    n_calls = n_iter
    n_random_starts = min(
        max(10, n_iter // 8),
        n_iter)
    if x0 is not None:
        n_calls = max(n_calls - len(x0), 0)
        n_random_starts = max(n_random_starts - len(x0), 0)

    optimization_result = skopt.gp_minimize(
        func=_objective,
        dimensions=space,
        n_calls=n_calls,
        n_random_starts=n_random_starts,
        x0=x0,
        y0=y0,
        callback=[
            CheckpointSaver(
                optimization_checkpoint_path,
                compress=COMPRESSION_LEVEL,
                # do not store the objective because it's not
                # serializable
                store_objective=False)
        ])

    # Step 6: Save the results.

    logger.info('Optimization complete. Saving results.')

    with open(best_model_hyper_parameters_path, 'w') \
         as best_model_hyper_parameters_file:
        json.dump(
            {
                dim.name: value.tolist()
                for dim, value in zip(
                        optimization_result.space.dimensions,
                        optimization_result.x)
            },
            best_model_hyper_parameters_file)

    skopt.dump(
        optimization_result,
        optimization_result_path,
        compress=COMPRESSION_LEVEL,
        # do not store the objective because it's not serializable
        store_objective=False)

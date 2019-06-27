"""Fine-tune pre-trained LMs on the socialnorms benchmark."""

import json
import logging
import math
import os
import shutil
from typing import Optional

from apex.optimizers import (
    FP16_Optimizer,
    FusedAdam)
import click
from pytorch_pretrained_bert.optimization import (
    BertAdam,
    warmup_linear)
import tensorboardX
import torch
from torch.utils.data import DataLoader
import tqdm

from .... import settings, baselines
from ....dataset.readers import SocialnormsBenchmarkDataset


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
    type=click.Choice(baselines.BENCHMARK_FINE_TUNE_LM_BASELINES.keys()),
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
    # Step 1: Manage and construct paths.

    logger.info('Creating the model directory.')

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    tensorboard_dir = os.path.join(model_dir, 'tensorboard')
    os.makedirs(model_dir)
    os.makedirs(checkpoints_dir)
    os.makedirs(tensorboard_dir)

    config_file_path = os.path.join(model_dir, 'config.json')
    log_file_path = os.path.join(model_dir, 'log.txt')
    best_checkpoint_path = os.path.join(
        checkpoints_dir, 'best.checkpoint.pkl')
    last_checkpoint_path = os.path.join(
        checkpoints_dir, 'last.checkpoint.pkl')

    # Step 2: Setup the log file.

    logger.info('Configuring log files.')

    log_file_handler = logging.FileHandler(log_file_path)
    log_file_handler.setLevel(logging.DEBUG)
    log_file_handler.setFormatter(logging.Formatter(settings.LOG_FORMAT))
    logging.root.addHandler(log_file_handler)

    # Step 3: Record the script's arguments.

    logger.info(f'Writing arguments to {config_file_path}.')

    with open(config_file_path, 'w') as config_file:
        json.dump({
            'data_dir': data_dir,
            'model_dir': model_dir,
            'baseline': baseline,
            'train_batch_size': train_batch_size,
            'predict_batch_size': predict_batch_size,
            'n_epochs': n_epochs,
            'n_gradient_accumulation': n_gradient_accumulation,
            'mixed_precision': mixed_precision,
            'gpu_ids': gpu_ids
        }, config_file)

    # Step 4: Configure GPUs.

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

    # Step 5: Fetch the baseline information.

    logger.info('Retrieving baseline and related parameters.')

    Model, hyper_params, make_transform =\
        baselines.BENCHMARK_FINE_TUNE_LM_BASELINES[baseline]

    # Step 6: Load the dataset.

    logger.info(f'Loading the dataset from {data_dir}.')

    featurize = make_transform(**hyper_params['transform'])

    train = SocialnormsBenchmarkDataset(
        data_dir=data_dir,
        split='train',
        transform=featurize)
    dev = SocialnormsBenchmarkDataset(
        data_dir=data_dir,
        split='dev',
        transform=featurize)

    train_loader = DataLoader(
        dataset=train,
        batch_size=train_batch_size // n_gradient_accumulation,
        shuffle=True,
        num_workers=len(gpu_ids),
        pin_memory=bool(gpu_ids))
    dev_loader = DataLoader(
        dataset=dev,
        batch_size=predict_batch_size,
        shuffle=False,
        num_workers=len(gpu_ids),
        pin_memory=bool(gpu_ids))

    # Step 7: Create the model, optimizer, and loss.

    logger.info('Initializing the model.')

    model = torch.nn.DataParallel(Model(**hyper_params['model']))

    if mixed_precision:
        model.half()

    model.to(device)

    n_optimization_steps = n_epochs * math.ceil(len(train) / train_batch_size)
    parameter_groups = [
        {
            'params': [
                param
                for name, param in model.named_parameters()
                if 'bias' in name
                or 'LayerNorm.bias' in name
                or 'LayerNorm.weight' in name
            ],
            'weight_decay': 0
        },
        {
            'params': [
                param
                for name, param in model.named_parameters()
                if 'bias' not in name
                and 'LayerNorm.bias' not in name
                and 'LayerNorm.weight' not in name
            ],
            'weight_decay': hyper_params['optimizer']['weight_decay']
        }
    ]
    if mixed_precision:
        optimizer = FP16_Optimizer(
            FusedAdam(
                parameter_groups,
                lr=hyper_params['optimizer']['lr'],
                bias_correction=False,
                max_grad_norm=1.0),
            dynamic_loss_scale=True)
    else:
        optimizer = BertAdam(
            parameter_groups,
            lr=hyper_params['optimizer']['lr'],
            warmup=hyper_params['optimizer']['warmup_proportion'],
            t_total=n_optimization_steps)

    loss = torch.nn.CrossEntropyLoss()

    # Step 8: Run training.

    n_train_batches_per_epoch = math.ceil(len(train) / train_batch_size)
    n_dev_batch_per_epoch = math.ceil(len(dev) / predict_batch_size)

    writer = tensorboardX.SummaryWriter(log_dir=tensorboard_dir)

    best_dev_accuracy = - math.inf
    for epoch in range(n_epochs):
        # set the model to training mode
        model.train()

        # run training for the epoch
        epoch_train_loss = 0
        epoch_train_accuracy = 0
        for i, (_, features, labels) in tqdm.tqdm(
                enumerate(train_loader),
                total=n_gradient_accumulation * n_train_batches_per_epoch,
                **settings.TQDM_KWARGS
        ):
            # move the data onto the device
            features = {k: v.to(device) for k, v in features.items()}
            labels = labels.to(device)

            # make predictions
            logits = model(**features)
            _, predictions = torch.max(logits, 1)

            batch_loss = loss(logits, labels)
            batch_accuracy = (predictions == labels).float().mean()

            # update training statistics
            epoch_train_loss = (
                batch_loss.item() + i * epoch_train_loss
            ) / (i + 1)
            epoch_train_accuracy = (
                batch_accuracy.item() + i * epoch_train_accuracy
            ) / (i + 1)

            # update the network
            if mixed_precision:
                optimizer.backward(batch_loss)
            else:
                batch_loss.backward()

            if (i + 1) % n_gradient_accumulation == 0:
                if mixed_precision:
                    new_lr = warmup_linear(
                        (
                            n_train_batches_per_epoch * epoch
                            + ((i+1) // n_gradient_accumulation)
                        ) / n_optimization_steps,
                        hyper_params['optimizer']['warmup_proportion']
                    ) * hyper_params['optimizer']['lr']
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr

                optimizer.step()
                optimizer.zero_grad()

            # write training statistics to tensorboard

            step = n_train_batches_per_epoch * epoch + (
                (i + 1) // n_gradient_accumulation)
            if step % 100 == 0 and (i + 1) % n_gradient_accumulation == 0:
                writer.add_scalar('train/loss', epoch_train_loss, step)
                writer.add_scalar('train/accuracy', epoch_train_accuracy, step)

        # run evaluation
        with torch.no_grad():
            # set the model to evaluation mode
            model.eval()

            # run validation for the epoch
            epoch_dev_loss = 0
            epoch_dev_accuracy = 0
            for i, (_, features, labels) in tqdm.tqdm(
                    enumerate(dev_loader),
                    total=n_dev_batch_per_epoch,
                    **settings.TQDM_KWARGS):
                # move the data onto the device
                features = {k: v.to(device) for k, v in features.items()}
                labels = labels.to(device)

                # make predictions
                logits = model(**features)
                _, predictions = torch.max(logits, 1)

                batch_loss = loss(logits, labels)
                batch_accuracy = (predictions == labels).float().mean()

                # update validation statistics
                epoch_dev_loss = (
                    batch_loss.item() + i * epoch_dev_loss
                ) / (i + 1)
                epoch_dev_accuracy = (
                    batch_accuracy.item() + i * epoch_dev_accuracy
                ) / (i + 1)

            # write validation statistics to tensorboard
            writer.add_scalar('dev/loss', epoch_dev_loss, step)
            writer.add_scalar('dev/accuracy', epoch_dev_accuracy, step)

        logger.info(
            f'\n\n'
            f'  epoch {epoch}:\n'
            f'    train loss     : {epoch_train_loss:.4f}\n'
            f'    train accuracy : {epoch_train_accuracy:.4f}\n'
            f'    dev loss       : {epoch_dev_loss:.4f}\n'
            f'    dev accuracy   : {epoch_dev_accuracy:.4f}\n')

        # update checkpoints

        torch.save(
            {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            },
            last_checkpoint_path)

        # update the current best model
        if epoch_dev_accuracy > best_dev_accuracy:
            shutil.copyfile(last_checkpoint_path, best_checkpoint_path)
            best_dev_accuracy = epoch_dev_accuracy

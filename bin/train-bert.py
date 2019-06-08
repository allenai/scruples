"""Train BERT on socialnorms."""

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
from pytorch_pretrained_bert.modeling import (
    BertForSequenceClassification,
    PRETRAINED_MODEL_ARCHIVE_MAP)
from pytorch_pretrained_bert.optimization import (
    BertAdam,
    warmup_linear)
from pytorch_pretrained_bert.tokenization import BertTokenizer
import tensorboardX
import torch
from torch.utils.data import DataLoader
import tqdm

from socialnorms import settings, utils
from socialnorms.data.labels import Label
from socialnorms.dataset.readers import SocialnormsCorpusDataset
from socialnorms.dataset.transforms import (
    BertTransform,
    Compose)


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
    type=click.Path(exists=False, file_okay=False, dir_okay=True))
@click.option(
    '--train-batch-size', type=int, default=32,
    help='The batch size for training.')
@click.option(
    '--predict-batch-size', type=int, default=128,
    help='The batch size for prediction.')
@click.option(
    '--pretrained-bert',
    type=click.Choice(PRETRAINED_MODEL_ARCHIVE_MAP.keys()),
    default='bert-base-uncased',
    help='The pretrained BERT model to use. Defaults to'
         ' "bert-base-uncased".')
@click.option(
    '--max-sequence-length', type=int, default=512,
    help='The maximum sequence length accepted by the BERT model.')
@click.option(
    '--truncation-strategy-title',
    type=click.Choice(BertTransform.TRUNCATION_STRATEGIES),
    default='beginning',
    help='The strategy to use for truncating too long titles.')
@click.option(
    '--truncation-strategy-text',
    type=click.Choice(BertTransform.TRUNCATION_STRATEGIES),
    default='beginning',
    help='The strategy to use for truncating too long body texts.')
@click.option(
    '--lr', type=float, default=5e-5,
    help='The initial learning rate for Adam.')
@click.option(
    '--weight-decay', type=float, default=0.01,
    help='The weight decay to apply during optimization.')
@click.option(
    '--warmup-proportion', type=float, default=0.1,
    help='The proportion of optimization steps to use for warming up the'
          ' learning rate during training.')
@click.option(
    '--n-epochs', type=int, default=3,
    help='The number of epochs to train for.')
@click.option(
    '--mixed-precision', is_flag=True,
    help='Use mixed precision training.')
@click.option(
    '--n-gradient-accumulation', type=int, default=1,
    help='The number of gradient accumulation steps. Defaults to 1.')
@click.option(
    '--gpu-ids', type=str, default=None,
    help='The GPU IDs to use for training as a comma-separated list.')
@click.option(
    '--verbose', is_flag=True,
    help='Set the log level to DEBUG.')
def train_bert(
        cache_dir: str,
        data_dir: str,
        results_dir: str,
        train_batch_size: int,
        predict_batch_size: int,
        pretrained_bert: str,
        max_sequence_length: int,
        truncation_strategy_title: str,
        truncation_strategy_text: str,
        lr: float,
        weight_decay: float,
        warmup_proportion: float,
        n_epochs: int,
        mixed_precision: bool,
        n_gradient_accumulation: int,
        gpu_ids: Optional[str],
        verbose: bool
) -> None:
    """Train BERT on socialnorms and report dev performance.

    Train BERT on socialnorms, reading the dataset from DATA_DIR, and
    writing checkpoint files, logs, and other results to RESULTS_DIR.
    """
    # configure logging

    utils.configure_logging(verbose=verbose)


    # manage paths

    logging.info('Creating results directories.')

    checkpoints_dir = os.path.join(results_dir, 'checkpoints')
    tensorboard_dir = os.path.join(results_dir, 'tensorboard')
    os.makedirs(results_dir)
    os.makedirs(checkpoints_dir)
    os.makedirs(tensorboard_dir)

    config_file_path = os.path.join(results_dir, 'config.json')
    log_file_path = os.path.join(results_dir, 'log.txt')
    best_checkpoint_path = os.path.join(
        checkpoints_dir, 'best.checkpoint.pkl')
    last_checkpoint_path = os.path.join(
        checkpoints_dir, 'last.checkpoint.pkl')


    # setup the log file

    logging.info('Configuring log files.')

    log_file_handler = logging.FileHandler(log_file_path)
    log_file_handler.setLevel(logging.DEBUG)
    log_file_handler.setFormatter(logging.Formatter(settings.LOG_FORMAT))
    logging.root.addHandler(log_file_handler)


    # record the script arguments

    logger.info(f'Writing arguments to {config_file_path}.')

    with open(config_file_path, 'w') as config_file:
        json.dump({
            'cache_dir': cache_dir,
            'data_dir': data_dir,
            'results_dir': results_dir,
            'train_batch_size': train_batch_size,
            'predict_batch_size': predict_batch_size,
            'pretrained_bert': pretrained_bert,
            'max_sequence_length': max_sequence_length,
            'truncation_strategy_title': truncation_strategy_title,
            'truncation_strategy_text': truncation_strategy_text,
            'lr': lr,
            'weight_decay': weight_decay,
            'warmup_proportion': warmup_proportion,
            'n_epochs': n_epochs,
            'mixed_precision': mixed_precision,
            'n_gradient_accumulation': n_gradient_accumulation,
            'gpu_ids': gpu_ids,
            'verbose': verbose
        }, config_file)


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


    # load the dataset

    logger.info(f'Loading the dataset from {data_dir}.')

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

    train = SocialnormsCorpusDataset(
        data_dir=data_dir,
        split='train',
        transform=featurize,
        label_transform=labelize)
    dev = SocialnormsCorpusDataset(
        data_dir=data_dir,
        split='dev',
        transform=featurize,
        label_transform=labelize)

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


    # create the model, optimizer, and loss

    logger.info('Initializing the model.')

    model = torch.nn.DataParallel(
        BertForSequenceClassification.from_pretrained(
            pretrained_bert,
            cache_dir=cache_dir,
            num_labels=len(Label)))

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
            'weight_decay': weight_decay,
        },
        {
            'params': [
                param
                for name, param in model.named_parameters()
                if 'bias' not in name
                and 'LayerNorm.bias' not in name
                and 'LayerNorm.weight' not in name
            ],
            'weight_decay': 0
        }
    ]
    if mixed_precision:
        optimizer = FP16_Optimizer(
            FusedAdam(
                parameter_groups,
                lr=lr,
                bias_correction=False,
                max_grad_norm=1.0),
            dynamic_loss_scale=True)
    else:
        optimizer = BertAdam(
            parameter_groups,
            lr=lr,
            warmup=warmup_proportion,
            t_total=n_optimization_steps)

    loss = torch.nn.CrossEntropyLoss()


    # run training

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
                        warmup_proportion) * lr
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr

                optimizer.step()
                optimizer.zero_grad()

            # write training statistics to tensorboard

            step = i + n_train_batches_per_epoch * epoch
            if i % 100 == 0:
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
            f'    train loss    : {epoch_train_loss:.4f}\n'
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


if __name__ == '__main__':
    train_bert()

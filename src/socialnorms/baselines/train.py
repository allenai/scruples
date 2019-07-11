"""Fine-tune pre-trained LMs on the scruples datasets."""

import json
import logging
import math
import os
import shutil
from typing import (
    Any,
    Dict,
    List,
    Optional)

from apex.optimizers import (
    FP16_Optimizer,
    FusedAdam)
from pytorch_pretrained_bert.optimization import (
    BertAdam,
    warmup_linear)
import tensorboardX
import torch
from torch.utils.data import DataLoader
import tqdm

from . import benchmark, corpus
from .. import settings
from ..data.labels import Label
from ..dataset.readers import (
    ScruplesBenchmarkDataset,
    ScruplesCorpusDataset)


def train_lm(
        data_dir: str,
        model_dir: str,
        dataset: str,
        baseline: str,
        hyper_params: Dict[str, Any],
        compute_train_batch_size: int,
        predict_batch_size: int,
        mixed_precision: bool,
        gpu_ids: Optional[List[int]],
        logger: logging.Logger = None
) -> None:
    """Fine-tune a pre-trained LM baseline on a scruples dataset.

    Fine-tune ``baseline`` on ``dataset``, writing all results and
    artifacts to ``model_dir``. Return the best accuracy achieved on dev
    after any epoch.

    Parameters
    ----------
    data_dir : str
        The path to the directory containing the dataset.
    model_dir : str
        The path to the directory in which to save results.
    dataset : str
        The dataset to use when fine-tuning ``baseline``. Must be either
        "benchmark" or "corpus".
    baseline : str
        The pre-trained LM to fine-tune. Should be one of the keys for
        ``scruples.baselines.$dataset.FINE_TUNE_LM_BASELINES`` where
        ``$dataset`` corresponds to the ``dataset`` argument to this
        function.
    hyper_params : Dict[str, Any]
        The dictionary of hyper-parameters for the model.
    compute_train_batch_size : int
        The largest batch size that will fit on the hardware during
        training. Gradient accumulation will be used to make sure the
        actual size of the batch on the hardware respects this limit.
    predict_batch_size : int
        The number of instances to use in a predicting batch.
    gpu_ids : Optional[List[int]]
        A list of IDs for GPUs to use.
    logger : Optional[logging.Logger], optional (default=None)
        The logger to use when logging messages. If ``None``, then no
        messages will be logged.

    Returns
    -------
    float
        The best accuracy on dev achieved after any epoch.
    bool
        ``True`` if the training loss diverged, ``False`` otherwise.
    """
    # Step 1: Manage and construct paths.

    if logger is not None:
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

    if logger is not None:
        logger.info('Configuring log files.')

    log_file_handler = logging.FileHandler(log_file_path)
    log_file_handler.setLevel(logging.DEBUG)
    log_file_handler.setFormatter(logging.Formatter(settings.LOG_FORMAT))
    logging.root.addHandler(log_file_handler)

    # Step 3: Record the script's arguments.

    if logger is not None:
        logger.info(f'Writing arguments to {config_file_path}.')

    with open(config_file_path, 'w') as config_file:
        json.dump({
            'data_dir': data_dir,
            'model_dir': model_dir,
            'dataset': dataset,
            'baseline': baseline,
            'hyper_params': hyper_params,
            'compute_train_batch_size': compute_train_batch_size,
            'predict_batch_size': predict_batch_size,
            'mixed_precision': mixed_precision,
            'gpu_ids': gpu_ids
        }, config_file)

    # Step 4: Configure GPUs.

    if gpu_ids:
        if logger is not None:
            logger.info(
                f'Configuring environment to use {len(gpu_ids)} GPUs:'
                f' {", ".join(str(gpu_id) for gpu_id in gpu_ids)}.')

        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))

        if not torch.cuda.is_available():
            raise EnvironmentError('CUDA must be available to use GPUs.')

        device = torch.device('cuda')
    else:
        if logger is not None:
            logger.info('Configuring environment to use CPU.')

        device = torch.device('cpu')

    # Step 5: Fetch the baseline information and training loop parameters.

    if logger is not None:
        logger.info('Retrieving baseline and related parameters.')

    if dataset == 'benchmark':
        Model, baseline_config, _, make_transform =\
            benchmark.FINE_TUNE_LM_BASELINES[baseline]
    elif dataset == 'corpus':
        Model, baseline_config, _, make_transform =\
            corpus.FINE_TUNE_LM_BASELINES[baseline]
    else:
        raise ValueError(
            f'dataset must be either "benchmark" or "corpus", not'
            f' {dataset}.')

    n_epochs = hyper_params['n_epochs']
    train_batch_size = hyper_params['train_batch_size']
    n_gradient_accumulation = math.ceil(
        train_batch_size / (compute_train_batch_size * len(gpu_ids)))

    # Step 6: Load the dataset.

    if logger is not None:
        logger.info(f'Loading the dataset from {data_dir}.')

    featurize = make_transform(**baseline_config['transform'])
    if dataset == 'benchmark':
        Dataset = ScruplesBenchmarkDataset
        labelize = None
    elif dataset == 'corpus':
        Dataset = ScruplesCorpusDataset
        labelize = lambda s: getattr(Label, s).index
    else:
        raise ValueError(
            f'dataset must be either "benchmark" or "corpus", not'
            f' {dataset}.')

    train = Dataset(
        data_dir=data_dir,
        split='train',
        transform=featurize,
        label_transform=labelize)
    dev = Dataset(
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

    # Step 7: Create the model, optimizer, and loss.

    if logger is not None:
        logger.info('Initializing the model.')

    model = torch.nn.DataParallel(Model(**baseline_config['model']))

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
            'weight_decay': hyper_params['weight_decay']
        }
    ]
    if mixed_precision:
        optimizer = FP16_Optimizer(
            FusedAdam(
                parameter_groups,
                lr=hyper_params['lr'],
                bias_correction=False,
                max_grad_norm=1.0),
            dynamic_loss_scale=True)
    else:
        optimizer = BertAdam(
            parameter_groups,
            lr=hyper_params['lr'],
            warmup=hyper_params['warmup_proportion'],
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
                        hyper_params['warmup_proportion']
                    ) * hyper_params['lr']
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

            if logger is not None:
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

        # exit early if the training loss has diverged
        if math.isnan(epoch_train_loss):
            logger.info('Training loss has diverged. Exiting early.')

            return best_dev_accuracy, True

    logger.info(
        f'Training complete. Best dev accuracy was {best_dev_accuracy:.4f}')

    return best_dev_accuracy, False

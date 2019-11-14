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

import numpy as np
from transformers import (
    AdamW,
    WarmupLinearSchedule)
from scipy.special import softmax
import tensorboardX
import torch
from torch.utils.data import DataLoader
import tqdm

from . import resource, corpus
from .. import settings, utils
from ..baselines.loss import (
    SoftCrossEntropyLoss,
    DirichletMultinomialLoss)
from ..data.labels import Label
from ..dataset.readers import (
    ScruplesResourceDataset,
    ScruplesCorpusDataset)


def train_lm(
        data_dir: str,
        model_dir: str,
        dataset: str,
        baseline: str,
        hyper_params: Dict[str, Any],
        loss_type: str,
        compute_train_batch_size: int,
        predict_batch_size: int,
        gpu_ids: Optional[List[int]],
        logger: Optional[logging.Logger] = None
) -> None:
    """Fine-tune a pre-trained LM baseline on a scruples dataset.

    Fine-tune ``baseline`` on ``dataset``, writing all results and
    artifacts to ``model_dir``. Return the best calibrated xentropy achieved on
    dev after any epoch.

    Parameters
    ----------
    data_dir : str
        The path to the directory containing the dataset.
    model_dir : str
        The path to the directory in which to save results.
    dataset : str
        The dataset to use when fine-tuning ``baseline``. Must be either
        "resource" or "corpus".
    baseline : str
        The pre-trained LM to fine-tune. Should be one of the keys for
        ``scruples.baselines.$dataset.FINE_TUNE_LM_BASELINES`` where
        ``$dataset`` corresponds to the ``dataset`` argument to this
        function.
    hyper_params : Dict[str, Any]
        The dictionary of hyper-parameters for the model.
    loss_type : str
        The type of loss to use. Should be one of ``"xentropy-hard"``,
        ``"xentropy-soft"``, ``"xentropy-full"`` or
        ``"dirichlet-multinomial"``.
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
        The best calibrated xentropy on dev achieved after any epoch.
    bool
        ``True`` if the training loss diverged, ``False`` otherwise.
    """
    if loss_type not in settings.LOSS_TYPES:
        raise ValueError(
            f'Unrecognized loss type: {loss_type}. Please use one of'
            f' "xentropy-hard", "xentropy-soft", "xentropy-full" or'
            f' "dirichlet-multinomial".')

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
            'loss_type': loss_type,
            'compute_train_batch_size': compute_train_batch_size,
            'predict_batch_size': predict_batch_size,
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

    if dataset == 'resource':
        Model, baseline_config, _, make_transform =\
            resource.FINE_TUNE_LM_BASELINES[baseline]
    elif dataset == 'corpus':
        Model, baseline_config, _, make_transform =\
            corpus.FINE_TUNE_LM_BASELINES[baseline]
    else:
        raise ValueError(
            f'dataset must be either "resource" or "corpus", not'
            f' {dataset}.')

    n_epochs = hyper_params['n_epochs']
    train_batch_size = hyper_params['train_batch_size']
    n_gradient_accumulation = math.ceil(
        train_batch_size / (compute_train_batch_size * len(gpu_ids)))

    # Step 6: Load the dataset.

    if logger is not None:
        logger.info(f'Loading the dataset from {data_dir}.')

    featurize = make_transform(**baseline_config['transform'])
    if dataset == 'resource':
        Dataset = ScruplesResourceDataset
        labelize = None
        labelize_scores = lambda scores: np.array(scores).astype(float)
    elif dataset == 'corpus':
        Dataset = ScruplesCorpusDataset
        labelize = lambda s: getattr(Label, s).index
        labelize_scores = lambda scores: np.array([
            score
            for _, score in sorted(
                    scores.items(),
                    key=lambda t: labelize(t[0]))
        ]).astype(float)
    else:
        raise ValueError(
            f'dataset must be either "resource" or "corpus", not'
            f' {dataset}.')

    train = Dataset(
        data_dir=data_dir,
        split='train',
        transform=featurize,
        label_transform=labelize,
        label_scores_transform=labelize_scores)
    dev = Dataset(
        data_dir=data_dir,
        split='dev',
        transform=featurize,
        label_transform=labelize,
        label_scores_transform=labelize_scores)

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

    model = Model(**baseline_config['model'])
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
    optimizer = AdamW(parameter_groups, lr=hyper_params['lr'])

    if loss_type == 'xentropy-hard':
        loss = torch.nn.CrossEntropyLoss()
    elif loss_type == 'xentropy-soft':
        loss = SoftCrossEntropyLoss()
    elif loss_type == 'xentropy-full':
        loss = SoftCrossEntropyLoss()
    elif loss_type == 'dirichlet-multinomial':
        loss = DirichletMultinomialLoss()

    xentropy = SoftCrossEntropyLoss()

    scheduler = WarmupLinearSchedule(
        optimizer=optimizer,
        warmup_steps=int(
            hyper_params['warmup_proportion']
            * n_optimization_steps
        ),
        t_total=n_optimization_steps)

    # add data parallelism support
    model = torch.nn.DataParallel(model)

    # Step 8: Run training.

    n_train_batches_per_epoch = math.ceil(len(train) / train_batch_size)
    n_dev_batch_per_epoch = math.ceil(len(dev) / predict_batch_size)

    writer = tensorboardX.SummaryWriter(log_dir=tensorboard_dir)

    best_dev_calibrated_xentropy = math.inf
    for epoch in range(n_epochs):
        # set the model to training mode
        model.train()

        # run training for the epoch
        epoch_train_loss = 0
        epoch_train_xentropy = 0
        for i, (_, features, labels, label_scores) in tqdm.tqdm(
                enumerate(train_loader),
                total=n_gradient_accumulation * n_train_batches_per_epoch,
                **settings.TQDM_KWARGS
        ):
            # move the data onto the device
            features = {k: v.to(device) for k, v in features.items()}

            # create the targets
            if loss_type == 'xentropy-hard':
                targets = labels
            elif loss_type == 'xentropy-soft':
                targets = label_scores / torch.unsqueeze(
                    torch.sum(label_scores, dim=-1), dim=-1)
            elif loss_type == 'xentropy-full':
                targets = label_scores
            elif loss_type == 'dirichlet-multinomial':
                targets = label_scores
            # create the soft labels
            soft_labels = label_scores / torch.unsqueeze(
                torch.sum(label_scores, dim=-1), dim=-1)

            # move the targets and soft labels to the device
            targets = targets.to(device)
            soft_labels = soft_labels.to(device)

            # make predictions
            logits = model(**features)[0]

            batch_loss = loss(logits, targets)
            batch_xentropy = xentropy(logits, soft_labels)

            # update training statistics
            epoch_train_loss = (
                batch_loss.item() + i * epoch_train_loss
            ) / (i + 1)
            epoch_train_xentropy = (
                batch_xentropy.item() + i * epoch_train_xentropy
            ) / (i + 1)

            # update the network
            batch_loss.backward()

            if (i + 1) % n_gradient_accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()

                scheduler.step()

            # write training statistics to tensorboard

            step = n_train_batches_per_epoch * epoch + (
                (i + 1) // n_gradient_accumulation)
            if step % 100 == 0 and (i + 1) % n_gradient_accumulation == 0:
                writer.add_scalar('train/loss', epoch_train_loss, step)
                writer.add_scalar('train/xentropy', epoch_train_xentropy, step)

        # run evaluation
        with torch.no_grad():
            # set the model to evaluation mode
            model.eval()

            # run validation for the epoch
            epoch_dev_loss = 0
            epoch_dev_soft_labels = []
            epoch_dev_logits = []
            for i, (_, features, labels, label_scores) in tqdm.tqdm(
                    enumerate(dev_loader),
                    total=n_dev_batch_per_epoch,
                    **settings.TQDM_KWARGS):
                # move the data onto the device
                features = {k: v.to(device) for k, v in features.items()}

                # create the targets
                if loss_type == 'xentropy-hard':
                    targets = labels
                elif loss_type == 'xentropy-soft':
                    targets = label_scores / torch.unsqueeze(
                        torch.sum(label_scores, dim=-1), dim=-1)
                elif loss_type == 'xentropy-full':
                    targets = label_scores
                elif loss_type == 'dirichlet-multinomial':
                    targets = label_scores

                # move the targets to the device
                targets = targets.to(device)

                # make predictions
                logits = model(**features)[0]

                batch_loss = loss(logits, targets)

                # update validation statistics
                epoch_dev_loss = (
                    batch_loss.item() + i * epoch_dev_loss
                ) / (i + 1)
                epoch_dev_soft_labels.extend(
                    (
                        label_scores
                        / torch.unsqueeze(torch.sum(label_scores, dim=-1), dim=-1)
                    ).cpu().numpy().tolist()
                )
                epoch_dev_logits.extend(logits.cpu().numpy().tolist())

            # compute validation statistics
            epoch_dev_soft_labels = np.array(epoch_dev_soft_labels)
            epoch_dev_logits = np.array(epoch_dev_logits)

            calibration_factor = utils.calibration_factor(
                logits=epoch_dev_logits,
                targets=epoch_dev_soft_labels)

            epoch_dev_xentropy = utils.xentropy(
                y_true=epoch_dev_soft_labels,
                y_pred=softmax(epoch_dev_logits, axis=-1))
            epoch_dev_calibrated_xentropy = utils.xentropy(
                y_true=epoch_dev_soft_labels,
                y_pred=softmax(epoch_dev_logits / calibration_factor, axis=-1))

            # write validation statistics to tensorboard
            writer.add_scalar('dev/loss', epoch_dev_loss, step)
            writer.add_scalar('dev/xentropy', epoch_dev_xentropy, step)
            writer.add_scalar(
                'dev/calibrated-xentropy', epoch_dev_calibrated_xentropy, step)

            if logger is not None:
                logger.info(
                    f'\n\n'
                    f'  epoch {epoch}:\n'
                    f'    train loss              : {epoch_train_loss:.4f}\n'
                    f'    train xentropy          : {epoch_train_xentropy:.4f}\n'
                    f'    dev loss                : {epoch_dev_loss:.4f}\n'
                    f'    dev xentropy            : {epoch_dev_xentropy:.4f}\n'
                    f'    dev calibrated xentropy : {epoch_dev_calibrated_xentropy:.4f}\n'
                    f'    calibration factor      : {calibration_factor:.4f}\n')

        # update checkpoints

        torch.save(
            {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'calibration_factor': calibration_factor
            },
            last_checkpoint_path)

        # update the current best model
        if epoch_dev_calibrated_xentropy < best_dev_calibrated_xentropy:
            shutil.copyfile(last_checkpoint_path, best_checkpoint_path)
            best_dev_calibrated_xentropy = epoch_dev_calibrated_xentropy

        # exit early if the training loss has diverged
        if math.isnan(epoch_train_loss):
            logger.info('Training loss has diverged. Exiting early.')

            return best_dev_calibrated_xentropy, True

    logger.info(
        f'Training complete. Best dev calibrated xentropy was'
        f' {best_dev_calibrated_xentropy:.4f}.')

    return best_dev_calibrated_xentropy, False

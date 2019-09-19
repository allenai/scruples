"""Run a latent trait analysis on binary judgments."""

import json
import logging
import os

import click
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import torch as th

from ...analysis import traits


logger = logging.getLogger(__name__)


# helper functions

def _compute_1d_margins(xs: np.ndarray) -> np.ndarray:
    # compute margins as length 2 arrays where the 0 position
    # corresponds to an observation of 0, and the 1 position corresponds
    # to an observation of 1.
    n_samples, n_variables = xs.shape

    xs_1d_margins = {}
    for i in range(n_variables):
        margin = np.zeros(2)
        for pattern, count in zip(*np.unique(
                xs[:,i],
                return_counts=True)
        ):
            margin[pattern] = count / n_samples
        xs_1d_margins[str(i)] = margin.tolist()

    return xs_1d_margins


def _compute_2d_margins(xs: np.ndarray) -> np.ndarray:
    # compute margins as 2x2 matrices where the zeroth axis corresponds
    # to the variable with smaller index. The 0 position corresponds to
    # an observation of 0, and the 1 position corresponds to an
    # observation of 1.
    n_samples, n_variables = xs.shape

    xs_2d_margins = {}
    for i in range(n_variables):
        for j in range(n_variables):
            if i >= j:
                # make sure to only process each unique pair once
                continue

            margin = np.zeros((2, 2))
            for pattern, count in zip(*np.unique(
                    xs[:,[i,j]],
                    axis=0,
                    return_counts=True)
            ):
                margin[pattern] = count / n_samples

            xs_2d_margins[str((i, j))] = margin.tolist()

    return xs_2d_margins


# main function

@click.command()
@click.argument(
    'judgments_path',
    type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument(
    'output_dir',
    type=click.Path(exists=False, file_okay=False, dir_okay=True))
@click.option(
    '--n-latent-samples', type=int, default=10000,
    help='The number of latent samples for Monte Carlo integration.')
@click.option(
    '--lr', type=float, default=1e1,
    help='The starting learning rate. Defaults to 1e1.')
@click.option(
    '--n-batch-size', type=int, default=128,
    help='The batch size to use when fitting the models. Defaults to'
         ' 128.')
@click.option(
    '--patience', type=int, default=5,
    help='The number of epochs to wait for the loss to go down before'
         ' reducing the learning rate. Defaults to 5.')
@click.option(
    '--n-epochs', type=int, default=100,
    help='The number of epochs to run training. Defaults to 100.')
@click.option(
    '--max-latent-dim', type=int, default=10,
    help='The largest latent dimension to evaluate. Defaults to 10.')
@click.option(
    '--target-latent-dim', type=int, default=3,
    help='The target latent dimension, used for evaluating marginal'
         ' fit. Defaults to 3.')
@click.option(
    '--gpu-ids', type=str, default=None,
    help='The GPU IDs to use for training as a comma-separated list.')
def latent_traits(
        judgments_path: str,
        output_dir: str,
        n_latent_samples: int,
        lr: float,
        n_batch_size: int,
        patience: int,
        n_epochs: int,
        max_latent_dim: int,
        target_latent_dim: int,
        gpu_ids: str
) -> None:
    """Run a latent trait analysis on the binary moral judgments.

    Read the judgments from JUDGMENTS_PATH, fit a latent trait model,
    and write the results to OUTPUT_DIR.
    """
    # Step 0: Configure the environment.

    os.makedirs(output_dir)

    if gpu_ids:
        gpu_ids = [int(gpu_id) for gpu_id in gpu_ids.split(',')]

        logger.info(
            f'Configuring environment to use {len(gpu_ids)} GPUs:'
            f' {", ".join(str(gpu_id) for gpu_id in gpu_ids)}.')

        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))

        if not th.cuda.is_available():
            raise EnvironmentError('CUDA must be available to use GPUs.')

        device = th.device('cuda')
    else:
        logger.info('Configuring environment to use CPU.')

        device = th.device('cpu')

    # Step 1: Read the data.

    logger.info(f'Reading judgments from {judgments_path}.')

    with click.open_file(judgments_path, 'r') as judgments_file:
        judgments = [json.loads(ln) for ln in judgments_file]

    # index the annotators and the questions
    annotator_id_to_idx = {
        annotator_id: idx
        for idx, annotator_id in enumerate(
                set(judgment['annotator_id']
                    for judgment in judgments))
    }
    instance_id_to_idx = {
        instance_id: idx
        for idx, instance_id in enumerate(
                set(judgment['instance_id']
                    for judgment in judgments))
    }

    # construct the data matrix
    data = [
        [float('nan') for _ in range(len(instance_id_to_idx))]
        for _ in range(len(annotator_id_to_idx))
    ]
    for judgment in judgments:
        n = annotator_id_to_idx[judgment['annotator_id']]
        k = instance_id_to_idx[judgment['instance_id']]
        data[n][k] = judgment['label']

    data = np.array(data)

    if np.any(np.isnan(data)):
        raise ValueError('Found missing judgments in the data matrix.')

    # Step 2: Fit the models.

    models = []
    for latent_dim in range(0, max_latent_dim + 1):
        logger.info(f'Fitting model ({latent_dim} / {max_latent_dim}).')

        model = traits.LatentTraitModel(latent_dim=latent_dim)
        model.fit(
            data,
            n_latent_samples=n_latent_samples,
            lr=lr,
            n_batch_size=n_batch_size,
            patience=patience,
            n_epochs=n_epochs,
            device=device)
        models.append(model)

        with open(
                os.path.join(output_dir, f'weights-{latent_dim}d.json'), 'w'
        ) as weights_file:
            json.dump(
                {
                    'W': model.W_.numpy().tolist(),
                    'b': model.b_.numpy().tolist()
                },
                weights_file)

    # Step 3: Plot the deviances.

    # create the figure

    fig, ax_left = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))
    ax_right = ax_left.twinx()

    min_deviance = min(model.deviance_ for model in models)
    max_deviance = max(model.deviance_ for model in models)

    # plot the data

    ax_left.plot(
        [model.latent_dim for model in models],
        [model.deviance_ for model in models],
        linestyle=':',
        marker='o')

    # format the left axis

    ax_left.set_xlabel('# traits')
    ax_left.set_ylabel('deviance', va='bottom')
    ax_left.set_ylim(0.995 * min_deviance, 1.005 * max_deviance)

    # format the right axis

    ax_right.set_ylabel('% residual deviance', rotation=270, va='bottom')
    ax_right.set_ylim(0.995 * min_deviance / max_deviance * 100, 100.5)

    # save the figure
    plt.savefig(os.path.join(output_dir, 'comparing-the-number-of-traits.png'))
    plt.close()

    # Step 4: Plot the samples projected into the latent space.

    logger.info('Projecting and plotting data in 2D latent space.')

    # project the data into the latent space

    model_2d = models[2]
    zs = model_2d.project(data, device=device)

    # create the plot

    joint_plot = sns.JointGrid(x=zs[:,0], y=zs[:,1])
    joint_plot.plot_joint(sns.kdeplot, cmap='viridis')
    joint_plot.plot_joint(plt.scatter, s=10, color='black', edgecolor='white')
    joint_plot.plot_marginals(sns.kdeplot, color='mediumseagreen', shade=True)

    # save the figure
    plt.savefig(os.path.join(output_dir, 'projected-data.png'))
    plt.close()

    # Step 5: Plot the variables based on their loadings.

    logger.info('Plotting questions based on their loadings.')

    # get the loading matrix for the 2D model

    model_2d = models[2]
    loadings = model_2d.W_.numpy().T

    # create the plot

    joint_plot = sns.JointGrid(x=loadings[:,0], y=loadings[:,1])
    joint_plot.plot_joint(sns.kdeplot, cmap='viridis')
    joint_plot.plot_joint(plt.scatter, s=40, color='black', edgecolor='white')
    joint_plot.plot_marginals(sns.kdeplot, color='mediumseagreen', shade=True)

    # save the figure
    plt.savefig(os.path.join(output_dir, 'question-loadings.png'))
    plt.close()

    # Step 6: Assess the marginal fits.

    logger.info('Assessing marginal fits.')

    n_samples, n_variables = data.shape

    # compute and save the data's margins

    data_1d_margins = _compute_1d_margins(xs=data)
    with open(
            os.path.join(output_dir, 'data-1d-margins.json'), 'w'
    ) as data_1d_margins_file:
        json.dump(data_1d_margins, data_1d_margins_file)

    data_2d_margins = _compute_2d_margins(xs=data)
    with open(
            os.path.join(output_dir, 'data-2d-margins.json'), 'w'
    ) as data_2d_margins_file:
        json.dump(data_2d_margins, data_2d_margins_file)

    # compute and save the model's margins

    n_model_samples = 10000
    model_samples = models[target_latent_dim].sample(
        size=n_model_samples,
        device=device).numpy()

    model_1d_margins = _compute_1d_margins(xs=model_samples)
    with open(
            os.path.join(output_dir, 'model-1d-margins.json'), 'w'
    ) as model_1d_margins_file:
        json.dump(model_1d_margins, model_1d_margins_file)

    model_2d_margins = _compute_2d_margins(xs=model_samples)
    with open(
            os.path.join(output_dir, 'model-2d-margins.json'), 'w'
    ) as model_2d_margins_file:
        json.dump(model_2d_margins, model_2d_margins_file)

    # compute the X^2 statistics and residuals for marginal independence

    null_x2 = {}
    null_x2_residuals = {}
    for i in range(n_variables):
        for j in range(n_variables):
            if i >= j:
                # make sure to only process each unique pair once
                continue

            p_i = np.array(data_1d_margins[str(i)]).reshape(2, 1)
            p_j = np.array(data_1d_margins[str(j)])

            observed = np.array(data_2d_margins[str((i, j))])
            expected = p_i * p_j

            null_x2_residual = (observed - expected)**2 / expected

            null_x2[str((i, j))] = np.sum(null_x2_residual).item()
            null_x2_residuals[str((i, j))] = null_x2_residual.tolist()

    with open(
            os.path.join(output_dir, 'null-x2.json'), 'w'
    ) as null_x2_file:
        json.dump(null_x2, null_x2_file)

    with open(
            os.path.join(output_dir, 'null-x2-residuals.json'), 'w'
    ) as null_x2_residuals_file:
        json.dump(null_x2_residuals, null_x2_residuals_file)

    # compute the X^2 residuals for the model
    #
    # Note: unlike the X^2 residuals for the independent model, we
    # cannot sum these residuals to get a X^2 statistic / perform
    # inference, because these are _marginal_ residuals from a model
    # trained on the full data, which complicates things.
    #
    # These residuals are purely for heuristic interpretation to see
    # where the model may be fitting poorly on the margins.

    model_x2_residuals = {}
    for i in range(n_variables):
        for j in range(n_variables):
            if i >= j:
                # make sure to only process each unique pair once
                continue

            observed = np.array(data_2d_margins[str((i, j))])
            expected = np.array(model_2d_margins[str((i, j))])

            model_x2_residual = (observed - expected)**2 / expected

            model_x2_residuals[str((i, j))] = model_x2_residual.tolist()

    with open(
            os.path.join(output_dir, 'model-x2-residuals.json'), 'w'
    ) as model_x2_residuals_file:
        json.dump(model_x2_residuals, model_x2_residuals_file)

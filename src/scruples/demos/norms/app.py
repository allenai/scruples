"""The Flask app implementing the norms demo."""

import base64
import functools
import io
import json
import os
from typing import (
    Callable,
    Tuple)

import flask
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import torch

from . import utils
from ... import baselines, settings
from ...data.labels import Label


app = flask.Flask(__name__)
"""The norms demo.

Requires the following environment variables:

- ``SCRUPLES_NORMS_ACTIONS_MODEL``: The path to the Dirichlet-multinomial model
  trained on the resource (actions).
- ``SCRUPLES_NORMS_CORPUS_MODEL``: The path to the Dirichlet-multinomial model
  trained on the corpus.
- ``SCRUPLES_NORMS_PREDICT_BATCH_SIZE``: The batch size to use for predictions.
- ``SCRUPLES_NORMS_GPU_IDS``: The GPU IDs to use for making predictions.

"""


@functools.lru_cache()
def get_device() -> torch.device:
    """Return the device to use for computing predictions.

    Returns
    -------
    torch.device
        The torch device to use for computing predictions.
    """
    gpu_ids = settings.NORMS_GPU_IDS
    if gpu_ids:
        gpu_ids = [int(gpu_id) for gpu_id in gpu_ids.split(',')]

        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))

        if not torch.cuda.is_available():
            raise EnvironmentError('CUDA must be available to use GPUs.')

        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    return device


@functools.lru_cache()
def load_model(
        dataset: str,
        model_dir: str
) -> Tuple[torch.nn.Module, Callable, Callable]:
    """Return the model loaded from ``model_dir``.

    Parameters
    ----------
    dataset: str, required
        The model's corresponding dataset. Must be either ``'resource'`` or
        ``'corpus'``.
    model_dir : str, required
        The path to the model's experiment directory.

    Returns
    -------
    torch.nn.Module
        The model to use for prediction.
    Callable
        The function for featurizing the data, before inputting it to
        the model.
    Callable
        The function for converting predicted indices back into the
        original labels.
    """
    # construct the paths
    config_file_path = os.path.join(model_dir, 'config.json')
    checkpoint_file_path = os.path.join(
        model_dir, 'checkpoints', 'best.checkpoint.pkl')

    # read the config
    with open(config_file_path, 'r') as config_file:
        config = json.load(config_file)

    # fetch the baseline
    if dataset == 'resource':
        Model, baseline_config, _, make_transform =\
            baselines.resource.FINE_TUNE_LM_BASELINES[config['baseline']]
    elif dataset == 'corpus':
        Model, baseline_config, _, make_transform =\
            baselines.corpus.FINE_TUNE_LM_BASELINES[config['baseline']]
    else:
        raise ValueError(f'Unrecognized dataset: {dataset}.')

    # load the model
    model = torch.nn.DataParallel(Model(**baseline_config['model']))
    model.load_state_dict(torch.load(checkpoint_file_path)['model'])
    model.to(get_device())
    model.eval()

    # create the transforms
    featurize = make_transform(**baseline_config['transform'])
    delabelize = (
        lambda idx: next(l.name for l in Label if l.index == idx)
        if dataset == 'corpus' else
        lambda idx: idx
    )

    return (
        model,
        featurize,
        delabelize
    )


@app.route('/', methods=['GET'])
def home():
    """Return the home page."""
    response = flask.render_template('index.html')
    return response, 200


@app.route('/api/actions/predict', methods=['POST'])
def predict_actions():
    """Return the model's predictions for actions data.

    The post data should contain an array of JSON objects with the
    following structure::

        {
            "action1": $ACTION1_DESCRIPTION,
            "action2": $ACTION2_DESCRIPTION
        }

    The response is a corresponding array of predictions with the
    following structure:

        {
            "action1": '$ACTION1_ALPHA,
            "action2": $ACTION2_ALPHA
        }

    Where the alpha variables are the parameters for a beta
    distribution.

    If the query string contains the query parameter ``plot`` set to
    ``"true"``, then the returned objects will have an additional key
    ``"plot"`` giving a base64 encoded png of a kernel density
    estimation plot for the density described by the parameters.
    """
    instances = flask.request.json

    # Validate the input.

    errors = []
    # ``errors`` collects error messages to return to the client. It has
    # the following shape:
    #
    #     [{"error": $error_string, "message": $help_message}, ...]
    #

    # validate that the request has JSON
    if instances is None:
        errors.append({
            'error': 'No JSON',
            'message': 'Requests to this API endpoint must contain JSON.'
        })

    if len(errors) > 0:
        return flask.jsonify(errors), 400

    # validate that instances is an array of objects
    if not isinstance(instances, list):
        errors.append({
            'error': 'Wrong Type',
            'message': 'The request data must be an array of objects.'
        })
    elif any(not isinstance(instance, dict) for instance in instances):
        errors.append({
            'error': 'Wrong Type',
            'message': 'Each element of the data array must be an object.'
        })

    if len(errors) > 0:
        return flask.jsonify(errors), 400

    # validate that each object in instances has the right keys and values
    keys = set(['action1', 'action2'])
    for key in keys:
        if any(key not in instance for instance in instances):
            errors.append({
                'error': 'Missing Key',
                'message': f'Each object must have an "{key}" key.'
            })
        elif any(not isinstance(instance[key], str) for instance in instances):
            errors.append({
                'error': 'Wrong Type',
                'message': f'The value corresponding to the "{key}" key'
                            ' must be a string.'
            })
    if any(not set(instance.keys()).issubset(keys) for instance in instances):
        errors.append({
            'error': 'Unexpected Key',
            'message': 'Each object must only have "action1" and "action2"'
                       ' keys.'
        })

    if len(errors) > 0:
        return flask.jsonify(errors), 400

    # make the predictions
    device = get_device()

    model, featurize, _ = load_model(
        dataset='resource',
        model_dir=settings.NORMS_ACTIONS_MODEL)

    dataset = utils.PredictionDataset(
        features=[
            [instance['action1'], instance['action2']]
            for instance in instances
        ],
        transform=featurize)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=settings.NORMS_PREDICT_BATCH_SIZE,
        shuffle=False)

    response = []
    with torch.no_grad():
        for mb_features in data_loader:
            mb_features = {k: v.to(device) for k, v in mb_features.items()}
            mb_alphas = torch.exp(model(**mb_features)[0])

            response.extend(
                {'action1': alphas[0], 'action2': alphas[1]}
                for alphas in mb_alphas.cpu().numpy().tolist()
            )

    # add in the plots
    if flask.request.args.get('plot') == 'true':
        sns.set(style='white', palette='muted', color_codes=True)
        for actions, response_object in zip(instances, response):
            # create the figure
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()

            sns.kdeplot(
                stats.beta.rvs(
                    response_object['action2'],
                    response_object['action1'],
                    size=10000),
                clip=(0.0, 1.0),
                color='g',
                shade=True)

            ax1.set_xlim(0, 1)
            ax1.set_xlabel('probability')

            left_label = actions['action1']
            if len(left_label) > 35:
                left_label = left_label[:32] + '...'
            ax1.set_ylabel(left_label)
            ax1.set_yticks([])
            sns.despine(fig=fig, ax=ax1, left=True, right=True)

            right_label = actions['action2']
            if len(right_label) > 35:
                right_label = right_label[:32] + '...'
            ax2.set_ylabel(right_label, rotation=270, va='bottom')
            ax2.set_yticks([])
            sns.despine(fig=fig, ax=ax2, left=True, right=True)

            plt.tight_layout()

            # base64 encode the figure
            bytes_io = io.BytesIO()

            plt.savefig(bytes_io, format='png')
            plt.clf()
            bytes_io.seek(0)

            plot_string = base64.b64encode(bytes_io.read()).decode()

            response_object['plot'] = plot_string

    return flask.jsonify(response), 200


@app.route('/api/corpus/predict', methods=['POST'])
def predict_corpus():
    """Return the model's predictions for corpus data.

    The post data should contain an array of JSON objects with the
    following structure::

        {
            "title": $TITLE,
            "text": $TEXT
        }

    The response is a corresponding array of predictions with the
    following structure:

        {
            "AUTHOR": $author_alpha,
            "OTHER": $other_alpha,
            "EVERYBODY": $everybody_alpha,
            "NOBODY": $nobody_alpha,
            "INFO": $info_alpha
        }

    Where the alpha variables are the parameters for a dirichlet
    distribution.

    If the query string contains the query parameter ``plot`` set to
    ``"true"``, then the returned objects will have additional keys:
    ``"plot_author"``, and ``"plot_other"``, giving base64 encoded pngs
    of kernel density estimation plots for the predicted probability
    that someone would consider the author and the other in the wrong,
    respectively.
    """
    instances = flask.request.json

    # Validate the input.

    errors = []
    # ``errors`` collects error messages to return to the client. It has
    # the following shape:
    #
    #     [{"error": $error_string, "message": $help_message}, ...]
    #

    # validate that the request has JSON
    if instances is None:
        errors.append({
            'error': 'No JSON',
            'message': 'Requests to this API endpoint must contain JSON.'
        })

    if len(errors) > 0:
        return flask.jsonify(errors), 400

    # validate that instances is an array of objects
    if not isinstance(instances, list):
        errors.append({
            'error': 'Wrong Type',
            'message': 'The request data must be an array of objects.'
        })
    elif any(not isinstance(instance, dict) for instance in instances):
        errors.append({
            'error': 'Wrong Type',
            'message': 'Each element of the data array must be an object.'
        })

    if len(errors) > 0:
        return flask.jsonify(errors), 400

    # validate that the objects all have the correct keys
    keys = set(['title', 'text'])
    for key in keys:
        if any(key not in instance for instance in instances):
            errors.append({
                'error': 'Missing Key',
                'message': f'Each object must have a "{key}" key.'
            })
        elif any(not isinstance(instance[key], str) for instance in instances):
            errors.append({
                'error': 'Wrong Type',
                'message': f'The value corresponding to the "{key}" key'
                            ' must be a string.'
            })
    if any(not set(instance.keys()).issubset(keys) for instance in instances):
        errors.append({
            'error': 'Unexpected Key',
            'message': 'Each object must only have "title" and "text"'
                       ' keys.'
        })

    if len(errors) > 0:
        return flask.jsonify(errors), 400

    # make predictions
    device = get_device()

    model, featurize, delabelize = load_model(
        dataset='corpus',
        model_dir=settings.NORMS_CORPUS_MODEL)

    dataset = utils.PredictionDataset(
        features=[
            [instance['title'], instance['text']]
            for instance in instances
        ],
        transform=featurize)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=settings.NORMS_PREDICT_BATCH_SIZE,
        shuffle=False)

    response = []
    with torch.no_grad():
        for mb_features in data_loader:
            mb_features = {k: v.to(device) for k, v in mb_features.items()}
            mb_alphas = torch.exp(model(**mb_features)[0])

            response.extend([
                {
                    delabelize(i): v
                    for i, v in enumerate(alphas)
                }
                for alphas in mb_alphas.cpu().numpy().tolist()
            ])

    # add in the plots
    if flask.request.args.get('plot') == 'true':
        sns.set(style='white', palette='muted', color_codes=True)
        for anecdote, response_object in zip(instances, response):
            samples = stats.dirichlet.rvs(
                [
                    response_object['AUTHOR'],
                    response_object['OTHER'],
                    response_object['EVERYBODY'],
                    response_object['NOBODY'],
                    response_object['INFO']
                ],
                size=10000
            )

            # create the author plot
            fig, ax = plt.subplots()

            sns.kdeplot(
                samples[:,0] + samples[:,2],
                clip=(0.0, 1.0),
                color='g',
                shade=True)

            ax.set_xlim(0, 1)
            ax.set_xlabel('probability (AUTHOR or EVERYBODY)')

            ax.set_yticks([])
            sns.despine(fig=fig, ax=ax, left=True, right=True)

            plt.tight_layout()

            # base64 encode the author plot
            bytes_io = io.BytesIO()

            plt.savefig(bytes_io, format='png')
            plt.clf()
            bytes_io.seek(0)

            plot_string = base64.b64encode(bytes_io.read()).decode()

            response_object['plot_author'] = plot_string

            # create the other plot
            fig, ax = plt.subplots()

            sns.kdeplot(
                samples[:,1] + samples[:,2],
                clip=(0.0, 1.0),
                color='g',
                shade=True)

            ax.set_xlim(0, 1)
            ax.set_xlabel('probability (OTHER or EVERYBODY)')

            ax.set_yticks([])
            sns.despine(fig=fig, ax=ax, left=True, right=True)

            plt.tight_layout()

            # base64 encode the other plot
            bytes_io = io.BytesIO()

            plt.savefig(bytes_io, format='png')
            plt.clf()
            bytes_io.seek(0)

            plot_string = base64.b64encode(bytes_io.read()).decode()

            response_object['plot_other'] = plot_string

    return flask.jsonify(response), 200

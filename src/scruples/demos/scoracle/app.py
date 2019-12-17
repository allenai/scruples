"""The Flask app implementing the scoracle demo."""

import math

import flask
import numpy as np
from sklearn import metrics

from scruples.utils import xentropy, oracle_performance


app = flask.Flask(__name__)


SUPPORTED_METRICS = {
    'accuracy': (
        metrics.accuracy_score,
        {},
        lambda ys: np.argmax(ys, axis=-1)
    ),
    'balanced accuracy': (
        metrics.balanced_accuracy_score,
        {},
        lambda ys: np.argmax(ys, axis=-1)
    ),
    'f1 (macro)': (
        metrics.f1_score,
        {'average': 'macro'},
        lambda ys: np.argmax(ys, axis=-1)
    ),
    'cross entropy (soft labels)': (
        xentropy,
        {},
        None
    )
}
"""Supported metrics for the demo.

Each metric requires a triple::

    (metric, kwargs, make_predictions)

where ``metric`` is the function that computes the metric, ``kwargs`` is a
dictionary of kwargs to be used in calls to the metric, and
``make_predictions`` is a function that converts raw probabilities into
appropriate predictions for ``metric``.
"""
# N.B.! The keys for this dictionary are sent client side so that users
# can pick their desired metrics to compute on their datasets.


@app.route('/')
def home():
    """Return the home page."""
    response = flask.render_template(
        'index.html',
        metrics=SUPPORTED_METRICS.keys())
    return response, 200


@app.route(f'/api/score', methods=['POST'])
def score():
    """Return the oracle score for the post data.

    The post data should contain a JSON object with two keys:

      1. ``labelCounts``: An NxK JSON array, where N is the number of
         data points and K is the number of classes. The ij'th entry of
         the array should count the annotators that said example i is in
         class j.
      2. ``metrics``: A list of metrics to compute.

    """
    request_json = flask.request.json

    # Validate the input.

    errors = []
    # ``errors`` collects error messages to return to the client. It has
    # the following shape:
    #
    #   [{"error": $error_string, "message": $help_message}, ...]
    #

    # validate that the request has JSON
    if request_json is None:
        errors.append({
            'error': 'No JSON',
            'message': 'Requests to this API endpoint must contain JSON.'
        })
        return flask.jsonify(errors), 400

    # validate that the request object contains no extraneous keys
    unexpected_keys = (
        set(request_json.keys()) - set(['labelCounts', 'metrics'])
    )
    if len(unexpected_keys) > 0:
        errors.append({
            'error': 'Unexpected Key',
            'message': 'The request object only accepts the "labelCounts" and'
                       ' "metrics" keys.'
        })

    # validate the type and values of labelCounts
    label_counts = request_json.get('labelCounts')
    # validate that labelCounts is present
    if 'labelCounts' not in request_json:
        errors.append({
            'error': 'Missing Key',
            'message': 'Please include "labelCounts" in your request.'
        })
    # validate that labelCounts is an array of arrays of integers
    elif not isinstance(label_counts, list):
        errors.append({
            'error': 'Wrong Type',
            'message': 'labelCounts must be an array.'
        })
    elif any(not isinstance(r, list) for r in label_counts):
        errors.append({
            'error': 'Wrong Type',
            'message': 'labelCounts must be an array of arrays.'
        })
    elif any(not isinstance(x, int) for r in label_counts for x in r):
        errors.append({
            'error': 'Wrong Type',
            'message': 'labelCounts must be an array of arrays of integers.'
        })
    # validate that labelCounts has only nonnegative entries
    elif any(x < 0 for r in label_counts for x in r):
        errors.append({
            'error': 'Wrong Value',
            'message': 'labelCounts must have only nonnegative entries.'
        })
    # validate that labelCounts is rectangular and non-empty
    elif len(label_counts) == 0:
        errors.append({
            'error': 'Bad List Length',
            'message': 'labelCounts must be non-empty.'
        })
    elif len(label_counts[0]) == 0:
        errors.append({
            'error': 'Bad List Length',
            'message': 'The arrays in labelCounts must be non-empty.'
        })
    elif any(len(r) != len(label_counts[0]) for r in label_counts):
        errors.append({
            'error': 'Bad List Length',
            'message': 'Each array in labelCounts must have the same length.'
        })

    # validate the type and values of metrics
    metrics = request_json.get('metrics')
    # validate that metrics is present
    if 'metrics' not in request_json:
        errors.append({
            'error': 'Missing Key',
            'message': 'Please include "metrics" in your request.'
        })
    # validate that metrics is a list of strings
    elif not isinstance(metrics, list):
        errors.append({
            'error': 'Wrong Type',
            'message': 'metrics must be an array.'
        })
    elif any(not isinstance(s, str) for s in metrics):
        errors.append({
            'error': 'Wrong Type',
            'message': 'metrics must be an array of strings.'
        })
    # validate that every string in metrics is a supported metric
    elif any(s not in SUPPORTED_METRICS for s in metrics):
        errors.append({
            'error': 'Bad Metric',
            'message': 'Unsupported metric found in metrics.'
        })

    if len(errors) > 0:
        return flask.jsonify(errors), 400

    # Compute the oracle peformance
    ys = np.array(label_counts)

    response = []
    for metric_name in metrics:
        # attempt to answer the question quickly, using 625 samples
        metric, kwargs, make_predictions = SUPPORTED_METRICS[metric_name]
        score, std_err = oracle_performance(
            ys=ys,
            metric=lambda y_true, y_pred: metric(
                y_true=y_true, y_pred=y_pred, **kwargs),
            make_predictions=make_predictions,
            n_samples=625)
        if (2 * std_err) / score > 1e-2:
            # if the error is unacceptably high, fall back to using more
            # samples.
            score, _ = oracle_performance(
                ys=ys,
                metric=lambda y_true, y_pred: metric(
                    y_true=y_true, y_pred=y_pred, **kwargs),
                make_predictions=make_predictions,
                n_samples=10000)

        # JSON doesn't support NaN, Infinite, or -Infinite, so we have to
        # encode these values as strings.
        if math.isnan(score):
            score = 'NaN'
        elif math.isinf(score) and score > 0:
            score = 'Infinite'
        elif math.isinf(score) and score < 0:
            score = '-Infinite'

        response.append({
            'metric': metric_name,
            'score':  score
        })

    return flask.jsonify(response), 200

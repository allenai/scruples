"""Metrics for assessing baseline models."""

from sklearn import metrics


METRICS = {
    'accuracy': (
        'accuracy',
        metrics.accuracy_score,
        {
            'greater_is_better': True,
            'needs_proba': False
        }
    ),
    'balanced_accuracy': (
        'balanced accuracy',
        metrics.balanced_accuracy_score,
        {
            'greater_is_better': True,
            'needs_proba': False
        }
    ),
    'precision_micro': (
        'precision (micro)',
        lambda y_true, y_pred: metrics.precision_score(
            y_true=y_true,
            y_pred=y_pred,
            average='micro'),
        {
            'greater_is_better': True,
            'needs_proba': False
        }
    ),
    'recall_micro': (
        'recall (micro)',
        lambda y_true, y_pred: metrics.recall_score(
            y_true=y_true,
            y_pred=y_pred,
            average='micro'),
        {
            'greater_is_better': True,
            'needs_proba': False
        }
    ),
    'f1_micro': (
        'f1 (micro)',
        lambda y_true, y_pred: metrics.f1_score(
            y_true=y_true,
            y_pred=y_pred,
            average='micro'),
        {
            'greater_is_better': True,
            'needs_proba': False
        }
    ),
    'precision_macro': (
        'precision (macro)',
        lambda y_true, y_pred: metrics.precision_score(
            y_true=y_true,
            y_pred=y_pred,
            average='macro'),
        {
            'greater_is_better': True,
            'needs_proba': False
        }
    ),
    'recall_macro': (
        'recall (macro)',
        lambda y_true, y_pred: metrics.recall_score(
            y_true=y_true,
            y_pred=y_pred,
            average='macro'),
        {
            'greater_is_better': True,
            'needs_proba': False
        }
    ),
    'f1_macro': (
        'f1 (macro)',
        lambda y_true, y_pred: metrics.f1_score(
            y_true=y_true,
            y_pred=y_pred,
            average='macro'),
        {
            'greater_is_better': True,
            'needs_proba': False
        }
    ),
    'f1_weighted': (
        'f1 (weighted)',
        lambda y_true, y_pred: metrics.f1_score(
            y_true=y_true,
            y_pred=y_pred,
            average='weighted'),
        {
            'greater_is_better': True,
            'needs_proba': False
        }
    ),
    'log_loss': (
        'log loss',
        lambda y_true, y_pred: metrics.log_loss(
            y_true=y_true,
            y_pred=y_pred,
            eps=1e-9),
        {
            'greater_is_better': False,
            'needs_proba': True
        }
    ),
    'matthews_corrcoef': (
        'matthews correlation coefficient',
        metrics.matthews_corrcoef,
        {
            'greater_is_better': True,
            'needs_proba': False
        }
    )
    # N.B., do not include a key for "xentropy" in this dictionary.
    # "xentropy" is reserved for the cross-entropy between the predicted
    # probabilities and the dataset's label scores, which is computed in
    # the bin/analyze-predictions.py script.
}
"""A dictionary defining the important metrics to assess baselines.

The dictionary maps metric names to ``(name, metric, scorer_kwargs)``
tuples. ``name`` is the name of the metric, while ``metric`` is a
function for computing it, and ``scorer_kwargs`` is a dictionary
containing two keys: ``"greater_is_better"``, a boolean defining whether
or not higher values are better, and ``"needs_proba"``, a boolean
defining whether to pass the predicted labels or the predicted
probabilities.
"""

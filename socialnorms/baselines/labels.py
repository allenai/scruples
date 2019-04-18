"""Label only baselines."""

from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline


# the class prior baseline

PriorBaseline = Pipeline([
    ('classifier', DummyClassifier(strategy='prior'))
])
"""Predict using the label distribution."""

PRIOR_HYPER_PARAMETERS = {}
"""The hyper-param search space for ``PriorBaseline``."""


# the stratified sampling baseline

StratifiedBaseline = Pipeline([
    ('classifier', DummyClassifier(strategy='stratified'))
])
"""Predict by sampling a class according to its probability."""

STRATIFIED_HYPER_PARAMETERS = {}
"""The hyper-param search space for ``StratifiedBaseline``."""

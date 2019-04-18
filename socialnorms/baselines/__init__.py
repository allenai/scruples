"""Baselines for socialnorms."""

from . import (
    labels,
    linear,
    metrics,
    naivebayes,
    style,
    trees,
    utils)


BASELINES = [
    (
        'prior',
        labels.PriorBaseline,
        labels.PRIOR_HYPER_PARAMETERS
    ),
    (
        'stratified',
        labels.StratifiedBaseline,
        labels.STRATIFIED_HYPER_PARAMETERS
    ),
    (
        'logisticregression',
        linear.LogisticRegressionBaseline,
        linear.LOGISTIC_REGRESSION_HYPER_PARAMETERS
    ),
    (
        'bernoullinb',
        naivebayes.BernoulliNBBaseline,
        naivebayes.BERNOULLINB_HYPER_PARAMETERS
    ),
    (
        'multinomialnb',
        naivebayes.MultinomialNBBaseline,
        naivebayes.MULTINOMIALNB_HYPER_PARAMETERS
    ),
    (
        'complementnb',
        naivebayes.ComplementNBBaseline,
        naivebayes.COMPLEMENTNB_HYPER_PARAMETERS
    ),
    (
        'stylistic',
        style.StylisticXGBoostBaseline,
        style.STYLISTICXGBOOST_HYPER_PARAMETERS
    ),
    (
        'randomforest',
        trees.RandomForestBaseline,
        trees.RANDOM_FOREST_HYPER_PARAMETERS
    )
]

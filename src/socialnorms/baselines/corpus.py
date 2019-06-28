"""Baseline configuration for the socialnorms corpus."""

from . import (
    bert,
    labels,
    linear,
    naivebayes,
    style,
    trees)


SHALLOW_BASELINES = {
    'prior': (
        labels.PriorBaseline,
        labels.PRIOR_HYPER_PARAMETERS
    ),
    'stratified': (
        labels.StratifiedBaseline,
        labels.STRATIFIED_HYPER_PARAMETERS
    ),
    'logisticregression': (
        linear.LogisticRegressionBaseline,
        linear.LOGISTIC_REGRESSION_HYPER_PARAMETERS
    ),
    'bernoullinb': (
        naivebayes.BernoulliNBBaseline,
        naivebayes.BERNOULLINB_HYPER_PARAMETERS
    ),
    'multinomialnb': (
        naivebayes.MultinomialNBBaseline,
        naivebayes.MULTINOMIALNB_HYPER_PARAMETERS
    ),
    'complementnb': (
        naivebayes.ComplementNBBaseline,
        naivebayes.COMPLEMENTNB_HYPER_PARAMETERS
    ),
    'stylistic': (
        style.StylisticXGBoostBaseline,
        style.STYLISTICXGBOOST_HYPER_PARAMETERS
    ),
    'randomforest': (
        trees.RandomForestBaseline,
        trees.RANDOM_FOREST_HYPER_PARAMETERS
    )
}
"""Shallow baseline models for the socialnorms corpus."""


FINE_TUNE_LM_BASELINES = {
    'bert': (
        bert.BERTClassifier,
        bert.BERT_CLASSIFIER_CONFIG,
        bert.BERT_CLASSIFIER_HYPER_PARAM_SPACE,
        bert.BERT_CLASSIFIER_TRANSFORM
    )
}
"""Fine-tuned language model baselines for the socialnorms corpus."""

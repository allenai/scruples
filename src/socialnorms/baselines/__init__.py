"""Baselines for socialnorms."""

from . import (
    bert,
    labels,
    linear,
    metrics,
    naivebayes,
    style,
    trees,
    utils)


CORPUS_SHALLOW_BASELINES = {
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


CORPUS_FINE_TUNE_LM_BASELINES = {
    'bert': (
        bert.BERTClassifier,
        bert.BERT_CLASSIFIER_HYPER_PARAMETERS,
        bert.BERT_CLASSIFIER_TRANSFORM
    )
}
"""Fine-tuned language model baselines for the socialnorms corpus."""


BENCHMARK_SHALLOW_BASELINES = {}
"""Shallow baseline models for the socialnorms benchmark."""


BENCHMARK_FINE_TUNE_LM_BASELINES = {
    'bert': (
        bert.BERTRanker,
        bert.BERT_RANKER_HYPER_PARAMETERS,
        bert.BERT_RANKER_TRANSFORM
    )
}
"""Fine-tuned language model baselines for the socialnorms benchmark."""

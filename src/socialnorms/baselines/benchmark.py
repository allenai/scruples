"""Baseline configuration for the socialnorms benchmark."""

from . import (
    bert,
    labels)


SHALLOW_BASELINES = {
    'prior': (
        labels.PriorBaseline,
        labels.PRIOR_HYPER_PARAMETERS
    ),
    'stratified': (
        labels.StratifiedBaseline,
        labels.STRATIFIED_HYPER_PARAMETERS
    )
}
"""Shallow baseline models for the socialnorms benchmark."""


FINE_TUNE_LM_BASELINES = {
    'bert': (
        bert.BERTRanker,
        bert.BERT_RANKER_CONFIG,
        bert.BERT_RANKER_HYPER_PARAM_SPACE,
        bert.BERT_RANKER_TRANSFORM
    )
}
"""Fine-tuned language model baselines for the socialnorms benchmark."""

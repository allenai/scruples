"""Baseline configuration for the socialnorms benchmark."""

from . import bert


SHALLOW_BASELINES = {}
"""Shallow baseline models for the socialnorms benchmark."""


FINE_TUNE_LM_BASELINES = {
    'bert': (
        bert.BERTRanker,
        bert.BERT_RANKER_HYPER_PARAMETERS,
        bert.BERT_RANKER_TRANSFORM
    )
}
"""Fine-tuned language model baselines for the socialnorms benchmark."""

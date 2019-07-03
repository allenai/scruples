"""Baseline configuration for the socialnorms benchmark."""

from . import (
    bert,
    labels,
    linear,
    style)


SHALLOW_BASELINES = {
    'prior': (
        labels.PriorBaseline,
        labels.PRIOR_HYPER_PARAMETERS
    ),
    'stratified': (
        labels.StratifiedBaseline,
        labels.STRATIFIED_HYPER_PARAMETERS
    ),
    'fewestwords': (
        style.FewestWordsBaseline,
        style.FEWEST_WORDS_HYPER_PARAMETERS
    ),
    'mostwords': (
        style.MostWordsBaseline,
        style.MOST_WORDS_HYPER_PARAMETERS
    ),
    'fewestcharacters': (
        style.FewestCharactersBaseline,
        style.FEWEST_CHARACTERS_HYPER_PARAMETERS
    ),
    'mostcharacters': (
        style.MostCharactersBaseline,
        style.MOST_CHARACTERS_HYPER_PARAMETERS
    ),
    'logisticranker': (
        linear.LogisticRankerBaseline,
        linear.LOGISTIC_RANKER_HYPER_PARAMETERS
    ),
    'stylistic': (
        style.StyleRankerBaseline,
        style.STYLE_RANKER_HYPER_PARAMETERS
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

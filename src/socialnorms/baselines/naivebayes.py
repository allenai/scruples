"""Naive bayes baselines."""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import (
    BernoulliNB,
    MultinomialNB,
    ComplementNB)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from . import utils


# the bernoulli naive bayes baseline

BernoulliNBBaseline = Pipeline([
    (
        'concatenator',
        FunctionTransformer(func=utils.concat_title_and_text, validate=False)
    ),
    (
        'vectorizer',
        CountVectorizer(
            input='content',
            encoding='utf-8',
            decode_error='strict',
            preprocessor=None,
            tokenizer=None,
            token_pattern=r'(?u)\b\w\w+\b',
            max_features=None,
            vocabulary=None,
            binary=True)
    ),
    (
        'classifier',
        BernoulliNB(binarize=None, fit_prior=True)
    )
])
"""Predict using bernoulli naive bayes on bag-of-ngrams features."""

BERNOULLINB_HYPER_PARAMETERS = {
    'vectorizer__strip_accents': ['ascii', 'unicode', None],
    'vectorizer__lowercase': [True, False],
    'vectorizer__stop_words': ['english', None],
    'vectorizer__ngram_range': [
        (lo, hi)
        for lo in range(1, 2)
        for hi in range(lo, lo + 5)
    ],
    'vectorizer__analyzer': ['word', 'char', 'char_wb'],
    'vectorizer__max_df': (0.75, 1., 'uniform'),
    'vectorizer__min_df': (0., 0.25, 'uniform'),
    'classifier__alpha': (0., 5., 'uniform')
}
"""The hyper-param search space for ``BeroulliNBBaseline``."""


# the multinomial naive bayes baseline

MultinomialNBBaseline = Pipeline([
    (
        'concatenator',
        FunctionTransformer(func=utils.concat_title_and_text, validate=False)
    ),
    (
        'vectorizer',
        CountVectorizer(
            input='content',
            encoding='utf-8',
            decode_error='strict',
            preprocessor=None,
            tokenizer=None,
            token_pattern=r'(?u)\b\w\w+\b',
            max_features=None,
            vocabulary=None,
            binary=False)
    ),
    (
        'classifier',
        MultinomialNB(fit_prior=True)
    )
])
"""Predict using multinomial naive bayes on bag-of-ngrams features."""

MULTINOMIALNB_HYPER_PARAMETERS = {
    'vectorizer__strip_accents': ['ascii', 'unicode', None],
    'vectorizer__lowercase': [True, False],
    'vectorizer__stop_words': ['english', None],
    'vectorizer__ngram_range': [
        (lo, hi)
        for lo in range(1, 2)
        for hi in range(lo, lo + 5)
    ],
    'vectorizer__analyzer': ['word', 'char', 'char_wb'],
    'vectorizer__max_df': (0.75, 1., 'uniform'),
    'vectorizer__min_df': (0., 0.25, 'uniform'),
    'classifier__alpha': (0., 5., 'uniform')
}
"""The hyper-param search space for ``MultinomialNBBaseline``."""


# the complement naive bayes baseline

ComplementNBBaseline = Pipeline([
    (
        'concatenator',
        FunctionTransformer(func=utils.concat_title_and_text, validate=False)
    ),
    (
        'vectorizer',
        CountVectorizer(
            input='content',
            encoding='utf-8',
            decode_error='strict',
            preprocessor=None,
            tokenizer=None,
            token_pattern=r'(?u)\b\w\w+\b',
            max_features=None,
            vocabulary=None,
            binary=False)
    ),
    (
        'classifier',
        ComplementNB(fit_prior=True)
    )
])
"""Predict using complement naive bayes on bag-of-ngrams features."""

COMPLEMENTNB_HYPER_PARAMETERS = {
    'vectorizer__strip_accents': ['ascii', 'unicode', None],
    'vectorizer__lowercase': [True, False],
    'vectorizer__stop_words': ['english', None],
    'vectorizer__ngram_range': [
        (lo, hi)
        for lo in range(1, 2)
        for hi in range(lo, lo + 5)
    ],
    'vectorizer__analyzer': ['word', 'char', 'char_wb'],
    'vectorizer__max_df': (0.75, 1., 'uniform'),
    'vectorizer__min_df': (0., 0.25, 'uniform'),
    'classifier__alpha': (0., 5., 'uniform'),
    'classifier__norm': [True, False]
}
"""The hyper-param search space for ``ComplementNBBaseline``."""

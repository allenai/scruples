"""Linear model baselines."""

from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfTransformer)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from . import utils


# the logistic regression baseline

LogisticRegressionBaseline = Pipeline([
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
            vocabulary=None)
    ),
    (
        'tfidf',
        TfidfTransformer(smooth_idf=True)
    ),
    (
        'classifier',
        LogisticRegression(
            dual=False,
            tol=1e-4,
            intercept_scaling=1.,
            solver='saga',
            max_iter=100,
            warm_start=False)
    )
])
"""Predict using logistic regression on bag-of-ngrams features."""

LOGISTIC_REGRESSION_HYPER_PARAMETERS = {
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
    'vectorizer__binary': [True, False],
    'tfidf__norm': ['l1', 'l2', None],
    'tfidf__use_idf': [True, False],
    'tfidf__sublinear_tf': [True, False],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__C': (1e-6, 1e2, 'log-uniform'),
    'classifier__fit_intercept': [True, False],
    'classifier__class_weight': ['balanced', None],
    'classifier__multi_class': ['ovr', 'multinomial']
}
"""The hyper-param search space for ``LogisticRegressionBaseline``."""


# the logistic ranker baseline

LogisticRankerBaseline = Pipeline([
    (
        'featurizer',
        utils.BenchmarkTransformer(
            transformer=Pipeline([
                (
                    'vectorizer',
                    CountVectorizer(
                        input='content',
                        encoding='utf-8',
                        decode_error='featurizer',
                        preprocessor=None,
                        tokenizer=None,
                        token_pattern=r'(?u)\b\w\w+\b',
                        max_features=None,
                        vocabulary=None)
                ),
                (
                    'tfidf',
                    TfidfTransformer(smooth_idf=True)
                )
            ]))
    ),
    (
        'classifier',
        LogisticRegression(
            dual=False,
            tol=1e-4,
            fit_intercept=False,
            intercept_scaling=1.,
            solver='saga',
            max_iter=100,
            warm_start=False)
    )
])
"""Rank using logistic regression on bag-of-ngrams features."""

LOGISTIC_RANKER_HYPER_PARAMETERS = {
    'featurizer__transformer__vectorizer__strip_accents': ['ascii', 'unicode', None],
    'featurizer__transformer__vectorizer__lowercase': [True, False],
    'featurizer__transformer__vectorizer__stop_words': ['english', None],
    'featurizer__transformer__vectorizer__ngram_range': [
        (lo, hi)
        for lo in range(1, 2)
        for hi in range(lo, lo + 5)
    ],
    'featurizer__transformer__vectorizer__analyzer': ['word', 'char', 'char_wb'],
    'featurizer__transformer__vectorizer__max_df': (0.75, 1., 'uniform'),
    'featurizer__transformer__vectorizer__min_df': (0., 0.25, 'uniform'),
    'featurizer__transformer__vectorizer__binary': [True, False],
    'featurizer__transformer__tfidf__norm': ['l1', 'l2', None],
    'featurizer__transformer__tfidf__use_idf': [True, False],
    'featurizer__transformer__tfidf__sublinear_tf': [True, False],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__C': (1e-6, 1e2, 'log-uniform'),
    'classifier__class_weight': ['balanced', None],
}
"""The hyper-param search space for ``LogisticRankerBaseline``."""

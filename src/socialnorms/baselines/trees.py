"""Tree-based baselines."""

from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfTransformer)
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from . import utils


# the random forest baseline

RandomForestBaseline = Pipeline([
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
        RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            max_features='auto',
            max_leaf_nodes=None,
            min_impurity_decrease=0.,
            oob_score=False,
            n_jobs=1,
            verbose=0,
            warm_start=False)
    )
])
"""Predict using a random forest on bag-of-ngrams features."""

RANDOM_FOREST_HYPER_PARAMETERS = {
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
    'classifier__criterion': ['gini', 'entropy'],
    'classifier__min_samples_split': (2, 500),
    'classifier__min_samples_leaf': (1, 250),
    'classifier__min_weight_fraction_leaf': (0., .25, 'uniform'),
    'classifier__bootstrap': [True, False],
    'classifier__class_weight': ['balanced', 'balanced_subsample', None]
}
"""The hyper-param search space for ``RandomForestBaseline``."""

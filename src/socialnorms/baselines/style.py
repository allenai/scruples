"""Style features baselines."""

import string
from typing import Any, Iterable

import numpy as np
from sklearn.base import (
    BaseEstimator,
    TransformerMixin)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
import spacy
from spacy.parts_of_speech import IDS as POS_TAGS
from xgboost import XGBClassifier


# classes

class StyleFeaturizer(BaseEstimator, TransformerMixin):
    """Convert text to a suite a stylistic features.

    ``StyleFeaturizer`` converts text into a suite of style features,
    consisting of:

      1.  The length (in words) of the full document.
      2.  The length (in sentences) of the full document.
      3.  The minimum sentence length (in words).
      4.  The maximum sentence length (in words).
      5.  The median sentence length (in words).
      6.  The average sentence length (in words).
      7.  The standard deviation of sentence length (in words).
      8.  The lexical diversity (type-token ratio) of the full document.
      9.  The average lexical diversity (type-token ratio) of each
          sentence.
      10. The average word length (in characters), excluding
          punctuation.
      11. The average punctuation counts per sentence.
      12. The average POS tag counts per sentence.

    Attributes
    ----------
    See `Parameters`_.

    Parameters
    ----------
    This class takes no parameters.
    """
    _nlp = spacy.load('en', disable=['ner'])

    _PUNCT_TO_IDX = {
        punct: idx
        for idx, punct in enumerate(string.punctuation)
    }

    _POS_TAG_TO_IDX = {
        pos_tag: idx
        for idx, pos_tag in enumerate(POS_TAGS.keys())
    }

    def fit(self, X: Iterable[str], y: Any = None):
        """Fit the instance to ``X``.

        Fitting currently does nothing, but still must be called before
        transforming data to ensure backwards compatibility.

        Parameters
        ----------
        X : Iterable[str]
            An iterable of strings representing the training features.
        y : None
            Possibly the array of labels, though ``y`` is ignored.

        Returns
        -------
        self : object
            The instance.
        """
        self._fitted = True

        return self

    def transform(self, X: Iterable[str]):
        """Transform ``X`` into the stylistic features.

        Parameters
        ----------
        X : Iterable[str]
            An iterable of strings to transform into stylistic features.

        Returns
        -------
        np.ndarray[np.float64]
            ``X`` with each example transformed into an array of
            stylistic features.
        """
        check_is_fitted(self, '_fitted')

        style_features = []
        for doc in self._nlp.pipe(X, batch_size=64):
            sent_n_words = [len(sent) for sent in doc.sents]

            # compute the style features
            n_words = len(doc)
            n_sents = len(list(doc.sents))
            min_sent_n_words = np.min(sent_n_words)
            max_sent_n_words = np.max(sent_n_words)
            median_sent_n_words = np.median(sent_n_words)
            avg_sent_n_words = np.mean(sent_n_words)
            std_sent_n_words = np.std(sent_n_words)
            doc_lexical_diversity = len(set(t.lemma_ for t in doc)) / len(doc)
            avg_sent_lexical_diversity = np.mean([
                len(set(t.lemma_ for t in sent)) / len(sent)
                for sent in doc.sents
            ])
            avg_word_length = np.mean([
                len(token) for token in doc if not token.is_punct
            ])
            avg_punctuation_counts = [0 for _ in self._PUNCT_TO_IDX.keys()]
            for token in doc:
                if token.text in self._PUNCT_TO_IDX:
                    avg_punctuation_counts[self._PUNCT_TO_IDX[token.text]] +=\
                        1. / n_sents
            avg_pos_tag_counts = [0 for _ in self._POS_TAG_TO_IDX.keys()]
            for token in doc:
                if token.pos_ in self._POS_TAG_TO_IDX:
                    avg_pos_tag_counts[self._POS_TAG_TO_IDX[token.pos_]] +=\
                        1. / n_sents
            # append the instance to style_features
            style_features.append([
                n_words,
                n_sents,
                min_sent_n_words,
                max_sent_n_words,
                median_sent_n_words,
                avg_sent_n_words,
                std_sent_n_words,
                doc_lexical_diversity,
                avg_sent_lexical_diversity,
                avg_word_length,
                *avg_punctuation_counts,
                *avg_pos_tag_counts
            ])

        return np.array(style_features)


# the stylistic features baseline

StylisticXGBoostBaseline = Pipeline([
    (
        'featurizer',
        ColumnTransformer(
            [
                (
                    'title_style',
                    StyleFeaturizer(),
                    'title'
                ),
                (
                    'text_style',
                    StyleFeaturizer(),
                    'text'
                )
            ],
            remainder='drop')
    ),
    (
        'classifier',
        XGBClassifier(
            n_estimators=100,
            verbosity=0,
            objective='multi:softprob',
            booster='gbtree',
            n_jobs=1,
            max_delta_step=0,
            colsample_bylevel=1.,
            colsample_bynode=1.)
    )
])
"""Predict using a gradient boosted decision tree on style features."""

STYLISTICXGBOOST_HYPER_PARAMETERS = {
    'classifier__max_depth': (1, 10),
    'classifier__learning_rate': (1e-4, 1e1, 'log-uniform'),
    'classifier__gamma': (1e-15, 1e-1, 'log-uniform'),
    'classifier__min_child_weight': (1e-1, 1e2, 'log-uniform'),
    'classifier__subsample': (0.1, 1., 'uniform'),
    'classifier__colsample_bytree': (0.1, 1., 'uniform'),
    'classifier__reg_alpha': (1e-5, 1e1, 'log-uniform'),
    'classifier__reg_lambda': (1e-5, 1e1, 'log-uniform'),
    'classifier__scale_pos_weight': (0.1, 10, 'uniform'),
    'classifier__base_score': (0., 1., 'uniform')
}
"""The hyper-param search space for ``StylisticXGBoostBaseline``."""

"""Style features baselines."""

import string
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    TransformerMixin)
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.utils.validation import check_is_fitted, check_X_y
import spacy
from spacy.parts_of_speech import IDS as POS_TAGS
from xgboost import XGBClassifier

from ..utils import count_words
from . import utils


# classes

class LengthRanker(BaseEstimator, ClassifierMixin):
    """Choose an answer based on its length.

    Attributes
    ----------
    classes_ : List[int]
        The possible classes for the ranker (fitted from the data).

    See `Parameters`_ for more attributes.

    Parameters
    ----------
    choose : str, optional (default='shortest')
        Which answer to choose (based on length). Must either be
        ``'shortest'`` or ``'longest'``. Defaults to ``'shortest'``.
    length : str, optional (default='words')
        The way in which to measure length. Must either be ``'words'``
        or ``'characters'``. Defaults to ``'words'``.
    """
    def __init__(
            self,
            choose: str = 'shortest',
            length: str = 'words'
    ) -> None:
        self.choose = choose
        self.length = length

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray
    ) -> 'LengthRanker':
        """Fit the instance to ``X`` and ``y``.

        Parameters
        ----------
        X : np.ndarray
            An n x k array of strings where n is the number of instances
            and k is the number of choices.
        y : np.ndarray
            A length n array of integer labels, each giving the
            (0-based) index of the correct choice.

        Returns
        -------
        self : object
            The instance.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        X = np.array(X)

        if isinstance(y, pd.Series):
            y = y.values

        y = np.array(y)

        if self.choose not in ['shortest', 'longest']:
            raise ValueError(
                'The choose argument must be either "shortest" or "longest".')

        if self.length not in ['characters', 'words']:
            raise ValueError(
                'The length argument must be either "characters" or "words".')

        self.classes_ = list(range(X.shape[1]))

        return self

    def predict(
            self,
            X: np.ndarray
    ) -> np.ndarray:
        """Predict the correct answer based on length.

        Parameters
        ----------
        X : np.ndarray
            An n x k array of strings where n is the number of instances
            and k is the number of choices.

        Returns
        -------
        np.ndarray[int]
            The predicted labels.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        X = np.array(X)

        if self.length == 'words':
            lengths = np.array([[count_words(x) for x in xs] for xs in X])
        elif self.length == 'characters':
            lengths = np.array([[len(x) for x in xs] for xs in X])
        else:
            raise ValueError('length must either be "words" or "characters".')

        if self.choose == 'longest':
            labels = np.argmax(lengths, axis=1)
        elif self.choose == 'shortest':
            labels = np.argmax(-lengths, axis=1)
        else:
            raise ValueError('choose must be either "shortest" or "longest".')

        return labels

    def predict_proba(
            self,
            X: np.ndarray
    ) -> np.ndarray:
        """Predict probabilities for the answers based on length.

        All probability mass is placed on the chosen answer.

        Parameters
        ----------
        X : np.ndarray
            An n x k array of strings where n is the number of instances
            and k is the number of choices.

        Returns
        -------
        np.ndarray
            An array of arrays giving the predicted probabilities for
            each label.
        """
        return np.array([
            [1. if i == j else 0. for j in range(len(self.classes_))]
            for i in self.predict(X=X)
        ])


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

    def fit(
            self,
            X: Iterable[str],
            y: Any = None
    ) -> 'StyleFeaturizer':
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

        The empty string is encoded as the zero vector, by convention.

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

        # N.B. a hack to fix an issue in spaCy:
        # https://github.com/explosion/spaCy/issues/3456, where the
        # pipes can't handle empty documents. The empty string token has
        # 16 random ascii letters (upper and lower) and digits appended
        # to make collisions with actual documents very unlikely.
        #
        # Once issue 3456 is fixed in spaCy, these lines can be removed
        # and ``if doc.text == EMPTY_STRING_TOKEN:`` below can be
        # replaced with ``if doc.text == '':``.
        EMPTY_STRING_TOKEN = '<empty_string@jnk7pk8fsLyI6LI6>'
        X = [x if x != '' else EMPTY_STRING_TOKEN for x in X]

        style_features = []
        for doc in self._nlp.pipe(X, batch_size=64):
            if doc.text == EMPTY_STRING_TOKEN:
                # return the vector of all zeros for the empty string
                style_features.append(np.zeros(
                    10 + len(self._PUNCT_TO_IDX) + len(self._POS_TAG_TO_IDX)))
                continue

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

        return np.nan_to_num(np.array(style_features))


# the stylistic classifier baseline

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


# the length-based rankers

FewestWordsBaseline = Pipeline([
    ('classifier', LengthRanker(choose='shortest', length='words'))
])
"""Predict the answer with the fewest words."""

FEWEST_WORDS_HYPER_PARAMETERS = {}
"""The hyper-param search space for ``FewestWordsBaseline``."""

MostWordsBaseline = Pipeline([
    ('classifier', LengthRanker(choose='longest', length='words'))
])
"""Predict the answer with the most words."""

MOST_WORDS_HYPER_PARAMETERS = {}
"""The hyper-param search space for ``MostWordsBaseline``."""

FewestCharactersBaseline = Pipeline([
    ('classifier', LengthRanker(choose='shortest', length='characters'))
])
"""Predict the answer with the fewest characters."""

FEWEST_CHARACTERS_HYPER_PARAMETERS = {}
"""The hyper-param search space for ``FewestCharactersBaseline``."""

MostCharactersBaseline = Pipeline([
    ('classifier', LengthRanker(choose='longest', length='characters'))
])
"""Predict the answer with the most characters."""

MOST_CHARACTERS_HYPER_PARAMETERS = {}
"""The hyper-param search space for ``MostCharactersBaseline``."""


# stylistic linear ranker

StyleRankerBaseline = Pipeline([
    (
        'featurizer',
        utils.ResourceTransformer(
            transformer=Pipeline([
                (
                    'featurizer',
                    StyleFeaturizer()
                ),
                (
                    'scaler',
                    MaxAbsScaler()
                )
            ]))
    ),
    (
        'classifier',
        LogisticRegression(
            penalty='l2',
            dual=False,
            tol=1e-4,
            fit_intercept=False,
            intercept_scaling=1.,
            solver='lbfgs',
            max_iter=100,
            warm_start=True)
    )
])
"""Rank using a linear model on style features."""

STYLE_RANKER_HYPER_PARAMETERS = {
    'classifier__C': (1e-6, 1e2, 'log-uniform'),
    'classifier__class_weight': ['balanced', None],
}
"""The hyper-param search space for ``StyleRankerBaseline``."""

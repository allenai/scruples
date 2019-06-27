"""Tests for socialnorms.baselines.style."""

import unittest

import pandas as pd
import pytest
from sklearn  import metrics
from skopt import BayesSearchCV

from socialnorms.baselines import style


class StyleFeaturizerTestCase(unittest.TestCase):
    """Test socialnorms.baselines.style.StyleFeaturizer."""

    DOCS_AND_FEATURES = {
        'empty_string': (
            '',
            [
                # number of tokens in the document
                0.,
                # number of sentences in the document
                0.,
                # min sentence length in tokens
                0.,
                # max sentence length in tokens
                0.,
                # median sentence length in tokens
                0.,
                # average sentence length in tokens
                0.,
                # standard deviation of sentence length in tokens
                0.,
                # lexical diversity (type-token ratio) of the full
                # document
                0.,
                # average lexical diversity (type-token ratio) of
                # each sentence
                0.,
                # average word length (in characters), excluding
                # punctuation
                0.,
                # average punctuation counts per sentence
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                # average POS tag counts per sentence
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ]),
        'hello_world': (
            'Hello, world! And hello wonderful morning.',
            [
                # number of tokens in the document
                9.,
                # number of sentences in the document
                2.,
                # min sentence length in tokens
                4.,
                # max sentence length in tokens
                5.,
                # median sentence length in tokens
                4.5,
                # average sentence length in tokens
                4.5,
                # standard deviation of sentence length in tokens
                0.5,
                # lexical diversity (type-token ratio) of the full
                # document
                8. / 9.,
                # average lexical diversity (type-token ratio) of
                # each sentence
                1.,
                # average word length (in characters), excluding
                # punctuation
                34. / 6.,
                # average punctuation counts per sentence
                0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                # average POS tag counts per sentence
                0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5,
                0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ]),
        'no_punctuation': (
            'Interesting thought',
            [
                # number of tokens in the document
                2.,
                # number of sentences in the document
                1.,
                # min sentence length in tokens
                2.,
                # max sentence length in tokens
                2.,
                # median sentence length in tokens
                2.,
                # average sentence length in tokens
                2.,
                # standard deviation of sentence length in tokens
                0.,
                # lexical diversity (type-token ratio) of the full
                # document
                1.,
                # average lexical diversity (type-token ratio) of
                # each sentence
                1.,
                # average word length (in characters), excluding
                # punctuation
                9.,
                # average punctuation counts per sentence
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                # average POS tag counts per sentence
                0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ])
    }
    """A mapping from documents to their featurized representations."""

    def test_fit(self):
        transformer = style.StyleFeaturizer()

        # test that fit takes a ``y`` argument
        transformer.fit(X=['hello', 'world'], y=[0, 1])

    def test_transform(self):
        # test transform produces the correct feature values for a
        # document
        doc, gold_features = self.DOCS_AND_FEATURES['hello_world']

        X = [doc]
        y = [0]

        transformer = style.StyleFeaturizer()
        transformer.fit(X=X, y=y)

        features = transformer.transform(X).tolist()

        self.assertEqual(features, [gold_features])

    def test_transform_on_text_with_no_punctuation(self):
        # test transform on text that has no punctuation
        doc, gold_features = self.DOCS_AND_FEATURES['no_punctuation']

        X = [doc]
        y = [0]

        transformer = style.StyleFeaturizer()
        transformer.fit(X=X, y=y)

        features = transformer.transform(X).tolist()

        self.assertEqual(features, [gold_features])

    def test_transform_on_the_empty_string(self):
        # test transform produces correct output when the input is the
        # empty string
        doc, gold_features = self.DOCS_AND_FEATURES['empty_string']

        X = [doc]
        y = [0]

        transformer = style.StyleFeaturizer()
        transformer.fit(X=X, y=y)

        features = transformer.transform(X).tolist()

        self.assertEqual(features, [gold_features])

    def test_transform_multiple_docs(self):
        # test transform produces correct output when run on multiple
        # docs
        doc1, gold_features1 = self.DOCS_AND_FEATURES['hello_world']
        doc2, gold_features2 = self.DOCS_AND_FEATURES['no_punctuation']

        X = [doc1, doc2]
        y = [0, 0]

        transformer = style.StyleFeaturizer()
        transformer.fit(X=X, y=y)

        features = transformer.transform(X).tolist()

        self.assertEqual(features, [gold_features1, gold_features2])

    def test_transform_on_multiple_docs_with_empty_string(self):
        # test transform when it is passed multiple docs, some of which
        # might be the empty string

        # N.B. this test is a regression test for a bug found in spacy,
        # see: https://github.com/explosion/spaCy/issues/3456.
        doc1, gold_features1 = self.DOCS_AND_FEATURES['hello_world']
        doc2, gold_features2 = self.DOCS_AND_FEATURES['no_punctuation']
        empty_doc, empty_gold_features = self.DOCS_AND_FEATURES['empty_string']

        transformer = style.StyleFeaturizer()
        transformer.fit(X=[doc1], y=[0])

        # when empty string is first
        self.assertEqual(
            transformer.transform([empty_doc, doc1, doc2]).tolist(),
            [empty_gold_features, gold_features1, gold_features2])

        # when empty string is in the middle
        self.assertEqual(
            transformer.transform([doc1, empty_doc, doc2]).tolist(),
            [gold_features1, empty_gold_features, gold_features2])

        # when empty string is last
        self.assertEqual(
            transformer.transform([doc1, doc2, empty_doc]).tolist(),
            [gold_features1, gold_features2, empty_gold_features])


class StylisticXGBoostBaselineTestCase(unittest.TestCase):
    """Test the stylistic features baseline."""
    # The stylistic features aren't powerful enough to solve
    # corpus-easy (because the style is shared across the labels)
    # so we have to test this baseline differently from the others.

    def setUp(self):
        self.train_easy = pd.DataFrame({
            'title': 5 * [
                'Title',
                'Title',
                'Title',
                'Title',
                'Title',
                'Title'
            ],
            'text': 5 * [
                'This esteemed document utilizes elaborate words, and'
                ' demonstrates a tendency to continue sentences for'
                ' lengthy periods.',
                'This doc is short.',
                'Flowing and illustrious utterances trounce their'
                ' plebeian counterparts!',
                'Keep it brief.',
                'Hey!!?',
                'What?!!'
            ],
            'label': 5 * [
                'long',
                'short',
                'long',
                'short',
                'punctuation',
                'punctuation',
            ]
        })
        self.dev_easy = pd.DataFrame({
            'title': [
                'Title',
                'Title',
                'Title'
            ],
            'text': [
                'Short and sweet.',
                'Sentences should never in a million seasons conclude'
                ' before a multitude of words have been spoken.',
                '!?!'
            ],
            'label': [
                'short',
                'long',
                'punctuation'
            ]
        })

    @pytest.mark.slow
    def test_it_solves_the_easy_dataset_when_untuned(self):
        baseline = style.StylisticXGBoostBaseline
        baseline.fit(
            self.train_easy[['title', 'text']],
            self.train_easy['label'])
        predictions = baseline.predict(self.dev_easy[['title', 'text']])

        # check that the accuracy is 100%
        self.assertEqual(
            metrics.accuracy_score(
                y_true=self.dev_easy['label'],
                y_pred=predictions),
            1.)

    @pytest.mark.slow
    def test_it_solves_the_easy_dataset_when_tuned(self):
        baseline = BayesSearchCV(
            style.StylisticXGBoostBaseline,
            style.STYLISTICXGBOOST_HYPER_PARAMETERS,
            n_iter=16,
            n_points=2,
            cv=4,
            n_jobs=1)
        baseline.fit(
            self.train_easy[['title', 'text']],
            self.train_easy['label'])
        predictions = baseline.predict(self.dev_easy[['title', 'text']])

        # check that the accuracy is 100%
        self.assertEqual(
            metrics.accuracy_score(
                y_true=self.dev_easy['label'],
                y_pred=predictions),
            1.)

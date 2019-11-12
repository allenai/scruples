"""Tests for scruples.baselines.style."""

import unittest

import pandas as pd
import pytest
from sklearn  import metrics
from skopt import BayesSearchCV

from scruples.baselines import style


class LengthRankerTestCase(unittest.TestCase):
    """Test scruples.baselines.style.LengthRanker."""

    def test_fit(self):
        classifier = style.LengthRanker()

        # test that fit takes ``X`` and ``y`` arguments
        classifier.fit(X=[['foo', 'bar'], ['baz', 'quux']], y=[0, 1])

        # test that fit sets classes_ correctly
        self.assertEqual(classifier.classes_, [0, 1])

    def test_predict(self):
        # test fewest words
        #   regular case
        self.assertEqual(
            style.LengthRanker(
                choose='shortest', length='words'
            ).fit(
                [['word', 'word']], y=[0]
            ).predict(
                [['foo bar', 'baz'], ['foo', 'bar baz'], ['bar', 'foo baz']]
            ).tolist(),
            [1, 0, 0])
        #   where character lengths disagree with word lengths
        self.assertEqual(
            style.LengthRanker(
                choose='shortest', length='words'
            ).fit(
                [['word', 'word']], y=[0]
            ).predict(
                [['a b', 'cccc'], ['d', 'e f'], ['aaaaaa', 'b cc']]
            ).tolist(),
            [1, 0, 0])
        # test most words
        #   regular case
        self.assertEqual(
            style.LengthRanker(
                choose='longest', length='words'
            ).fit(
                [['word', 'word']], y=[0]
            ).predict(
                [['foo bar', 'baz'], ['foo', 'bar baz'], ['bar', 'foo baz']]
            ).tolist(),
            [0, 1, 1])
        #   where character lengths disagree with word lengths
        self.assertEqual(
            style.LengthRanker(
                choose='longest', length='words'
            ).fit(
                [['word', 'word']], y=[0]
            ).predict(
                [['a b', 'cccc'], ['d', 'e f'], ['aaaaaa', 'b cc']]
            ).tolist(),
            [0, 1, 1])
        # test fewest characters
        #   regular case
        self.assertEqual(
            style.LengthRanker(
                choose='shortest', length='characters'
            ).fit(
                [['word', 'word']], y=[0]
            ).predict(
                [['aaa', 'a'], ['b', 'bbb'], ['c', 'ccc']]
            ).tolist(),
            [1, 0, 0])
        #   where character lengths disagree with word lengths
        self.assertEqual(
            style.LengthRanker(
                choose='shortest', length='characters'
            ).fit(
                [['word', 'word']], y=[0]
            ).predict(
                [['a b', 'cccc'], ['d', 'e f'], ['aaaaaa', 'b cc']]
            ).tolist(),
            [0, 0, 1])
        # test most characters
        #   regular case
        self.assertEqual(
            style.LengthRanker(
                choose='longest', length='characters'
            ).fit(
                [['word', 'word']], y=[0]
            ).predict(
                [['aaa', 'a'], ['b', 'bbb'], ['c', 'ccc']]
            ).tolist(),
            [0, 1, 1])
        #   where character lengths disagree with word lengths
        self.assertEqual(
            style.LengthRanker(
                choose='longest', length='characters'
            ).fit(
                [['word', 'word']], y=[0]
            ).predict(
                [['a b', 'cccc'], ['d', 'e f'], ['aaaaaa', 'b cc']]
            ).tolist(),
            [1, 1, 0])

    def test_predict_with_empty_string_options(self):
        # test fewest words
        self.assertEqual(
            style.LengthRanker(
                choose='shortest', length='words'
            ).fit(
                [['word', 'word']], y=[0]
            ).predict(
                [['', 'aa'], ['aa', '']]
            ).tolist(),
            [0, 1])
        # test most words
        self.assertEqual(
            style.LengthRanker(
                choose='longest', length='words'
            ).fit(
                [['word', 'word']], y=[0]
            ).predict(
                [['', 'aa'], ['aa', '']]
            ).tolist(),
            [1, 0])
        # test fewest characters
        self.assertEqual(
            style.LengthRanker(
                choose='shortest', length='characters'
            ).fit(
                [['word', 'word']], y=[0]
            ).predict(
                [['', 'aa'], ['aa', '']]
            ).tolist(),
            [0, 1])
        # test most characters
        self.assertEqual(
            style.LengthRanker(
                choose='longest', length='characters'
            ).fit(
                [['word', 'word']], y=[0]
            ).predict(
                [['', 'aa'], ['aa', '']]
            ).tolist(),
            [1, 0])

    def test_predict_with_ties(self):
        predictions = style.LengthRanker(
            choose='shortest', length='characters'
        ).fit(
            [['word', 'word', 'word']], y=[0]
        ).predict([['', '', 'aaaa'], ['aaaa', 'aa', 'aa']]
        ).tolist()

        self.assertIn(predictions[0], [0, 1])
        self.assertIn(predictions[1], [1, 2])

    def test_predict_proba(self):
        # test fewest words
        #   regular case
        self.assertEqual(
            style.LengthRanker(
                choose='shortest', length='words'
            ).fit(
                [['word', 'word']], y=[0]
            ).predict_proba(
                [['foo bar', 'baz'], ['foo', 'bar baz'], ['bar', 'foo baz']]
            ).tolist(),
            [[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])
        #   where character lengths disagree with word lengths
        self.assertEqual(
            style.LengthRanker(
                choose='shortest', length='words'
            ).fit(
                [['word', 'word']], y=[0]
            ).predict_proba(
                [['a b', 'cccc'], ['d', 'e f'], ['aaaaaa', 'b cc']]
            ).tolist(),
            [[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])
        # test most words
        #   regular case
        self.assertEqual(
            style.LengthRanker(
                choose='longest', length='words'
            ).fit(
                [['word', 'word']], y=[0]
            ).predict_proba(
                [['foo bar', 'baz'], ['foo', 'bar baz'], ['bar', 'foo baz']]
            ).tolist(),
            [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        #   where character lengths disagree with word lengths
        self.assertEqual(
            style.LengthRanker(
                choose='longest', length='words'
            ).fit(
                [['word', 'word']], y=[0]
            ).predict_proba(
                [['a b', 'cccc'], ['d', 'e f'], ['aaaaaa', 'b cc']]
            ).tolist(),
            [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        # test fewest characters
        #   regular case
        self.assertEqual(
            style.LengthRanker(
                choose='shortest', length='characters'
            ).fit(
                [['word', 'word']], y=[0]
            ).predict_proba(
                [['aaa', 'a'], ['b', 'bbb'], ['c', 'ccc']]
            ).tolist(),
            [[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])
        #   where character lengths disagree with word lengths
        self.assertEqual(
            style.LengthRanker(
                choose='shortest', length='characters'
            ).fit(
                [['word', 'word']], y=[0]
            ).predict_proba(
                [['a b', 'cccc'], ['d', 'e f'], ['aaaaaa', 'b cc']]
            ).tolist(),
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        # test most characters
        #   regular case
        self.assertEqual(
            style.LengthRanker(
                choose='longest', length='characters'
            ).fit(
                [['word', 'word']], y=[0]
            ).predict_proba(
                [['aaa', 'a'], ['b', 'bbb'], ['c', 'ccc']]
            ).tolist(),
            [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        #   where character lengths disagree with word lengths
        self.assertEqual(
            style.LengthRanker(
                choose='longest', length='characters'
            ).fit(
                [['word', 'word']], y=[0]
            ).predict_proba(
                [['a b', 'cccc'], ['d', 'e f'], ['aaaaaa', 'b cc']]
            ).tolist(),
            [[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])


class StyleFeaturizerTestCase(unittest.TestCase):
    """Test scruples.baselines.style.StyleFeaturizer."""

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


class FewestWordsBaselineTestCase(unittest.TestCase):
    """Test scruples.baselines.style.FewestWordsBaseline."""

    FEATURES = [['', 'aa aa'], ['aa bb', 'cc']]
    LABELS = [0, 1]
    PROBABILITIES = [[1.0, 0.0], [0.0, 1.0]]

    def test_predicts_correctly(self):
        baseline = style.FewestWordsBaseline
        baseline.fit(X=self.FEATURES, y=[0 for _ in self.FEATURES])
        predictions = baseline.predict(self.FEATURES)

        self.assertEqual(predictions.tolist(), self.LABELS)

    def test_predicts_probabilities(self):
        baseline = style.FewestWordsBaseline
        baseline.fit(X=self.FEATURES, y=[0 for _ in self.FEATURES])
        probabilities = baseline.predict_proba(self.FEATURES)

        self.assertEqual(probabilities.tolist(), self.PROBABILITIES)


class MostWordsBaselineTestCase(unittest.TestCase):
    """Test scruples.baselines.style.MostWordsBaseline."""

    FEATURES = [['', 'aa aa'], ['aa bb', 'cc']]
    LABELS = [1, 0]
    PROBABILITIES = [[0.0, 1.0], [1.0, 0.0]]

    def test_predicts_correctly(self):
        baseline = style.MostWordsBaseline
        baseline.fit(X=self.FEATURES, y=[0 for _ in self.FEATURES])
        predictions = baseline.predict(self.FEATURES)

        self.assertEqual(predictions.tolist(), self.LABELS)

    def test_predicts_probabilities(self):
        baseline = style.MostWordsBaseline
        baseline.fit(X=self.FEATURES, y=[0 for _ in self.FEATURES])
        probabilities = baseline.predict_proba(self.FEATURES)

        self.assertEqual(probabilities.tolist(), self.PROBABILITIES)


class FewestCharactersBaselineTestCase(unittest.TestCase):
    """Test scruples.baselines.style.FewestCharactersBaseline."""

    FEATURES = [['', 'aa aa'], ['aa bb', 'cc']]
    LABELS = [0, 1]
    PROBABILITIES = [[1.0, 0.0], [0.0, 1.0]]

    def test_predicts_correctly(self):
        baseline = style.FewestCharactersBaseline
        baseline.fit(X=self.FEATURES, y=[0 for _ in self.FEATURES])
        predictions = baseline.predict(self.FEATURES)

        self.assertEqual(predictions.tolist(), self.LABELS)

    def test_predicts_probabilities(self):
        baseline = style.FewestCharactersBaseline
        baseline.fit(X=self.FEATURES, y=[0 for _ in self.FEATURES])
        probabilities = baseline.predict_proba(self.FEATURES)

        self.assertEqual(probabilities.tolist(), self.PROBABILITIES)


class MostCharactersBaselineTestCase(unittest.TestCase):
    """Test scruples.baselines.style.MostCharactersBaseline."""

    FEATURES = [['', 'aa aa'], ['aa bb', 'cc']]
    LABELS = [1, 0]
    PROBABILITIES = [[0.0, 1.0], [1.0, 0.0]]

    def test_predicts_correctly(self):
        baseline = style.MostCharactersBaseline
        baseline.fit(X=self.FEATURES, y=[0 for _ in self.FEATURES])
        predictions = baseline.predict(self.FEATURES)

        self.assertEqual(predictions.tolist(), self.LABELS)

    def test_predicts_probabilities(self):
        baseline = style.MostCharactersBaseline
        baseline.fit(X=self.FEATURES, y=[0 for _ in self.FEATURES])
        probabilities = baseline.predict_proba(self.FEATURES)

        self.assertEqual(probabilities.tolist(), self.PROBABILITIES)


class StyleRankerBaselineTestCase(unittest.TestCase):
    """Test the style ranking baseline."""
    # The stylistic features aren't powerful enough to solve
    # resource-easy (because the style is shared across the labels)
    # so we have to test this baseline differently from the others.

    def setUp(self):
        self.train_easy = pd.DataFrame({
            'action0': 5 * [
                "I'm short!",
                'I am a longer utterance than the short ones.',
                'Also short?!',
                '!!',
                'This sentence is longer than the short sentences.',
            ],
            'action1': 5 * [
                'This sentence is long long long long long.',
                'super short',
                'The long sentences represent lower ranked actions.',
                'A long sentence that has few punctuation marks other'
                ' than periods.',
                '??',
            ],
            'label': 5 * [0, 1, 0, 0, 1]
        })
        self.dev_easy = pd.DataFrame({
            'action0': [
                'Short and sweet.',
                'Sentences should never in a million seasons conclude'
                ' before a multitude of words have been spoken.',
                '!?!'
            ],
            'action1': [
                'This sentence is the kind of long sentence that gets'
                ' ranked lower',
                'Hm?',
                'Another long sentence for the style ranker development set.'
            ],
            'label': [0, 1, 0]
        })

    @pytest.mark.slow
    def test_it_solves_the_easy_dataset_when_untuned(self):
        baseline = style.StyleRankerBaseline
        baseline.fit(
            self.train_easy[['action0', 'action1']],
            self.train_easy['label'])
        predictions = baseline.predict(self.dev_easy[['action0', 'action1']])

        # check that the accuracy is 100%
        self.assertEqual(
            metrics.accuracy_score(
                y_true=self.dev_easy['label'],
                y_pred=predictions),
            1.)

    @pytest.mark.slow
    def test_it_solves_the_easy_dataset_when_tuned(self):
        baseline = BayesSearchCV(
            style.StyleRankerBaseline,
            style.STYLE_RANKER_HYPER_PARAMETERS,
            n_iter=16,
            n_points=2,
            cv=4,
            n_jobs=1)
        baseline.fit(
            self.train_easy[['action0', 'action1']],
            self.train_easy['label'])
        predictions = baseline.predict(self.dev_easy[['action0', 'action1']])

        # check that the accuracy is 100%
        self.assertEqual(
            metrics.accuracy_score(
                y_true=self.dev_easy['label'],
                y_pred=predictions),
            1.)

"""Tests for scruples.data.action."""

import json
import math
import pkg_resources
import unittest

import pytest

from scruples.data import action
from ... import settings


class GerundPhraseCaseTestCase(unittest.TestCase):
    """Test scruples.data.action.GerundPhraseCase."""

    def test_handles_gerund_phrases_correctly(self):
        # gerund phrase
        #   without negation
        self.assertEqual(
            action.GerundPhraseCase()('being the last one to a party'),
            ('being the last one to a party', True))
        #   with negation
        self.assertEqual(
            action.GerundPhraseCase()('not being the last one to a party'),
            ('not being the last one to a party', True))
        # "I'm" gerund phrase
        #   without negation
        self.assertEqual(
            action.GerundPhraseCase()("I'm running late all the time"),
            ('running late all the time', True))
        #   with negation
        self.assertEqual(
            action.GerundPhraseCase()("I'm not running late all the time"),
            ('not running late all the time', True))
        # "if I'm" gerund phrase
        #   without negation
        self.assertEqual(
            action.GerundPhraseCase()("if I'm running late all the time"),
            ('running late all the time', True))
        #   with negation
        self.assertEqual(
            action.GerundPhraseCase()("if I'm not running late all the time"),
            ('not running late all the time', True))

    def test_handles_words_with_dashes_correctly(self):
        self.assertEqual(
            action.GerundPhraseCase()('re-capturing the flag'),
            ('re-capturing the flag', True))
        self.assertEqual(
            action.GerundPhraseCase()('not re-capturing the flag'),
            ('not re-capturing the flag', True))
        self.assertEqual(
            action.GerundPhraseCase()("I'm re-capturing the flag"),
            ('re-capturing the flag', True))
        self.assertEqual(
            action.GerundPhraseCase()("I'm not re-capturing the flag"),
            ('not re-capturing the flag', True))
        self.assertEqual(
            action.GerundPhraseCase()("if I'm re-capturing the flag"),
            ('re-capturing the flag', True))
        self.assertEqual(
            action.GerundPhraseCase()('if I\'m not re-capturing the flag'),
            ('not re-capturing the flag', True))

    def test_does_not_match_non_gerund_phrases(self):
        self.assertEqual(
            action.GerundPhraseCase()('If I didn\'t say hello')[1],
            False)
        self.assertEqual(
            action.GerundPhraseCase()('got to leave')[1],
            False)


class PrepositionalPhraseCaseTestCase(unittest.TestCase):
    """Test scruples.data.action.PrepositionalPhraseCase."""

    def test_handles_prepositional_phrases_correctly(self):
        self.assertEqual(
            action.PrepositionalPhraseCase()('for running late'),
            ('running late', True))
        self.assertEqual(
            action.PrepositionalPhraseCase()('by running late'),
            ('running late', True))
        self.assertEqual(
            action.PrepositionalPhraseCase()('after running late'),
            ('running late', True))
        self.assertEqual(
            action.PrepositionalPhraseCase()('for not running late'),
            ('not running late', True))

    def test_does_not_match_non_prepositional_phrases(self):
        self.assertEqual(
            action.PrepositionalPhraseCase()('running late')[1],
            False)
        self.assertEqual(
            action.PrepositionalPhraseCase()('to run late')[1],
            False)


class IPhraseCaseTestCase(unittest.TestCase):
    """Test scruples.data.action.IPhraseCase."""

    def test_handles_I_phrases_correctly(self):
        self.assertEqual(
            action.IPhraseCase()('I ran late'),
            ('running late', True))
        self.assertEqual(
            action.IPhraseCase()('if I ran late'),
            ('running late', True))
        self.assertEqual(
            action.IPhraseCase()('cause I ran late'),
            ('running late', True))
        self.assertEqual(
            action.IPhraseCase()('when I ran late'),
            ('running late', True))

    def test_does_not_match_non_I_phrases(self):
        self.assertEqual(
            action.IPhraseCase()('running late')[1],
            False)
        self.assertEqual(
            action.IPhraseCase()('after running late')[1],
            False)
        self.assertEqual(
            action.IPhraseCase()('to run late')[1],
            False)


class InfinitivePhraseCase(unittest.TestCase):
    """Test scruples.InfinitivePhraseCase."""

    def test_handles_infinitives_correctly(self):
        self.assertEqual(
            action.InfinitivePhraseCase()('to run late'),
            ('running late', True))
        self.assertEqual(
            action.InfinitivePhraseCase()('to not run late'),
            ('not running late', True))

    def test_does_not_match_non_infinitive_phrases(self):
        self.assertEqual(
            action.InfinitivePhraseCase()('running late')[1],
            False)
        self.assertEqual(
            action.InfinitivePhraseCase()('after running late')[1],
            False)
        self.assertEqual(
            action.InfinitivePhraseCase()('I ran late')[1],
            False)


class ActionTestCase(unittest.TestCase):
    """Test scruples.data.action.Action."""

    # test computed properties

    def test_normativity(self):
        # when pronormative_score is 0
        self.assertEqual(
            action.Action(
                description='foo',
                pronormative_score=0,
                contranormative_score=5
            ).normativity,
            0.)
        # when contranormative_score is 0
        self.assertEqual(
            action.Action(
                description='foo',
                pronormative_score=5,
                contranormative_score=0
            ).normativity,
            1.)
        # when both pronormative_score and contranormative_score are
        # non-zero.
        self.assertEqual(
            action.Action(
                description='foo',
                pronormative_score=5,
                contranormative_score=5
            ).normativity,
            0.5)

    def test_is_good(self):
        # when both scores are positive
        action_ = action.Action(
            description='foo',
            pronormative_score=1,
            contranormative_score=1)
        self.assertTrue(action_.is_good)
        # when only the pronormative score is positive
        action_ = action.Action(
            description='foo',
            pronormative_score=1,
            contranormative_score=0)
        self.assertTrue(action_.is_good)
        # when only the contranormative_score is positive
        action_ = action.Action(
            description='foo',
            pronormative_score=0,
            contranormative_score=1)
        self.assertTrue(action_.is_good)
        # when both scores are zero
        action_ = action.Action(
            description='foo',
            pronormative_score=0,
            contranormative_score=0)
        self.assertFalse(action_.is_good)

    def test_normativity_when_divisor_is_zero(self):
        self.assertTrue(
            math.isnan(
                action.Action(
                    description='foo',
                    pronormative_score=0,
                    contranormative_score=0
                ).normativity))

    # test methods

    @pytest.mark.slow
    def test_extract_description_from_title(self):
        post_types = [
            'AITA',
            'Aita',
            'aita',
            'WIBTA',
            'Wibta',
            'wibta'
        ]
        spacers = ['', ' ', '  ']
        punctuations = ['', ':', '-', ',', '.', '?']
        negations = ['', ' not']
        prepositions = ['', ' for', ' by', ' after']
        I_phrase_starts = [
            ' if', ' cause', ' because', ' that', ' when', ' since'
        ]
        contents = [
            # I phrase / gerund phrase
            (' I said hello', ' saying hello'),
            (' I ran through a stop light', ' running through a stop light')
        ]
        sentence_ends = ['?', '.', '!']
        for post_type in post_types:
            for spacer1 in spacers:
                for punctuation in punctuations:
                    for spacer2 in spacers:
                        for preposition in prepositions:
                            for negation in negations:
                                for _, gerund in contents:
                                    for sentence_end in sentence_ends:
                                        title = (
                                            post_type
                                            + spacer1
                                            + punctuation
                                            + spacer2
                                            + preposition
                                            + negation
                                            + gerund
                                            + sentence_end)
                                        self.assertEqual(
                                            action.Action.extract_description_from_title(
                                                title),
                                            (negation + gerund).strip())

                        for I_phrase_start in I_phrase_starts:
                            for I_phrase, gerund in contents:
                                for sentence_end in sentence_ends:
                                    title = (
                                        post_type
                                        + spacer1
                                        + punctuation
                                        + spacer2
                                        + I_phrase_start
                                        + I_phrase
                                        + sentence_end)
                                    self.assertEqual(
                                        action.Action.extract_description_from_title(
                                            title),
                                        gerund.strip())

    def test_extract_description_from_title_on_gold_data(self):
        with pkg_resources.resource_stream(
                'tests', settings.GOLD_TITLE_DESCRIPTION_EXTRACTIONS_PATH
        ) as gold_title_description_extractions_file:
            gold_title_description_extractions = [
                json.loads(ln)
                for ln in gold_title_description_extractions_file
            ]

        for row in gold_title_description_extractions:
            if row['expected_failure']:
                continue

            # normalizing capitalization is not extremely important, so
            # only compare the lower-cased extracted and descriptions
            gold_description = row['description']
            if gold_description is not None:
                gold_description = gold_description.lower()
            extracted_description =\
                action.Action.extract_description_from_title(row['title'])
            if extracted_description is not None:
                extracted_description = extracted_description.lower()

            self.assertEqual(extracted_description, gold_description)

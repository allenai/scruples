"""Tests for socialnorms.extraction.transformers."""

import unittest

from socialnorms.extraction import transformers


class GerundifyingTransformerTestCase(unittest.TestCase):
    """Test socialnorms.extraction.transformers.GerundifyingTransformer."""

    def test_leaves_gerund_phrases_alone(self):
        f = transformers.GerundifyingTransformer()

        self.assertEqual(
            f('Running down the street'),
              'running down the street')
        self.assertEqual(
            f('Running down the street and waving goodbye to my friend.'),
              'running down the street and waving goodbye to my friend.')

    def test_transforms_infinitives(self):
        f = transformers.GerundifyingTransformer()

        # simple case
        self.assertEqual(
            f('to run through a stop light'),
              'running through a stop light')
        self.assertEqual(
            f('to wave to a friend.'),
              'waving to a friend.')
        # coordinated verbs
        self.assertEqual(
            f('to run through and totally ignore a stop light'),
              'running through and totally ignoring a stop light')
        self.assertEqual(
            f('to wave to a friend and say goodbye.'),
              'waving to a friend and saying goodbye.')

    def test_transforms_I_phrases(self):
        f = transformers.GerundifyingTransformer()

        # simple case
        self.assertEqual(
            f('I ran through a stop light'),
              'running through a stop light')
        self.assertEqual(
            f('I waved to a friend.'),
              'waving to a friend.')
        # coordinated verbs
        self.assertEqual(
            f('I ran through and totally ignored a stop light'),
              'running through and totally ignoring a stop light')
        self.assertEqual(
            f('I waved to a friend and said goodbye.'),
              'waving to a friend and saying goodbye.')

    def test_drops_auxiliary_verbs(self):
        f = transformers.GerundifyingTransformer()

        # will
        #   simple case
        self.assertEqual(
            f('I will run through a stop light'),
              'running through a stop light')
        self.assertEqual(
            f('I won\'t run through a stop light'),
              'not running through a stop light')
        self.assertEqual(
            f('I will wave to a friend.'),
              'waving to a friend.')
        self.assertEqual(
            f('I won\'t wave to a friend.'),
              'not waving to a friend.')
        #   coordinated verbs
        self.assertEqual(
            f('I will run through and totally ignore a stop light'),
              'running through and totally ignoring a stop light')
        self.assertEqual(
            f('I won\'t run through and totally ignore a stop light'),
              'not running through and totally ignoring a stop light')
        self.assertEqual(
            f('I will wave to a friend and say goodbye.'),
              'waving to a friend and saying goodbye.')
        self.assertEqual(
            f('I won\'t wave to a friend and say goodbye.'),
              'not waving to a friend and saying goodbye.')
        # do
        #   simple case
        self.assertEqual(
            f('I did run through a stop light'),
              'running through a stop light')
        self.assertEqual(
            f('I didn\'t run through a stop light'),
              'not running through a stop light')
        self.assertEqual(
            f('I did wave to a friend.'),
              'waving to a friend.')
        self.assertEqual(
            f('I didn\'t wave to a friend.'),
              'not waving to a friend.')
        #   coordinated verbs
        self.assertEqual(
            f('I did run through and totally ignore a stop light'),
              'running through and totally ignoring a stop light')
        self.assertEqual(
            f('I didn\'t run through and totally ignore a stop light'),
              'not running through and totally ignoring a stop light')
        self.assertEqual(
            f('I did wave to a friend and say goodbye.'),
              'waving to a friend and saying goodbye.')
        self.assertEqual(
            f('I didn\'t wave to a friend and say goodbye.'),
              'not waving to a friend and saying goodbye.')

    def test_handles_auxiliary_forms_of_to_be(self):
        f = transformers.GerundifyingTransformer()

        # to be when it is _not_ auxiliary
        self.assertEqual(
            f("I'm mad at my friend"),
              'being mad at my friend')
        self.assertEqual(
            f("I'm happy all the time."),
              'being happy all the time.')
        # to be when it is auxiliary
        self.assertEqual(
            f("I'm thinking about making some coffee"),
              'thinking about making some coffee')
        self.assertEqual(
            f("I'm running for president."),
              'running for president.')

    def test_handles_Im(self):
        f = transformers.GerundifyingTransformer()

        self.assertEqual(
            f('Im happy to see you'),
              'being happy to see you')
        self.assertEqual(
            f('Im sure it is.'),
              'being sure it is.')

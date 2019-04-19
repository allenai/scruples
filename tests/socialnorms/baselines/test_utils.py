"""Tests for socialnorms.baselines.utils."""

import unittest

import pandas as pd

from socialnorms.baselines import utils


class ConcatTitleAndTextTestCase(unittest.TestCase):
    """Test concat_title_and_text."""

    def test_concatenates_title_and_text_columns(self):
        features = pd.DataFrame({
            'title': ['The Title A', 'The Title B'],
            'text': ['The text A.', 'The text B.']
        })

        self.assertEqual(
            utils.concat_title_and_text(features).tolist(),
            ['The Title A\nThe text A.', 'The Title B\nThe text B.'])

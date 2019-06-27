"""Tests for socialnorms.dataset.transforms."""

import unittest
from unittest.mock import Mock

import pytest
from pytorch_pretrained_bert.tokenization import BertTokenizer

from socialnorms.dataset import transforms


class BertTransformTestCase(unittest.TestCase):
    """Test socialnorms.dataset.transforms.BertTransform."""

    @pytest.mark.slow
    def test_it_transforms_text_correctly(self):
        # test when only the first text is present
        transform = transforms.BertTransform(
            tokenizer=BertTokenizer.from_pretrained(
                'bert-base-uncased',
                do_lower_case=True),
            max_sequence_length=10)
        transformed = transform(('This sentence is for testing.', None))
        self.assertEqual(
            transformed['input_ids'],
            [  101, 2023, 6251, 2003, 2005, 5604, 1012,  102, 0, 0])
        self.assertEqual(
            transformed['input_mask'],
            [    1,    1,    1,    1,    1,    1,    1,    1, 0, 0])
        self.assertEqual(
            transformed['segment_ids'],
            [    0,    0,    0,    0,    0,    0,    0,    0, 0, 0])

        # test when both texts are present
        transform = transforms.BertTransform(
            tokenizer=BertTokenizer.from_pretrained(
                'bert-base-uncased',
                do_lower_case=True),
            max_sequence_length=18)
        transformed = transform((
            'This sentence is for testing.',
            'This sentence is also for testing.'
        ))
        self.assertEqual(
            transformed['input_ids'],
            [ 101, 2023, 6251, 2003, 2005, 5604,
             1012,  102, 2023, 6251, 2003, 2036,
             2005, 5604, 1012,  102,    0,    0])
        self.assertEqual(
            transformed['input_mask'],
            [   1,    1,    1,    1,    1,    1,
                1,    1,    1,    1,    1,    1,
                1,    1,    1,    1,    0,    0])
        self.assertEqual(
            transformed['segment_ids'],
            [   0,    0,    0,    0,    0,    0,
                0,    0,    1,    1,    1,    1,
                1,    1,    1,    1,    0,    0])

    @pytest.mark.slow
    def test_doesnt_unnecessarily_truncate_short_sequences(self):
        transform = transforms.BertTransform(
            tokenizer=BertTokenizer.from_pretrained(
                'bert-base-uncased',
                do_lower_case=True),
            max_sequence_length=16,
            truncation_strategy=('beginning', 'beginning'))
        # when the first text is more than half the max length
        transformed = transform((
            'A sentence that is more than 8 word pieces.',
            'Short.'
        ))
        self.assertEqual(
            transformed['input_ids'],
            [  101, 1037, 6251, 2008, 2003, 2062, 2084, 1022,
              2773, 4109, 1012,  102, 2460, 1012,  102,    0])
        self.assertEqual(
            transformed['input_mask'],
            [    1,    1,    1,    1,    1,    1,    1,    1,
                 1,    1,    1,    1,    1,    1,    1,    0])
        self.assertEqual(
            transformed['segment_ids'],
            [    0,    0,    0,    0,    0,    0,    0,    0,
                 0,    0,    0,    0,    1,    1,    1,    0])
        # when the second text is more than half the max length
        transformed = transform((
            'Short.',
            'A sentence that is more than 8 word pieces.'
        ))
        self.assertEqual(
            transformed['input_ids'],
            [  101, 2460, 1012,  102, 1037, 6251, 2008, 2003,
              2062, 2084, 1022, 2773, 4109, 1012,  102,    0])
        self.assertEqual(
            transformed['input_mask'],
            [    1,    1,    1,    1,    1,    1,    1,    1,
                 1,    1,    1,    1,    1,    1,    1,    0])
        self.assertEqual(
            transformed['segment_ids'],
            [    0,    0,    0,    0,    1,    1,    1,    1,
                 1,    1,    1,    1,    1,    1,    1,    0])
        # when both texts are less than half the max length
        transformed = transform((
            'Short.',
            'Also short.'
        ))
        self.assertEqual(
            transformed['input_ids'],
            [  101, 2460, 1012,  102, 2036, 2460, 1012,  102,
                 0,    0,    0,    0,    0,    0,    0,    0])
        self.assertEqual(
            transformed['input_mask'],
            [    1,    1,    1,    1,    1,    1,    1,    1,
                 0,    0,    0,    0,    0,    0,    0,    0])
        self.assertEqual(
            transformed['segment_ids'],
            [    0,    0,    0,    0,    1,    1,    1,    1,
                 0,    0,    0,    0,    0,    0,    0,    0])

    @pytest.mark.slow
    def test_truncates_the_longer_text_first(self):
        transform = transforms.BertTransform(
            tokenizer=BertTokenizer.from_pretrained(
                'bert-base-uncased',
                do_lower_case=True),
            max_sequence_length=12,
            truncation_strategy=('beginning', 'beginning'))
        # when the first text is longer
        transformed = transform((
            'A sentence that is more than 8 word pieces.',
            'Short.'
        ))
        self.assertEqual(
            transformed['input_ids'],
            [  101, 1037, 6251, 2008, 2003, 2062,
              2084, 1022,  102, 2460, 1012,  102])
        self.assertEqual(
            transformed['input_mask'],
            [    1,    1,    1,    1,    1,    1,
                 1,    1,    1,    1,    1,    1])
        self.assertEqual(
            transformed['segment_ids'],
            [    0,    0,    0,    0,    0,    0,
                 0,    0,    0,    1,    1,    1])
        # when the second text is longer
        transformed = transform((
            'Short.',
            'A sentence that is more than 8 word pieces.'
        ))
        self.assertEqual(
            transformed['input_ids'],
            [  101, 2460, 1012,  102, 1037, 6251,
              2008, 2003, 2062, 2084, 1022,  102])
        self.assertEqual(
            transformed['input_mask'],
            [    1,    1,    1,    1,    1,    1,
                 1,    1,    1,    1,    1,    1])
        self.assertEqual(
            transformed['segment_ids'],
            [    0,    0,    0,    0,    1,    1,
                 1,    1,    1,    1,    1,    1])

    @pytest.mark.slow
    def test_truncates_texts_to_half_max_length_when_lots_of_text(self):
        transform = transforms.BertTransform(
            tokenizer=BertTokenizer.from_pretrained(
                'bert-base-uncased',
                do_lower_case=True),
            max_sequence_length=11,
            truncation_strategy=('beginning', 'beginning'))
        transformed = transform((
            'This sentence is a test.',
            'A sentence that is more than 8 word pieces.'
        ))
        self.assertEqual(
            transformed['input_ids'],
            [  101, 2023, 6251, 2003, 1037,
               102, 1037, 6251, 2008, 2003,
               102])
        self.assertEqual(
            transformed['input_mask'],
            [    1,    1,    1,    1,    1,
                 1,    1,    1,    1,    1,
                 1])
        self.assertEqual(
            transformed['segment_ids'],
            [    0,    0,    0,    0,    0,
                 0,    1,    1,    1,    1,
                 1])

    @pytest.mark.slow
    def test_it_uses_the_correct_truncation_strategies(self):
        # test beginning strategy for first text
        transform = transforms.BertTransform(
            tokenizer=BertTokenizer.from_pretrained(
                'bert-base-uncased',
                do_lower_case=True),
            max_sequence_length=11,
            truncation_strategy=('beginning', 'beginning'))
        transformed = transform((
            'A sentence that is more than 8 word pieces.',
            'Short.'
        ))
        self.assertEqual(
            transformed['input_ids'],
            [  101, 1037, 6251, 2008, 2003,
              2062, 2084,  102, 2460, 1012,
               102])
        self.assertEqual(
            transformed['input_mask'],
            [    1,    1,    1,    1,    1,
                 1,    1,    1,    1,    1,
                 1])
        self.assertEqual(
            transformed['segment_ids'],
            [    0,    0,    0,    0,    0,
                 0,    0,    0,    1,    1,
                 1])

        # test ending strategy for first text
        transform = transforms.BertTransform(
            tokenizer=BertTokenizer.from_pretrained(
                'bert-base-uncased',
                do_lower_case=True),
            max_sequence_length=11,
            truncation_strategy=('ending', 'beginning'))
        transformed = transform((
            'A sentence that is more than 8 word pieces.',
            'Short.'
        ))
        self.assertEqual(
            transformed['input_ids'],
            [  101, 2062, 2084, 1022, 2773,
              4109, 1012,  102, 2460, 1012,
               102])
        self.assertEqual(
            transformed['input_mask'],
            [    1,    1,    1,    1,    1,
                 1,    1,    1,    1,    1,
                 1])
        self.assertEqual(
            transformed['segment_ids'],
            [    0,    0,    0,    0,    0,
                 0,    0,    0,    1,    1,
                 1])

        # test beginning strategy for second text
        transform = transforms.BertTransform(
            tokenizer=BertTokenizer.from_pretrained(
                'bert-base-uncased',
                do_lower_case=True),
            max_sequence_length=11,
            truncation_strategy=('beginning', 'beginning'))
        transformed = transform((
            'Short.',
            'A sentence that is more than 8 word pieces.'
        ))
        self.assertEqual(
            transformed['input_ids'],
            [  101, 2460, 1012,  102, 1037,
              6251, 2008, 2003, 2062, 2084,
               102])
        self.assertEqual(
            transformed['input_mask'],
            [    1,    1,    1,    1,    1,
                 1,    1,    1,    1,    1,
                 1])
        self.assertEqual(
            transformed['segment_ids'],
            [    0,    0,    0,    0,    1,
                 1,    1,    1,    1,    1,
                 1])

        # test ending strategy for second text
        transform = transforms.BertTransform(
            tokenizer=BertTokenizer.from_pretrained(
                'bert-base-uncased',
                do_lower_case=True),
            max_sequence_length=11,
            truncation_strategy=('beginning', 'ending'))
        transformed = transform((
            'Short.',
            'A sentence that is more than 8 word pieces.'
        ))
        self.assertEqual(
            transformed['input_ids'],
            [  101, 2460, 1012,  102, 2062,
               2084, 1022, 2773, 4109, 1012,
               102])
        self.assertEqual(
            transformed['input_mask'],
            [    1,    1,    1,    1,    1,
                 1,    1,    1,    1,    1,
                 1])
        self.assertEqual(
            transformed['segment_ids'],
            [    0,    0,    0,    0,    1,
                 1,    1,    1,    1,    1,
                 1])


class ComposeTestCase(unittest.TestCase):
    """Test socialnorms.dataset.transforms.Compose."""

    def test_empty_list_is_identity(self):
        transform = transforms.Compose([])

        self.assertEqual(transform(1), 1)
        self.assertEqual(transform('a'), 'a')

    def test_composes_mocked_functions(self):
        mock1 = Mock(return_value='bar')
        mock2 = Mock(return_value='baz')

        transform = transforms.Compose([mock1, mock2])

        self.assertEqual(transform('foo'), 'baz')

        mock1.assert_called_with('foo')
        mock2.assert_called_with('bar')

    def test_composes_actual_functions(self):
        transform = transforms.Compose([
            lambda x: x**2,
            lambda x: x+1
        ])

        self.assertEqual(transform(-2), 5)

        transform = transforms.Compose([
            lambda s: s.lower(),
            lambda s: s + 'b'
        ])

        self.assertEqual(transform('A'), 'ab')


class MapTestCase(unittest.TestCase):
    """Test socialnorms.dataset.transforms.Map."""

    def test_mapping_the_empty_list(self):
        transform = transforms.Map(lambda x: x**2)

        self.assertEqual(transform([]), [])

    def test_it_maps_a_mocked_function(self):
        mock = Mock(return_value='foo')

        transform = transforms.Map(mock)

        self.assertEqual(
            transform([1, 2, 3]),
            ['foo', 'foo', 'foo'])

        mock.assert_any_call(1)
        mock.assert_any_call(2)
        mock.assert_any_call(3)

        self.assertEqual(mock.call_count, 3)

    def test_it_maps_an_actual_function(self):
        transform = transforms.Map(transform=lambda x: x**2)

        self.assertEqual(transform([1, 2, 3]), [1, 4, 9])

        transform = transforms.Map(transform=str.lower)

        self.assertEqual(transform(['Aa', 'BB', 'cC']), ['aa', 'bb', 'cc'])

    def test_mapping_different_sequence_types(self):
        transform = transforms.Map(transform=lambda x: x**2)

        self.assertEqual(transform((1, 2, 3)), [1, 4, 9])

"""Tests for scruples.demos.norms.utils."""

import unittest
from unittest import mock

from scruples.demos.norms import utils


class PredictionDatasetTestCase(unittest.TestCase):
    """Test scruples.demos.norms.utils.PredictionDataset."""

    def setUp(self):
        self.features = [[1, 2], [1, 0]]
        self.transform = lambda x: x

        self.dataset = utils.PredictionDataset(
            features=self.features,
            transform=self.transform)

    def test___init__(self):
        self.assertEqual(self.dataset.features, self.features)
        self.assertEqual(self.dataset.transform, self.transform)

    def test___len__(self):
        self.assertEqual(len(self.dataset), len(self.features))

    def test___get_item__(self):
        for i in range(len(self.features)):
            self.assertEqual(self.dataset[i], self.features[i])

    def test_it_applies_transform(self):
        features = self.features
        mock_transform = mock.MagicMock(return_value='foo')

        dataset = utils.PredictionDataset(
            features=features,
            transform=mock_transform)

        self.assertEqual(dataset[1], 'foo')
        mock_transform.assert_called_with(features[1])

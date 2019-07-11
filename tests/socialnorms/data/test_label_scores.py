"""Tests for socialnorms.data.label_scores."""

import unittest

from socialnorms.data import label_scores
from socialnorms.data import labels


class LabelScoresTestCase(unittest.TestCase):
    """Test socialnorms.data.label_scores.LabelScores."""

    def test_best_label(self):
        # test that the best label is the one with the highest score
        # when the best label is OTHER
        self.assertEqual(
            label_scores.LabelScores(
                label_to_score={
                    labels.Label.OTHER: 10,
                    labels.Label.AUTHOR: 2,
                    labels.Label.EVERYBODY: 1,
                    labels.Label.NOBODY: 9,
                    labels.Label.INFO: 3
                }).best_label,
            labels.Label.OTHER)
        # when the best label is AUTHOR
        self.assertEqual(
            label_scores.LabelScores(
                label_to_score={
                    labels.Label.OTHER: 0,
                    labels.Label.AUTHOR: 10,
                    labels.Label.EVERYBODY: 0,
                    labels.Label.NOBODY: 0,
                    labels.Label.INFO: 0
                }).best_label,
            labels.Label.AUTHOR)
        # when the best label is EVERYBODY
        self.assertEqual(
            label_scores.LabelScores(
                label_to_score={
                    labels.Label.OTHER: 1,
                    labels.Label.AUTHOR: 1,
                    labels.Label.EVERYBODY: 3,
                    labels.Label.NOBODY: 2,
                    labels.Label.INFO: 1
                }).best_label,
            labels.Label.EVERYBODY)
        # when the best label is NOBODY
        self.assertEqual(
            label_scores.LabelScores(
                label_to_score={
                    labels.Label.OTHER: 0,
                    labels.Label.AUTHOR: 0,
                    labels.Label.EVERYBODY: 0,
                    labels.Label.NOBODY: 6,
                    labels.Label.INFO: 0
                }).best_label,
            labels.Label.NOBODY)
        # when the best label is INFO
        self.assertEqual(
            label_scores.LabelScores(
                label_to_score={
                    labels.Label.OTHER: 0,
                    labels.Label.AUTHOR: 5,
                    labels.Label.EVERYBODY: 3,
                    labels.Label.NOBODY: 0,
                    labels.Label.INFO: 23
                }).best_label,
            labels.Label.INFO)

    def test_is_all_zero(self):
        # test when is_all_zero should be true
        self.assertEqual(
            label_scores.LabelScores(
                label_to_score={
                    label: 0
                    for label in labels.Label
                }).is_all_zero,
            True)

        # test when is_all_zero should be false
        self.assertEqual(
            label_scores.LabelScores(
                label_to_score={
                    labels.Label.OTHER: 1,
                    labels.Label.AUTHOR: 0,
                    labels.Label.EVERYBODY: 0,
                    labels.Label.NOBODY: 0,
                    labels.Label.INFO: 0
                }).is_all_zero,
            False)

    def test_has_unique_highest_scoring_label(self):
        # test when has_unique_highest_scoring_label should be true
        self.assertEqual(
            label_scores.LabelScores(
                label_to_score={
                    label: i
                    for i, label in enumerate(labels.Label)
                }).has_unique_highest_scoring_label,
            True)

        # test when has_unique_highest_scoring_label should be false
        self.assertEqual(
            label_scores.LabelScores(
                label_to_score={
                    label: 10
                    for label in labels.Label
                }).has_unique_highest_scoring_label,
            False)

    def test_is_good(self):
        # test when is_good should be true
        self.assertEqual(
            label_scores.LabelScores(
                label_to_score={
                    label: i
                    for i, label in enumerate(labels.Label)
                }).is_good,
            True)

        # test when is_good should be false
        # when is_all_zero is true
        self.assertEqual(
            label_scores.LabelScores(
                label_to_score={
                    label: 0
                    for label in labels.Label
                }).is_good,
            False)
        # when has_unique_highest_scoring_label is false
        self.assertEqual(
            label_scores.LabelScores(
                label_to_score={
                    label: 10
                    for label in labels.Label
                }).is_good,
            False)

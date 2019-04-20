"""Tests for socialnorms.data.label_scores."""

import unittest

from socialnorms.data import label_scores
from socialnorms.data import labels


class LabelScoresTestCase(unittest.TestCase):
    """Test socialnorms.data.label_scores.LabelScores."""

    def test_best_label(self):
        # test that the best label is the one with the highest score
        # when the best label is NTA
        self.assertEqual(
            label_scores.LabelScores(
                label_to_score={
                    labels.Label.NTA: 10,
                    labels.Label.YTA: 2,
                    labels.Label.ESH: 1,
                    labels.Label.NAH: 9,
                    labels.Label.INFO: 3
                }).best_label,
            labels.Label.NTA)
        # when the best label is YTA
        self.assertEqual(
            label_scores.LabelScores(
                label_to_score={
                    labels.Label.NTA: 0,
                    labels.Label.YTA: 10,
                    labels.Label.ESH: 0,
                    labels.Label.NAH: 0,
                    labels.Label.INFO: 0
                }).best_label,
            labels.Label.YTA)
        # when the best label is ESH
        self.assertEqual(
            label_scores.LabelScores(
                label_to_score={
                    labels.Label.NTA: 1,
                    labels.Label.YTA: 1,
                    labels.Label.ESH: 3,
                    labels.Label.NAH: 2,
                    labels.Label.INFO: 1
                }).best_label,
            labels.Label.ESH)
        # when the best label is NAH
        self.assertEqual(
            label_scores.LabelScores(
                label_to_score={
                    labels.Label.NTA: 0,
                    labels.Label.YTA: 0,
                    labels.Label.ESH: 0,
                    labels.Label.NAH: 6,
                    labels.Label.INFO: 0
                }).best_label,
            labels.Label.NAH)
        # when the best label is INFO
        self.assertEqual(
            label_scores.LabelScores(
                label_to_score={
                    labels.Label.NTA: 0,
                    labels.Label.YTA: 5,
                    labels.Label.ESH: 3,
                    labels.Label.NAH: 0,
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
                    labels.Label.NTA: 1,
                    labels.Label.YTA: 0,
                    labels.Label.ESH: 0,
                    labels.Label.NAH: 0,
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

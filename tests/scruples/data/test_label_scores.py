"""Tests for scruples.data.label_scores."""

import unittest

from scruples.data import label_scores
from scruples.data import labels


class LabelScoresTestCase(unittest.TestCase):
    """Test scruples.data.label_scores.LabelScores."""

    def test_binarized_label_to_score(self):
        # test binarized_label_to_score on a typical input
        self.assertEqual(
            label_scores.LabelScores(
                label_to_score={
                    labels.Label.OTHER: 10,
                    labels.Label.AUTHOR: 2,
                    labels.Label.EVERYBODY: 1,
                    labels.Label.NOBODY: 9,
                    labels.Label.INFO: 3
                }).binarized_label_to_score,
            {
                labels.BinarizedLabel.RIGHT: 19,
                labels.BinarizedLabel.WRONG: 3
            })
        # test binarized_label_to_score on only INFO labels
        self.assertEqual(
            label_scores.LabelScores(
                label_to_score={
                    labels.Label.OTHER: 0,
                    labels.Label.AUTHOR: 0,
                    labels.Label.EVERYBODY: 0,
                    labels.Label.NOBODY: 0,
                    labels.Label.INFO: 10
                }).binarized_label_to_score,
            {
                labels.BinarizedLabel.RIGHT: 0,
                labels.BinarizedLabel.WRONG: 0
            })

    def test_best_binarized_label(self):
        # test that the best_binarized_label is the one with the highest
        # score when the best label is RIGHT
        self.assertEqual(
            label_scores.LabelScores(
                label_to_score={
                    labels.Label.OTHER: 10,
                    labels.Label.AUTHOR: 2,
                    labels.Label.EVERYBODY: 1,
                    labels.Label.NOBODY: 9,
                    labels.Label.INFO: 3
                }).best_binarized_label,
            labels.BinarizedLabel.RIGHT)
        # when the best_binarized_label is WRONG
        self.assertEqual(
            label_scores.LabelScores(
                label_to_score={
                    labels.Label.OTHER: 2,
                    labels.Label.AUTHOR: 10,
                    labels.Label.EVERYBODY: 9,
                    labels.Label.NOBODY: 1,
                    labels.Label.INFO: 3
                }).best_binarized_label,
            labels.BinarizedLabel.WRONG)

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

    def test_has_all_zero_binarized_label_scores(self):
        # test when has_all_zero_binarized_label_scores should be true
        self.assertEqual(
            label_scores.LabelScores(
                label_to_score={
                    label: 0
                    for label in labels.Label
                }).has_all_zero_binarized_label_scores,
            True)
        self.assertEqual(
            label_scores.LabelScores(
                label_to_score={
                    label: 0 if label != labels.Label.INFO else 10
                    for label in labels.Label
                }).has_all_zero_binarized_label_scores,
            True)

        # test when has_all_zero_binarized_label_scores should be false
        self.assertEqual(
            label_scores.LabelScores(
                label_to_score={
                    labels.Label.OTHER: 1,
                    labels.Label.AUTHOR: 0,
                    labels.Label.EVERYBODY: 0,
                    labels.Label.NOBODY: 0,
                    labels.Label.INFO: 0
                }).has_all_zero_binarized_label_scores,
            False)

    def test_has_all_zero_label_scores(self):
        # test when has_all_zero_label_scores should be true
        self.assertEqual(
            label_scores.LabelScores(
                label_to_score={
                    label: 0
                    for label in labels.Label
                }).has_all_zero_label_scores,
            True)

        # test when has_all_zero_label_scores should be false
        self.assertEqual(
            label_scores.LabelScores(
                label_to_score={
                    labels.Label.OTHER: 1,
                    labels.Label.AUTHOR: 0,
                    labels.Label.EVERYBODY: 0,
                    labels.Label.NOBODY: 0,
                    labels.Label.INFO: 0
                }).has_all_zero_label_scores,
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
        # when has_all_zero_binarized_label_scores is true
        self.assertEqual(
            label_scores.LabelScores(
                label_to_score={
                    label: 0 if label != labels.Label.INFO else 10
                    for label in labels.Label
                }).is_good,
            False)
        # when has_all_zero_label_scores is true
        self.assertEqual(
            label_scores.LabelScores(
                label_to_score={
                    label: 0
                    for label in labels.Label
                }).is_good,
            False)

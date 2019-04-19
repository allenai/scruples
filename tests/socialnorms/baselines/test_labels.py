"""Tests for socialnorms.baselines.labels."""

import unittest

import numpy as np

from socialnorms.baselines import labels


class PriorBaselineTestCase(unittest.TestCase):
    """Test socialnorms.baselines.labels.PriorBaseline."""

    FEATURES = [
        [0, 1],
        [0, 1],
        [1, 0]
    ]
    LABELS = ['a', 'a', 'b']

    def test_predicts_most_frequent_label(self):
        baseline = labels.PriorBaseline
        baseline.fit(X=self.FEATURES, y=self.LABELS)
        predictions = baseline.predict(self.FEATURES)

        self.assertEqual(predictions.tolist(), ['a', 'a', 'a'])

    def test_predicts_probabilities_by_class_prior(self):
        baseline = labels.PriorBaseline
        baseline.fit(X=self.FEATURES, y=self.LABELS)
        probabilities = baseline.predict_proba(self.FEATURES)

        self.assertAlmostEqual(
            probabilities.tolist(),
            [
                [2./3., 1./3.],
                [2./3., 1./3.],
                [2./3., 1./3.]
            ])


class StratifiedBaselineTestCase(unittest.TestCase):
    """Test socialnorms.baselines.labels.StratifiedBaseline."""

    FEATURES = [
        [0, 1],
        [0, 1],
        [1, 0]
    ]
    LABELS = ['a', 'a', 'b']

    # the number of trials to perform in statistically testing that the
    # predictions adhere to the label frequencies from the training data
    N_TRIALS = 10000

    def test_predicts_random_label_by_class_probability(self):
        baseline = labels.StratifiedBaseline
        baseline.fit(X=self.FEATURES, y=self.LABELS)
        predictions = baseline.predict([[0, 0] for _ in range(self.N_TRIALS)])

        elements, counts = np.unique(predictions, return_counts=True)

        # test that all labels are predicted
        self.assertEqual(elements.tolist(), ['a', 'b'])

        # check that the mean counts for each label are within 5
        # standard deviations of their expectations
        std = ((2./3. * 1./3.) / self.N_TRIALS)**0.5
        a_mean = counts[0] / self.N_TRIALS
        b_mean = counts[1] / self.N_TRIALS
        self.assertGreater(a_mean, 2./3. - 5 * std)
        self.assertLess(a_mean, 2./3. + 5 * std)
        self.assertGreater(b_mean, 1./3. - 5 * std)
        self.assertLess(b_mean, 1./3. + 5 * std)

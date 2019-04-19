"""Tests for socialnorms.baselines.linear."""

import unittest

from socialnorms.baselines import linear
from .utils import BaselineTestMixin


class LogististicRegressionBaselineTestCase(
        BaselineTestMixin,
        unittest.TestCase
):
    """Test the logistic regression on bag-of-ngrams baseline."""

    BASELINE_MODEL = linear.LogisticRegressionBaseline
    BASELINE_HYPER_PARAMETERS = linear.LOGISTIC_REGRESSION_HYPER_PARAMETERS

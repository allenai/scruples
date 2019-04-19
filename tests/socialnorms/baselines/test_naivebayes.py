"""Tests for socialnorms.baselines.naivebayes."""

import unittest

from socialnorms.baselines import naivebayes
from .utils import BaselineTestMixin


class BernoulliNBBaselineTestCase(
        BaselineTestMixin,
        unittest.TestCase
):
    """Test the bernoulli naive bayes on bag-of-ngrams baseline."""

    BASELINE_MODEL = naivebayes.BernoulliNBBaseline
    BASELINE_HYPER_PARAMETERS = naivebayes.BERNOULLINB_HYPER_PARAMETERS


class MultinomialNBBaselineTestCase(
        BaselineTestMixin,
        unittest.TestCase
):
    """Test the multinomial naive bayes on bag-of-ngrams baseline."""

    BASELINE_MODEL = naivebayes.MultinomialNBBaseline
    BASELINE_HYPER_PARAMETERS = naivebayes.MULTINOMIALNB_HYPER_PARAMETERS


class ComplementNBBaselineTestCase(
        BaselineTestMixin,
        unittest.TestCase
):
    """Test the complement naive bayes on bag-of-ngrams baseline."""

    BASELINE_MODEL = naivebayes.ComplementNBBaseline
    BASELINE_HYPER_PARAMETERS = naivebayes.COMPLEMENTNB_HYPER_PARAMETERS

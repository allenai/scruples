"""Utilities for testing baselines."""

import pkg_resources

import pandas as pd
import pytest
from sklearn import metrics
from skopt import BayesSearchCV

from ... import settings


# classes

class BaselineTestMixin:
    """Test a baseline model by verifying it solves an easy dataset.

    Attributes
    ----------
    BASELINE_MODEL : sklearn.base.BaseEstimator
        The baseline model to test.
    BASELINE_HYPER_PARAMETERS : Dict
        The dictionary defining the hyper-parameter search space for the
        baseline model.

    Examples
    --------
    To create a test case for a baseline model, inherit from this class
    along with ``unittest.TestCase`` and provide the ``BASELINE_MODEL``
    and ``BASELINE_HYPER_PARAMETERS`` class attributes::

        class LogisticRegressionBaselineTestCase(
                BaselineTestMixin,
                unittest.TestCase
        ):
            '''Test the logistic regression baseline.'''

            BASELINE_MODEL = LogisticRegressionBaseline
            BASELINE_HYPER_PARAMETERS = LOGISTIC_REGRESSION_HYPER_PARAMS

    """

    BASELINE_MODEL = None
    BASELINE_HYPER_PARAMETERS = None

    def setUp(self):
        super().setUp()

        if self.BASELINE_MODEL is None:
            raise ValueError(
                'Subclasses of BaselineTestMixin must provide a'
                ' BASELINE_MODEL class attribute.')

        if self.BASELINE_HYPER_PARAMETERS is None:
            raise ValueError(
                'Subclasses of BaselineTestMixin must provide a'
                ' BASELINE_HYPER_PARAMETERS class attribute.')

        self.train_easy = pd.read_json(
            pkg_resources.resource_stream(
                'tests', settings.SOCIALNORMS_EASY_TRAIN_PATH),
            lines=True)
        self.dev_easy = pd.read_json(
            pkg_resources.resource_stream(
                'tests', settings.SOCIALNORMS_EASY_DEV_PATH),
            lines=True)

    @pytest.mark.slow
    def test_it_solves_socialnorms_easy_when_untuned(self):
        baseline = self.BASELINE_MODEL
        baseline.fit(
            self.train_easy[['title', 'text']],
            self.train_easy['label'])
        predictions = baseline.predict(self.dev_easy[['title', 'text']])

        # check that the accuracy is 100%
        self.assertEqual(
            metrics.accuracy_score(
                y_true=self.dev_easy['label'],
                y_pred=predictions),
            1.)

    @pytest.mark.slow
    def test_it_solves_socialnorms_easy_when_tuned(self):
        baseline = BayesSearchCV(
            self.BASELINE_MODEL,
            self.BASELINE_HYPER_PARAMETERS,
            n_iter=16,
            n_points=2,
            cv=4,
            n_jobs=1)
        baseline.fit(
            self.train_easy[['title', 'text']],
            self.train_easy['label'])
        predictions = baseline.predict(self.dev_easy[['title', 'text']])

        # check that the accuracy is 100%
        self.assertEqual(
            metrics.accuracy_score(
                y_true=self.dev_easy['label'],
                y_pred=predictions),
            1.)

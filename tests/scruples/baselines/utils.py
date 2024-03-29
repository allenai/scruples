"""Utilities for testing baselines."""

import os
import tempfile

import pandas as pd
import pytest
from sklearn import metrics
from skopt import BayesSearchCV

from scruples import settings as scruples_settings
from scruples.dataset import readers
from ... import settings, utils


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
    DATASET : str
        The dataset against which the baseline should be run. Must be
        either ``"resource"`` or ``"corpus"``.

    Examples
    --------
    To create a test case for a baseline model, inherit from this class
    along with ``unittest.TestCase`` and provide the ``BASELINE_MODEL``
    ``BASELINE_HYPER_PARAMETERS``, and ``DATASET`` class attributes::

        class LogisticRegressionBaselineTestCase(
                BaselineTestMixin,
                unittest.TestCase
        ):
            '''Test the logistic regression baseline.'''

            BASELINE_MODEL = LogisticRegressionBaseline
            BASELINE_HYPER_PARAMETERS = LOGISTIC_REGRESSION_HYPER_PARAMS
            DATASET = 'corpus'

    """

    BASELINE_MODEL = None
    BASELINE_HYPER_PARAMETERS = None
    DATASET = None

    def setUp(self):
        super().setUp()

        # validate the class

        if self.BASELINE_MODEL is None:
            raise ValueError(
                'Subclasses of BaselineTestMixin must provide a'
                ' BASELINE_MODEL class attribute.')

        if self.BASELINE_HYPER_PARAMETERS is None:
            raise ValueError(
                'Subclasses of BaselineTestMixin must provide a'
                ' BASELINE_HYPER_PARAMETERS class attribute.')

        if self.DATASET is None:
            raise ValueError(
                'Subclasses of BaselineTestMixin must provide a DATASET'
                ' class attribute.')

        if self.DATASET not in ['resource', 'corpus']:
            raise ValueError(
                'The DATASET class attribute must either be'
                ' "resource", or "corpus".')

        # copy the dataset fixture from the package to disk

        if self.DATASET == 'resource':
            Reader = readers.ScruplesResource
            fixture_path = settings.RESOURCE_EASY_DIR
            split_filename_template =\
                scruples_settings.RESOURCE_FILENAME_TEMPLATE
        elif self.DATASET == 'corpus':
            Reader = readers.ScruplesCorpus
            fixture_path = settings.CORPUS_EASY_DIR
            split_filename_template =\
                scruples_settings.CORPUS_FILENAME_TEMPLATE

        self.temp_dir = tempfile.TemporaryDirectory()

        for split in Reader.SPLITS:
            split_filename = split_filename_template.format(split=split)
            utils.copy_pkg_resource_to_disk(
                pkg='tests',
                src=os.path.join(fixture_path, split_filename),
                dst=os.path.join(self.temp_dir.name, split_filename))

        # load the dataset

        self.dataset = Reader(data_dir=self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    @pytest.mark.slow
    def test_it_solves_scruples_easy_when_untuned(self):
        baseline = self.BASELINE_MODEL

        # train the model
        _, train_features, train_labels, train_label_scores =\
            self.dataset.train
        baseline.fit(train_features, train_labels)

        # predict with the model on dev
        _, dev_features, dev_labels, dev_label_scores =\
            self.dataset.dev
        predictions = baseline.predict(dev_features)

        # check that the accuracy is 100%
        self.assertEqual(
            metrics.accuracy_score(
                y_true=dev_labels,
                y_pred=predictions),
            1.)

    @pytest.mark.slow
    def test_it_solves_scruples_easy_when_tuned(self):
        baseline = BayesSearchCV(
            self.BASELINE_MODEL,
            self.BASELINE_HYPER_PARAMETERS,
            n_iter=16,
            n_points=2,
            cv=4,
            n_jobs=1,
            refit=True)

        # train the model, tuning hyper-parameters
        _, train_features, train_labels, train_label_scores =\
            self.dataset.train
        baseline.fit(train_features, train_labels)

        # predict with the model on dev
        _, dev_features, dev_labels, dev_label_scores =\
            self.dataset.dev
        predictions = baseline.predict(dev_features)

        # check that the accuracy is 100%
        self.assertEqual(
            metrics.accuracy_score(
                y_true=dev_labels,
                y_pred=predictions),
            1.)

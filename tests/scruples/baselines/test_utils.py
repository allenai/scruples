"""Tests for scruples.baselines.utils."""

import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd

from scruples.baselines import utils


class ConcatTitleAndTextTestCase(unittest.TestCase):
    """Test concat_title_and_text."""

    def test_concatenates_title_and_text_columns(self):
        features = pd.DataFrame({
            'title': ['The Title A', 'The Title B'],
            'text': ['The text A.', 'The text B.']
        })

        self.assertEqual(
            utils.concat_title_and_text(features).tolist(),
            ['The Title A\nThe text A.', 'The Title B\nThe text B.'])


class DirichletMultinomialTestCase(unittest.TestCase):
    """Test dirichlet_multinomial."""

    def test_dirichlet_multinomial(self):
        self.assertTrue(np.allclose(
            utils.dirichlet_multinomial(
                [[np.log(0.1), np.log(0.1)]]
            ).tolist(),
            [[0.5, 0.5]]))
        self.assertTrue(np.allclose(
            utils.dirichlet_multinomial(
                [[np.log(1.), np.log(1.)],
                 [np.log(1.), np.log(4.)]]
            ).tolist(),
            [[0.5, 0.5], [0.2, 0.8]]))
        self.assertTrue(np.allclose(
            utils.dirichlet_multinomial(
                [[np.log(1.), np.log(1.), np.log(1.)]]
            ).tolist(),
            [[1/3., 1/3., 1/3.]]))
        self.assertTrue(np.allclose(
            utils.dirichlet_multinomial(
                [[np.log(1.), np.log(2.), np.log(1.)],
                 [np.log(2.), np.log(3.), np.log(5.)]]
            ).tolist(),
            [[0.25, 0.5, 0.25], [0.2, 0.3, 0.5]]))


class ResourceTransformerTestCase(unittest.TestCase):
    """Test ResourceTransformer."""

    def test_set_params(self):
        # mock out the inputs
        transformer = Mock()
        transformer.get_params.return_value = {'x': None}
        second_transformer = Mock()
        second_transformer.get_params.return_value = {'x': None}
        third_transformer = Mock()
        third_transformer.get_params.return_value = {'x': None}

        resource_transformer = utils.ResourceTransformer(
            transformer=transformer)

        # test setting params on the transformer attribute
        resource_transformer.set_params(transformer__x=1)

        transformer.set_params.assert_called()
        transformer.set_params.assert_called_with(x=1)

        # test setting params on the ResourceTransformer
        self.assertNotEqual(
            resource_transformer.transformer,
            second_transformer)

        resource_transformer.set_params(
            transformer=second_transformer)

        self.assertEqual(
            resource_transformer.transformer,
            second_transformer)

        # test setting params on the ResourceTransformer and the new
        # transformer at the same time
        self.assertNotEqual(
            resource_transformer.transformer,
            third_transformer)

        resource_transformer.set_params(
            transformer=third_transformer,
            transformer__x='foo')

        self.assertEqual(
            resource_transformer.transformer,
            third_transformer)

        third_transformer.set_params.asset_called()
        third_transformer.set_params.assert_called_with(x='foo')

    def test_fit(self):
        # create the data
        X = pd.DataFrame([
            {'action0': 1, 'action1': 4},
            {'action0': 2, 'action1': 5},
            {'action0': 3, 'action1': 6},
        ])

        # mock arguments to instantiate ResourceTransformer
        transformer = Mock()

        # create the ResourceTransformer instance
        resource_transformer = utils.ResourceTransformer(
            transformer=transformer)

        # run tests
        self.assertIsInstance(
            resource_transformer.fit(X),
            utils.ResourceTransformer)

        transformer.fit.assert_called()
        self.assertEqual(
            transformer.fit.call_args[0][0].tolist(),
            pd.concat([X['action0'], X['action1']]).tolist())

    def test_transform(self):
        # create the data
        X = pd.DataFrame([
            {'action0': 1, 'action1': 4},
            {'action0': 2, 'action1': 4},
            {'action0': 3, 'action1': 4}
        ])

        X_transformed = [6, 4, 2]

        class TimesTwoTransformer(object):
            def fit(self, X, y = None):
                return self

            def transform(self, X):
                return 2 * X

        resources = ['action0', 'action1']
        transformer = TimesTwoTransformer()

        resource_transformer = utils.ResourceTransformer(
            transformer=transformer)

        resource_transformer.fit(X)

        self.assertEqual(
            resource_transformer.transform(X).tolist(),
            X_transformed)

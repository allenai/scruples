"""Tests for scruples.demos.scoracle.app."""

import unittest

import numpy as np
import pytest
from sklearn import metrics

from scruples import utils
from scruples.demos.scoracle import app


class AppTestCase(unittest.TestCase):
    """Test scruples.demos.scoracle.app.app."""

    def setUp(self):
        # initialize the test client
        app.app.config['TESTING'] = True
        self.client = app.app.test_client()

    def test_home(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)

    def test_score_throws_error_for_no_json(self):
        response = self.client.post('/api/score')
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'No JSON',
                'message': 'Requests to this API endpoint must contain JSON.'
            }
        ])

    def test_score_throws_error_for_unexpected_keys(self):
        # test that score throws an error if there are only unexpected keys
        response = self.client.post('/api/score', json={'foo': 'bar'})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            next(e for e in response.json if e['error'] == 'Unexpected Key'),
            {
                'error': 'Unexpected Key',
                'message': 'The request object only accepts the "labelCounts"'
                           ' and "metrics" keys.'
            })

        # test that score throws an error if unexpected keys accompany expected
        # ones.
        response = self.client.post(
            '/api/score',
            json={
                'foo': 'bar',
                'labelCounts': [[1, 0], [0, 1]],
                'metrics': ['accuracy']
            })
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Unexpected Key',
                'message': 'The request object only accepts the "labelCounts"'
                           ' and "metrics" keys.'
            }
        ])

    def test_score_throws_error_for_missing_label_counts(self):
        response = self.client.post(
            '/api/score',
            json={'metrics': ['accuracy', 'f1 (macro)']})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Missing Key',
                'message': 'Please include "labelCounts" in your request.'
            }
        ])

    def test_score_throws_error_for_non_list_label_counts(self):
        # when labelCounts is null
        response = self.client.post(
            '/api/score',
            json={
                'labelCounts': None,
                'metrics': ['accuracy']
            })
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Wrong Type',
                'message': 'labelCounts must be an array.'
            }
        ])

        # when label counts is a string
        response = self.client.post(
            '/api/score',
            json={
                'labelCounts': 'foo',
                'metrics': ['accuracy']
            })
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Wrong Type',
                'message': 'labelCounts must be an array.'
            }
        ])

    def test_score_throws_error_for_list_of_non_lists_label_counts(self):
        # when labelCounts contains null items
        response = self.client.post(
            '/api/score',
            json={
                'labelCounts': [None],
                'metrics': ['accuracy']
            })
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Wrong Type',
                'message': 'labelCounts must be an array of arrays.'
            }
        ])

        response = self.client.post(
            '/api/score',
            json={
                'labelCounts': [[0, 1], None],
                'metrics': ['accuracy']
            })
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Wrong Type',
                'message': 'labelCounts must be an array of arrays.'
            }
        ])

        response = self.client.post(
            '/api/score',
            json={
                'labelCounts': [None, [1, 0]],
                'metrics': ['accuracy']
            })
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Wrong Type',
                'message': 'labelCounts must be an array of arrays.'
            }
        ])

        # when label counts contains string items
        response = self.client.post(
            '/api/score',
            json={
                'labelCounts': ['foo'],
                'metrics': ['accuracy']
            })
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Wrong Type',
                'message': 'labelCounts must be an array of arrays.'
            }
        ])

        response = self.client.post(
            '/api/score',
            json={
                'labelCounts': [[1, 0], 'foo'],
                'metrics': ['accuracy']
            })
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Wrong Type',
                'message': 'labelCounts must be an array of arrays.'
            }
        ])

        response = self.client.post(
            '/api/score',
            json={
                'labelCounts': ['foo', [0, 1]],
                'metrics': ['accuracy']
            })
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Wrong Type',
                'message': 'labelCounts must be an array of arrays.'
            }
        ])

    def test_score_throws_error_for_label_counts_with_non_integer_entries(self):
        # when labelCounts has arrays containing null items
        response = self.client.post(
            '/api/score',
            json={
                'labelCounts': [[None, 1], [0, 1]],
                'metrics': ['accuracy']
            })
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Wrong Type',
                'message': 'labelCounts must be an array of arrays of integers.'
            }
        ])

        response = self.client.post(
            '/api/score',
            json={
                'labelCounts': [[0, 1], [1, None]],
                'metrics': ['accuracy']
            })
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Wrong Type',
                'message': 'labelCounts must be an array of arrays of integers.'
            }
        ])

        response = self.client.post(
            '/api/score',
            json={
                'labelCounts': [[None, None], [1, 0]],
                'metrics': ['accuracy']
            })
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Wrong Type',
                'message': 'labelCounts must be an array of arrays of integers.'
            }
        ])

        # when label counts has arrays containing string items
        response = self.client.post(
            '/api/score',
            json={
                'labelCounts': [['foo', 1], [1, 2]],
                'metrics': ['accuracy']
            })
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Wrong Type',
                'message': 'labelCounts must be an array of arrays of integers.'
            }
        ])

        response = self.client.post(
            '/api/score',
            json={
                'labelCounts': [[1, 0], ['foo', 1]],
                'metrics': ['accuracy']
            })
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Wrong Type',
                'message': 'labelCounts must be an array of arrays of integers.'
            }
        ])

        response = self.client.post(
            '/api/score',
            json={
                'labelCounts': [['foo', 'bar'], [0, 1]],
                'metrics': ['accuracy']
            })
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Wrong Type',
                'message': 'labelCounts must be an array of arrays of integers.'
            }
        ])

    def test_score_throws_error_for_negative_entries(self):
        # some negative entries
        response = self.client.post(
            '/api/score',
            json={
                'labelCounts': [[-1, 1], [1, 1]],
                'metrics': ['accuracy']
            })
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Wrong Value',
                'message': 'labelCounts must have only nonnegative entries.'
            }
        ])

        # all negative entries
        response = self.client.post(
            '/api/score',
            json={
                'labelCounts': [[-1, -2], [-3, -1]],
                'metrics': ['accuracy']
            })
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Wrong Value',
                'message': 'labelCounts must have only nonnegative entries.'
            }
        ])

    def test_score_throws_error_for_empty_label_counts(self):
        response = self.client.post(
            '/api/score',
            json={'labelCounts': [], 'metrics': ['accuracy']})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Bad List Length',
                'message': 'labelCounts must be non-empty.'
            }
        ])

    def test_score_throws_error_for_label_counts_with_empty_arrays(self):
        response = self.client.post(
            '/api/score',
            json={'labelCounts': [[], []], 'metrics': ['accuracy']})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Bad List Length',
                'message': 'The arrays in labelCounts must be non-empty.'
            }
        ])

    def test_score_throws_error_for_non_rectangular_label_counts(self):
        response = self.client.post(
            '/api/score',
            json={'labelCounts': [[1, 2], [1]], 'metrics': ['accuracy']})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Bad List Length',
                'message': 'Each array in labelCounts must have the same length.'
            }
        ])

    def test_score_throws_error_for_missing_metrics(self):
        response = self.client.post(
            '/api/score',
            json={'labelCounts': [[1, 0], [0, 1]]})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Missing Key',
                'message': 'Please include "metrics" in your request.'
            }
        ])

    def test_score_throws_error_for_non_list_metrics(self):
        # when metrics is null
        response = self.client.post(
            '/api/score',
            json={
                'labelCounts': [[1, 0], [0, 1]],
                'metrics': None
            })
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Wrong Type',
                'message': 'metrics must be an array.'
            }
        ])

        # when metrics is a string
        response = self.client.post(
            '/api/score',
            json={
                'labelCounts': [[1, 0], [0, 1]],
                'metrics': 'foo'
            })
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Wrong Type',
                'message': 'metrics must be an array.'
            }
        ])

    def test_score_throws_error_for_metrics_having_non_string_entries(self):
        # when metrics contains a null
        response = self.client.post(
            '/api/score',
            json={
                'labelCounts': [[1, 0], [0, 1]],
                'metrics': ['accuracy', None]
            })
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Wrong Type',
                'message': 'metrics must be an array of strings.'
            }
        ])

        # when metrics contains an integer
        response = self.client.post(
            '/api/score',
            json={
                'labelCounts': [[1, 0], [0, 1]],
                'metrics': ['f1 (macro)', 1]
            })
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Wrong Type',
                'message': 'metrics must be an array of strings.'
            }
        ])

    def test_score_throws_error_for_unsupported_metrics(self):
        # when the only metric is unsupported
        response = self.client.post(
            '/api/score',
            json={
                'labelCounts': [[1, 0], [0, 1]],
                'metrics': ['foo score']
            })
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Bad Metric',
                'message': 'Unsupported metric found in metrics.'
            }
        ])

        # when there's a mix of supported and unsupported metrics
        response = self.client.post(
            '/api/score',
            json={
                'labelCounts': [[1, 0], [0, 1]],
                'metrics': ['accuracy', 'foo score']
            })
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Bad Metric',
                'message': 'Unsupported metric found in metrics.'
            }
        ])

    def test_score_can_throw_multiple_errors(self):
        response = self.client.post(
            '/api/score',
            json={
                'labelCounts': [[0], [0, 1]],
                'metrics': ['foo score']
            })
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Bad List Length',
                'message': 'Each array in labelCounts must have the same length.'
            },
            {
                'error': 'Bad Metric',
                'message': 'Unsupported metric found in metrics.'
            }
        ])

    @pytest.mark.slow
    def test_score_computes_oracle_scores(self):
        # when providing a single metric
        response = self.client.post(
            '/api/score',
            json={
                'labelCounts': [[3, 1], [1, 1], [1, 3]],
                'metrics': ['accuracy']
            })
        # test that the score is correct
        score = response.json[0]['score']
        gold_score, _ = utils.oracle_performance(
            ys=np.array([[3, 1], [1, 1], [1, 3]]),
            metric=metrics.accuracy_score,
            make_predictions=lambda ys: np.argmax(ys, axis=-1),
            n_samples=10000)
        self.assertLess(
            abs(gold_score - score) / gold_score,
            0.05)
        # test that the response is correct
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, [
            {
                'metric': 'accuracy',
                'score': score
            }
        ])

        # when providing multiple metrics
        response = self.client.post(
            '/api/score',
            json={
                'labelCounts': [[3, 1], [1, 1], [1, 3]],
                'metrics': ['accuracy', 'f1 (macro)']
            })
        metric_to_score = {
            x['metric']: x['score']
            for x in response.json
        }
        # test that the score is correct
        accuracy_score = metric_to_score['accuracy']
        gold_accuracy_score, _ = utils.oracle_performance(
            ys=np.array([[3, 1], [1, 1], [1, 3]]),
            metric=metrics.accuracy_score,
            make_predictions=lambda ys: np.argmax(ys, axis=-1),
            n_samples=10000)
        self.assertLess(
            abs(gold_accuracy_score - accuracy_score) / gold_accuracy_score,
            0.05)
        f1_score = metric_to_score['f1 (macro)']
        gold_f1_score, _ = utils.oracle_performance(
            ys=np.array([[3, 1], [1, 1], [1, 3]]),
            metric=lambda y_true, y_pred: metrics.f1_score(
                y_true=y_true, y_pred=y_pred, average='macro'),
            make_predictions=lambda ys: np.argmax(ys, axis=-1),
            n_samples=10000)
        self.assertLess(
            abs(gold_f1_score - f1_score) / gold_f1_score,
            0.05)
        # test that the response is correct
        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            response.json == [
                {
                    'metric': 'accuracy',
                    'score': accuracy_score
                },
                {
                    'metric': 'f1 (macro)',
                    'score': f1_score
                }
            ]
            or
            response.json == [
                {
                    'metric': 'f1 (macro)',
                    'score': f1_score
                },
                {
                    'metric': 'accuracy',
                    'score': accuracy_score
                }
            ])

"""Tests for scruples.demos.norms.app."""

import unittest

import pytest

from scruples import settings
from scruples.demos.norms import app


class AppTestCase(unittest.TestCase):
    """Test scruples.demos.norms.app.app."""

    def setUp(self):
        # initialize the test client
        app.app.config['TESTING'] = True
        self.client = app.app.test_client()

    def test_home(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)

    def test_predict_actions_throws_error_for_no_json(self):
        response = self.client.post('/api/actions/predict')
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'No JSON',
                'message': 'Requests to this API endpoint must contain JSON.'
            }
        ])

    def test_predict_actions_throws_error_for_non_list(self):
        # when the posted JSON is an object
        response = self.client.post('/api/actions/predict', json={'foo': 'bar'})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Wrong Type',
                'message': 'The request data must be an array of objects.'
            }
        ])

        # when the posted JSON is a string
        response = self.client.post('/api/actions/predict', json='foo')
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Wrong Type',
                'message': 'The request data must be an array of objects.'
            }
        ])

    def test_predict_actions_throws_error_for_when_entries_are_not_objects(self):
        # when the posted JSON has null entries
        response = self.client.post('/api/actions/predict', json=[None])
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Wrong Type',
                'message': 'Each element of the data array must be an object.'
            }
        ])
        response = self.client.post('/api/actions/predict', json=[
            {'action1': 'foo', 'action2': 'bar'},
            None
        ])
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Wrong Type',
                'message': 'Each element of the data array must be an object.'
            }
        ])

        # when the posted JSON has string entries
        response = self.client.post('/api/actions/predict', json=['foo'])
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Wrong Type',
                'message': 'Each element of the data array must be an object.'
            }
        ])
        response = self.client.post('/api/actions/predict', json=[
            {'action1': 'foo', 'action2': 'bar'},
            'foo'
        ])
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Wrong Type',
                'message': 'Each element of the data array must be an object.'
            }
        ])

        # when the posted JSON has array entries
        response = self.client.post('/api/actions/predict', json=[['foo']])
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Wrong Type',
                'message': 'Each element of the data array must be an object.'
            }
        ])
        response = self.client.post('/api/actions/predict', json=[
            {'action1': 'foo', 'action2': 'bar'},
            ['foo']
        ])
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Wrong Type',
                'message': 'Each element of the data array must be an object.'
            }
        ])

    def test_predict_actions_throws_error_for_missing_keys_in_object(self):
        # when action1 is missing in an object
        response = self.client.post('/api/actions/predict', json=[
            {'action2': 'foo'}
        ])
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Missing Key',
                'message': 'Each object must have an "action1" key.'
            }
        ])

        # when action1 is missing in some objects
        response = self.client.post('/api/actions/predict', json=[
            {'action1': 'foo', 'action2': 'bar'},
            {'action2': 'foo'}
        ])
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Missing Key',
                'message': 'Each object must have an "action1" key.'
            }
        ])

        # when action2 is missing in an object
        response = self.client.post('/api/actions/predict', json=[
            {'action1': 'foo'}
        ])
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Missing Key',
                'message': 'Each object must have an "action2" key.'
            }
        ])

        # when action2 is missing in some objects
        response = self.client.post('/api/actions/predict', json=[
            {'action1': 'foo', 'action2': 'bar'},
            {'action1': 'foo'}
        ])
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Missing Key',
                'message': 'Each object must have an "action2" key.'
            }
        ])

    def test_predict_actions_throws_error_on_non_string_values(self):
        # when action1 has a null, non-string value
        response = self.client.post('/api/actions/predict', json=[
            {'action1': None, 'action2': 'bar'},
        ])
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Wrong Type',
                'message': 'The value corresponding to the "action1" key'
                            ' must be a string.'
            }
        ])

        # when action2 has a null, non-string value
        response = self.client.post('/api/actions/predict', json=[
            {'action1': 'foo', 'action2': None},
        ])
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Wrong Type',
                'message': 'The value corresponding to the "action2" key'
                            ' must be a string.'
            }
        ])

        # when action1 has an integer, non-string value
        response = self.client.post('/api/actions/predict', json=[
            {'action1': 1, 'action2': 'bar'},
        ])
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Wrong Type',
                'message': 'The value corresponding to the "action1" key'
                            ' must be a string.'
            }
        ])

        # when action2 has an integer, non-string value
        response = self.client.post('/api/actions/predict', json=[
            {'action1': 'foo', 'action2': 5},
        ])
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Wrong Type',
                'message': 'The value corresponding to the "action2" key'
                            ' must be a string.'
            }
        ])

    def test_predict_actions_throws_error_for_unrecognized_keys(self):
        # when an object has an unrecognized key
        response = self.client.post('/api/actions/predict', json=[
            {'action1': 'foo', 'action2': 'bar', 'baz': 'qux'}
        ])
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Unexpected Key',
                'message': 'Each object must only have "action1" and "action2"'
                           ' keys.'
            }
        ])

        # when some objects have unrecognized keys
        response = self.client.post('/api/actions/predict', json=[
            {'action1': 'foo', 'action2': 'bar'},
            {'action1': 'foo', 'action2': 'bar', 'baz': 'qux'}
        ])
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Unexpected Key',
                'message': 'Each object must only have "action1" and "action2"'
                           ' keys.'
            }
        ])

    def test_predict_actions_can_throw_multiple_errors(self):
        response = self.client.post('/api/actions/predict', json=[
            {'action1': 'foo'},
            {'action1': 'foo', 'action2': 'bar', 'baz': 'qux'}
        ])
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Missing Key',
                'message': 'Each object must have an "action2" key.'
            },
            {
                'error': 'Unexpected Key',
                'message': 'Each object must only have "action1" and "action2"'
                           ' keys.'
            }
        ])

    @pytest.mark.slow
    @pytest.mark.skipif(
        settings.NORMS_ACTIONS_MODEL is None
        or settings.NORMS_PREDICT_BATCH_SIZE is None,
        reason='requires the norms demo environment variables.')
    def test_predict_actions_computes_action_prediction(self):
        response = self.client.post('/api/actions/predict', json=[
            {'action1': 'foo', 'action2': 'bar'}
        ])
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json), 1)
        self.assertEqual(
            set(response.json[0].keys()),
            set(['action1', 'action2']))
        self.assertIsInstance(response.json[0]['action1'], float)
        self.assertIsInstance(response.json[0]['action2'], float)

    @pytest.mark.slow
    @pytest.mark.skipif(
        settings.NORMS_ACTIONS_MODEL is None
        or settings.NORMS_PREDICT_BATCH_SIZE is None,
        reason='requires the norms demo environment variables.')
    def test_predict_actions_computes_plots(self):
        response = self.client.post('/api/actions/predict?plot=true', json=[
            {'action1': 'foo', 'action2': 'bar'}
        ])
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json), 1)
        self.assertEqual(
            set(response.json[0].keys()),
            set(['action1', 'action2', 'plot']))
        self.assertIsInstance(response.json[0]['action1'], float)
        self.assertIsInstance(response.json[0]['action2'], float)
        self.assertIsInstance(response.json[0]['plot'], str)

    def test_predict_corpus_throws_error_for_no_json(self):
        response = self.client.post('/api/corpus/predict')
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'No JSON',
                'message': 'Requests to this API endpoint must contain JSON.'
            }
        ])

    def test_predict_corpus_throws_error_for_non_list(self):
        # when the posted JSON is an object
        response = self.client.post('/api/corpus/predict', json={'foo': 'bar'})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Wrong Type',
                'message': 'The request data must be an array of objects.'
            }
        ])

        # when the posted JSON is a string
        response = self.client.post('/api/corpus/predict', json='foo')
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Wrong Type',
                'message': 'The request data must be an array of objects.'
            }
        ])

    def test_predict_corpus_throws_error_when_entries_are_not_objects(self):
        # when the posted JSON has null entries
        response = self.client.post('/api/corpus/predict', json=[None])
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Wrong Type',
                'message': 'Each element of the data array must be an object.'
            }
        ])
        response = self.client.post('/api/corpus/predict', json=[
            {'title': 'foo', 'text': 'bar'},
            None
        ])
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Wrong Type',
                'message': 'Each element of the data array must be an object.'
            }
        ])

        # when the posted JSON has string entries
        response = self.client.post('/api/corpus/predict', json=['foo'])
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Wrong Type',
                'message': 'Each element of the data array must be an object.'
            }
        ])
        response = self.client.post('/api/corpus/predict', json=[
            {'title': 'foo', 'text': 'bar'},
            'foo'
        ])
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Wrong Type',
                'message': 'Each element of the data array must be an object.'
            }
        ])

        # when the posted JSON has array entries
        response = self.client.post('/api/corpus/predict', json=[['foo']])
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Wrong Type',
                'message': 'Each element of the data array must be an object.'
            }
        ])
        response = self.client.post('/api/corpus/predict', json=[
            {'title': 'foo', 'text': 'bar'},
            ['foo']
        ])
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Wrong Type',
                'message': 'Each element of the data array must be an object.'
            }
        ])

    def test_predict_corpus_throws_error_for_missing_keys_in_object(self):
        # when title is missing in an object
        response = self.client.post('/api/corpus/predict', json=[
            {'text': 'foo'}
        ])
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Missing Key',
                'message': 'Each object must have a "title" key.'
            }
        ])

        # when title is missing in some objects
        response = self.client.post('/api/corpus/predict', json=[
            {'title': 'foo', 'text': 'bar'},
            {'text': 'foo'}
        ])
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Missing Key',
                'message': 'Each object must have a "title" key.'
            }
        ])

        # when text is missing in an object
        response = self.client.post('/api/corpus/predict', json=[
            {'title': 'foo'}
        ])
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Missing Key',
                'message': 'Each object must have a "text" key.'
            }
        ])

        # when text is missing in some objects
        response = self.client.post('/api/corpus/predict', json=[
            {'title': 'foo', 'text': 'bar'},
            {'title': 'foo'}
        ])
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Missing Key',
                'message': 'Each object must have a "text" key.'
            }
        ])

    def test_predict_corpus_throws_error_on_non_string_values(self):
        # when title has a null, non-string value
        response = self.client.post('/api/corpus/predict', json=[
            {'title': None, 'text': 'bar'},
        ])
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Wrong Type',
                'message': 'The value corresponding to the "title" key'
                            ' must be a string.'
            }
        ])

        # when text has a null, non-string value
        response = self.client.post('/api/corpus/predict', json=[
            {'title': 'foo', 'text': None},
        ])
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Wrong Type',
                'message': 'The value corresponding to the "text" key'
                            ' must be a string.'
            }
        ])

        # when title has an integer, non-string value
        response = self.client.post('/api/corpus/predict', json=[
            {'title': 1, 'text': 'bar'},
        ])
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Wrong Type',
                'message': 'The value corresponding to the "title" key'
                            ' must be a string.'
            }
        ])

        # when text has an integer, non-string value
        response = self.client.post('/api/corpus/predict', json=[
            {'title': 'foo', 'text': 5},
        ])
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Wrong Type',
                'message': 'The value corresponding to the "text" key'
                            ' must be a string.'
            }
        ])

    def test_predict_corpus_throws_error_for_unrecognized_keys(self):
        # when an object has an unrecognized key
        response = self.client.post('/api/corpus/predict', json=[
            {'title': 'foo', 'text': 'bar', 'baz': 'qux'}
        ])
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Unexpected Key',
                'message': 'Each object must only have "title" and "text"'
                           ' keys.'
            }
        ])

        # when some objects have unrecognized keys
        response = self.client.post('/api/corpus/predict', json=[
            {'title': 'foo', 'text': 'bar'},
            {'title': 'foo', 'text': 'bar', 'baz': 'qux'}
        ])
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Unexpected Key',
                'message': 'Each object must only have "title" and "text"'
                           ' keys.'
            }
        ])

    def test_predict_corpus_can_throw_multiple_errors(self):
        response = self.client.post('/api/corpus/predict', json=[
            {'title': 'foo'},
            {'title': 'foo', 'text': 'bar', 'baz': 'qux'}
        ])
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, [
            {
                'error': 'Missing Key',
                'message': 'Each object must have a "text" key.'
            },
            {
                'error': 'Unexpected Key',
                'message': 'Each object must only have "title" and "text"'
                           ' keys.'
            }
        ])

    @pytest.mark.slow
    @pytest.mark.skipif(
        settings.NORMS_CORPUS_MODEL is None
        or settings.NORMS_PREDICT_BATCH_SIZE is None,
        reason='requires the norms demo environment variables.')
    def test_predict_corpus_computes_corpus_prediction(self):
        response = self.client.post('/api/corpus/predict', json=[
            {'title': 'foo', 'text': 'bar'}
        ])
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json), 1)
        self.assertEqual(
            set(response.json[0].keys()),
            set([
                'AUTHOR',
                'OTHER',
                'EVERYBODY',
                'NOBODY',
                'INFO'
            ]))
        self.assertIsInstance(response.json[0]['AUTHOR'], float)
        self.assertIsInstance(response.json[0]['OTHER'], float)
        self.assertIsInstance(response.json[0]['EVERYBODY'], float)
        self.assertIsInstance(response.json[0]['NOBODY'], float)
        self.assertIsInstance(response.json[0]['INFO'], float)

    @pytest.mark.slow
    @pytest.mark.skipif(
        settings.NORMS_CORPUS_MODEL is None
        or settings.NORMS_PREDICT_BATCH_SIZE is None,
        reason='requires the norms demo environment variables.')
    def test_predict_corpus_computes_plots(self):
        response = self.client.post('/api/corpus/predict?plot=true', json=[
            {'title': 'foo', 'text': 'bar'}
        ])
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json), 1)
        self.assertEqual(
            set(response.json[0].keys()),
            set([
                'AUTHOR',
                'OTHER',
                'EVERYBODY',
                'NOBODY',
                'INFO',
                'plot_author',
                'plot_other'
            ]))
        self.assertIsInstance(response.json[0]['AUTHOR'], float)
        self.assertIsInstance(response.json[0]['OTHER'], float)
        self.assertIsInstance(response.json[0]['EVERYBODY'], float)
        self.assertIsInstance(response.json[0]['NOBODY'], float)
        self.assertIsInstance(response.json[0]['INFO'], float)
        self.assertIsInstance(response.json[0]['plot_author'], str)
        self.assertIsInstance(response.json[0]['plot_other'], str)

"""Tests for socialnorms.data.comment."""

import json
import pkg_resources
import unittest

from socialnorms.data import comment
from socialnorms.data import labels
from ... import settings


class CommentTestCase(unittest.TestCase):
    """Test socialnorms.data.comment.Comment."""

    def setUp(self):
        with pkg_resources.resource_stream(
                'tests', settings.TEST_COMMENT_PATH
        ) as comment_file:
            self.comment_kwargs = json.load(comment_file)

    def test_extracts_labels_correctly(self):
        # test that NTA is extracted correctly
        nta_kwargs = self.comment_kwargs.copy()
        nta_kwargs.update(body='I believe this post is NTA.')
        self.assertEqual(
            comment.Comment(**nta_kwargs).label,
            labels.Label.NTA)

        # test that YTA is extracted correctly
        yta_kwargs = self.comment_kwargs.copy()
        yta_kwargs.update(body='I believe this post is YTA.')
        self.assertEqual(
            comment.Comment(**yta_kwargs).label,
            labels.Label.YTA)

        # test that ESH is extracted correctly
        esh_kwargs = self.comment_kwargs.copy()
        esh_kwargs.update(body='I believe this post is ESH.')
        self.assertEqual(
            comment.Comment(**esh_kwargs).label,
            labels.Label.ESH)

        # test that NAH is extracted correctly
        nah_kwargs = self.comment_kwargs.copy()
        nah_kwargs.update(body='I believe this post is NAH.')
        self.assertEqual(
            comment.Comment(**nah_kwargs).label,
            labels.Label.NAH)

        # test that INFO is extracted correctly
        info_kwargs = self.comment_kwargs.copy()
        info_kwargs.update(body='I believe this post is INFO.')
        self.assertEqual(
            comment.Comment(**info_kwargs).label,
            labels.Label.INFO)

    def test_is_top_level(self):
        # test when is_top_level should be true
        kwargs = self.comment_kwargs.copy()
        kwargs.update(link_id='t3_aaaaaa', parent_id='t3_aaaaaa')

        self.assertEqual(
            comment.Comment(**kwargs).is_top_level,
            True)

        # test when is_top_level should be false
        kwargs = self.comment_kwargs.copy()
        kwargs.update(link_id='t3_aaaaaa', parent_id='t1_aaaaaa')

        self.assertEqual(
            comment.Comment(**kwargs).is_top_level,
            False)

    def test_has_empty_body(self):
        # test when has_empty_body should be true
        kwargs = self.comment_kwargs.copy()
        kwargs.update(body='')

        self.assertEqual(
            comment.Comment(**kwargs).has_empty_body,
            True)

        # test when has_empty_body should be false
        kwargs = self.comment_kwargs.copy()
        kwargs.update(body='This string is not empty.')

        self.assertEqual(
            comment.Comment(**kwargs).has_empty_body,
            False)

    def test_is_deleted(self):
        # test when is_deleted should be true
        kwargs = self.comment_kwargs.copy()
        kwargs.update(body='[deleted]')

        self.assertEqual(
            comment.Comment(**kwargs).is_deleted,
            True)

        kwargs = self.comment_kwargs.copy()
        kwargs.update(body='[removed]')

        self.assertEqual(
            comment.Comment(**kwargs).is_deleted,
            True)

        # test when is_deleted should be false
        kwargs = self.comment_kwargs.copy()
        kwargs.update(body='A comment.')

        self.assertEqual(
            comment.Comment(**kwargs).is_deleted,
            False)

    def test_is_by_automoderator(self):
        # test when is_by_automoderator should be true
        kwargs = self.comment_kwargs.copy()
        kwargs.update(author='AutoModerator')

        self.assertEqual(
            comment.Comment(**kwargs).is_by_automoderator,
            True)

        # test when is_by_automoderator should be false
        kwargs = self.comment_kwargs.copy()
        kwargs.update(author='example-user')

        self.assertEqual(
            comment.Comment(**kwargs).is_by_automoderator,
            False)

    def test_is_good(self):
        # test when is_good should be true
        kwargs = self.comment_kwargs.copy()

        self.assertEqual(
            comment.Comment(**kwargs).is_good,
            True)

        # test when is_good should be false
        # is_top_level == False
        kwargs = self.comment_kwargs.copy()
        kwargs.update(parent_id='t1_aaaaaa')
        self.assertEqual(
            comment.Comment(**kwargs).is_good,
            False)
        # has_empty_body == True
        kwargs = self.comment_kwargs.copy()
        kwargs.update(body='')
        self.assertEqual(
            comment.Comment(**kwargs).is_good,
            False)
        # is_deleted == True
        kwargs = self.comment_kwargs.copy()
        kwargs.update(body='[deleted]')
        self.assertEqual(
            comment.Comment(**kwargs).is_good,
            False)
        # is_by_automoderator == True
        kwargs = self.comment_kwargs.copy()
        kwargs.update(author='AutoModerator')
        self.assertEqual(
            comment.Comment(**kwargs).is_good,
            False)

"""Tests for scruples.data.comment."""

import json
import pkg_resources
import unittest

from scruples.data import comment
from scruples.data import labels
from ... import settings


class CommentTestCase(unittest.TestCase):
    """Test scruples.data.comment.Comment."""

    def setUp(self):
        with pkg_resources.resource_stream(
                'tests', settings.TEST_COMMENT_PATH
        ) as comment_file:
            self.comment_kwargs = json.load(comment_file)

    def test_extracts_labels_correctly(self):
        # test that OTHER is extracted correctly
        other_kwargs = self.comment_kwargs.copy()
        other_kwargs.update(body='I believe this post is NTA.')
        self.assertEqual(
            comment.Comment(**other_kwargs).label,
            labels.Label.OTHER)

        # test that AUTHOR is extracted correctly
        author_kwargs = self.comment_kwargs.copy()
        author_kwargs.update(body='I believe this post is YTA.')
        self.assertEqual(
            comment.Comment(**author_kwargs).label,
            labels.Label.AUTHOR)

        # test that EVERYBODY is extracted correctly
        everybody_kwargs = self.comment_kwargs.copy()
        everybody_kwargs.update(body='I believe this post is ESH.')
        self.assertEqual(
            comment.Comment(**everybody_kwargs).label,
            labels.Label.EVERYBODY)

        # test that NOBODY is extracted correctly
        nobody_kwargs = self.comment_kwargs.copy()
        nobody_kwargs.update(body='I believe this post is NAH.')
        self.assertEqual(
            comment.Comment(**nobody_kwargs).label,
            labels.Label.NOBODY)

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

    def test_is_spam(self):
        # test when is_spam should be false
        kwargs = self.comment_kwargs.copy()

        self.assertEqual(
            comment.Comment(**kwargs).is_spam,
            False)

        # test when is_spam should be true
        # has_empty_body == True
        kwargs = self.comment_kwargs.copy()
        kwargs.update(body='')
        self.assertEqual(
            comment.Comment(**kwargs).is_spam,
            True)
        # is_deleted == True
        kwargs = self.comment_kwargs.copy()
        kwargs.update(body='[deleted]')
        self.assertEqual(
            comment.Comment(**kwargs).is_spam,
            True)
        # is_by_automoderator == True
        kwargs = self.comment_kwargs.copy()
        kwargs.update(author='AutoModerator')
        self.assertEqual(
            comment.Comment(**kwargs).is_spam,
            True)

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
        # is_spam == True
        kwargs = self.comment_kwargs.copy()
        test_comment = comment.Comment(**kwargs)
        # patch test_comment with True for is_spam. We'll patch where
        # the cached_property decorator will look for the cached value,
        # since the attribute itself is read-only. See
        # scruples.data.utils.cached_property for how the cached
        # properties work.
        object.__setattr__(test_comment, '_is_spam', True)
        self.assertEqual(test_comment.is_good, False)

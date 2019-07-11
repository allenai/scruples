"""Tests for scruples.data.post."""

import json
import pkg_resources
import unittest
from unittest.mock import patch

import attr

from scruples.data import post
from scruples.data.comment import Comment
from scruples.data.label_scores import LabelScores
from scruples.data.labels import Label
from scruples.data.post_types import PostType

from ... import settings


class PostTestCase(unittest.TestCase):
    """Test scruples.data.post.Post."""

    def setUp(self):
        # load the comments for the test post
        with pkg_resources.resource_stream(
                'tests', settings.TEST_POST_COMMENTS_PATH
        ) as test_post_comments_file:
            self.post_comments = [
                Comment(**json.loads(ln))
                for ln in test_post_comments_file
            ]

        # load the test post kwargs
        with pkg_resources.resource_stream(
                'tests', settings.TEST_POST_PATH
        ) as test_post_file:
            post_kwargs = json.load(test_post_file)
            post_kwargs['comments'] = self.post_comments
            self.post_kwargs = post_kwargs

    def test_label_scores(self):
        kwargs = self.post_kwargs.copy()

        self.assertEqual(
            post.Post(**kwargs).label_scores,
            LabelScores(label_to_score={
                Label.AUTHOR: 1,
                Label.OTHER: 2,
                Label.EVERYBODY: 0,
                Label.NOBODY: 0,
                Label.INFO: 0
            }))

    def test_action(self):
        # test that action extraction failures become None
        kwargs = self.post_kwargs.copy()

        self.assertEqual(post.Post(**kwargs).action, None)

        # test a correct action extraction
        kwargs = self.post_kwargs.copy()
        kwargs.update(
            title='AITA for telling someone that they were in my way?')

        action = post.Post(**kwargs).action

        self.assertEqual(
            action.description,
            'telling someone that they were in my way')
        self.assertEqual(action.pronormative_score, 2)
        self.assertEqual(action.contranormative_score, 1)

    def test_original_text(self):
        # test when the original text can be extracted from the comments
        kwargs = self.post_kwargs.copy()

        self.assertEqual(
            post.Post(**kwargs).original_text,
            "The original text.")

        # test when the original text cannot be extracted from the
        # comments
        kwargs = self.post_kwargs.copy()
        kwargs.update(comments=[])

        self.assertEqual(
            post.Post(**kwargs).original_text,
            None)

        # test when the original text is the empty string, since the
        # empty string being false-y can cause it to be treated
        # similarly to None
        kwargs = self.post_kwargs.copy()
        original_text_comment = kwargs['comments'][-1]
        original_text_comment = attr.evolve(
            original_text_comment,
            body='^^^^AUTOMOD  ***This is a copy of the above post.'
                 ' It is a record of the post as originally written, in'
                 ' case the post is deleted or edited.***\n\n\n\n*I am'
                 ' a bot, and this action was performed automatically.'
                 ' Please [contact the moderators of this'
                 ' subreddit](/message/compose/?to=/r/AmItheAsshole)'
                 ' if you have any questions or concerns.*')
        kwargs.update(comments=[original_text_comment])

        self.assertEqual(post.Post(**kwargs).original_text, '')

    def test_post_type(self):
        # test when post type should be HISTORICAL
        kwargs = self.post_kwargs.copy()
        kwargs.update(title='AITA for what I did?')

        self.assertEqual(
            post.Post(**kwargs).post_type,
            PostType.HISTORICAL)

        # test when post type should be HYPOTHETICAL
        kwargs = self.post_kwargs.copy()
        kwargs.update(title='WIBTA for this?')

        self.assertEqual(
            post.Post(**kwargs).post_type,
            PostType.HYPOTHETICAL)

        # test when post type should be META
        kwargs = self.post_kwargs.copy()
        kwargs.update(title='META: should we implement this change?')

        self.assertEqual(
            post.Post(**kwargs).post_type,
            PostType.META)

        # test when post type should be None
        kwargs = self.post_kwargs.copy()
        kwargs.update(title='A question')

        self.assertEqual(
            post.Post(**kwargs).post_type,
            None)

    def test_has_empty_selftext(self):
        # test when has_empty_selftext should be true
        kwargs = self.post_kwargs.copy()
        kwargs.update(selftext='')

        self.assertEqual(
            post.Post(**kwargs).has_empty_selftext,
            True)

        # test when has_empty_selftext should be false
        kwargs = self.post_kwargs.copy()
        kwargs.update(selftext='Not empty.')

        self.assertEqual(
            post.Post(**kwargs).has_empty_selftext,
            False)

    def test_is_deleted(self):
        # test when is_deleted should be true
        kwargs = self.post_kwargs.copy()
        kwargs.update(selftext='[deleted]')

        self.assertEqual(
            post.Post(**kwargs).is_deleted,
            True)

        kwargs = self.post_kwargs.copy()
        kwargs.update(selftext='[removed]')

        self.assertEqual(
            post.Post(**kwargs).is_deleted,
            True)

        # test when is_deleted should be false
        kwargs = self.post_kwargs.copy()
        kwargs.update(selftext='Not deleted.')

        self.assertEqual(
            post.Post(**kwargs).is_deleted,
            False)

    def test_has_deleted_author(self):
        # test when has_deleted_author should be true
        kwargs = self.post_kwargs.copy()
        kwargs.update(author='[deleted]')

        self.assertEqual(
            post.Post(**kwargs).has_deleted_author,
            True)

        # test when has_deleted_author should be false
        kwargs = self.post_kwargs.copy()
        kwargs.update(author='foo')

        self.assertEqual(
            post.Post(**kwargs).has_deleted_author,
            False)

    def test_has_post_type(self):
        # test when has_post_type should be true
        # when post_type is HISTORICAL
        kwargs = self.post_kwargs.copy()
        kwargs.update(title='AITA for this?')

        self.assertEqual(
            post.Post(**kwargs).has_post_type,
            True)

        # when post_type is HYPOTHETICAL
        kwargs = self.post_kwargs.copy()
        kwargs.update(title='WIBTA for this?')

        self.assertEqual(
            post.Post(**kwargs).has_post_type,
            True)

        # when post_type is META
        kwargs = self.post_kwargs.copy()
        kwargs.update(title='META What do you think about...?')

        self.assertEqual(
            post.Post(**kwargs).has_post_type,
            True)

        # test when has_post_type should be false
        kwargs = self.post_kwargs.copy()
        kwargs.update(title='A question')

        self.assertEqual(
            post.Post(**kwargs).has_post_type,
            False)

    def test_is_meta(self):
        # test when is_meta should be true
        kwargs = self.post_kwargs.copy()
        kwargs.update(title='[META]: what do you think?')

        self.assertEqual(
            post.Post(**kwargs).is_meta,
            True)

        # test when is_meta should be false
        # when post_type is present but not META
        kwargs = self.post_kwargs.copy()
        kwargs.update(title='AITA for this?')

        self.assertEqual(
            post.Post(**kwargs).is_meta,
            False)

        # when post_type is not present
        kwargs = self.post_kwargs.copy()
        kwargs.update(title='A question')

        test_post = post.Post(**kwargs)
        self.assertIsNone(test_post.post_type)
        self.assertEqual(test_post.is_meta, False)

    def test_has_original_text(self):
        # test when has_original_text should be true
        kwargs = self.post_kwargs.copy()

        self.assertEqual(
            post.Post(**kwargs).has_original_text,
            True)

        # test when has_original_text should be false
        kwargs = self.post_kwargs.copy()
        kwargs.update(comments=[])

        test_post = post.Post(**kwargs)
        self.assertIsNone(test_post.original_text)
        self.assertEqual(test_post.has_original_text, False)

    def test_has_enough_content(self):
        # test when has_enough_content should be true
        # when the original text is present
        kwargs = self.post_kwargs.copy()
        kwargs.update(
            title='AITA This title plus the original text'
                  ' tokens should have more than 16 tokens.')

        self.assertEqual(
            post.Post(**kwargs).has_enough_content,
            True)
        # when the original text is not present
        kwargs = self.post_kwargs.copy()
        kwargs.update(
            title='AITA A post with some content',
            selftext='This post should have more than 16 content tokens'
                     ' in total.',
            comments=[])

        test_post = post.Post(**kwargs)
        self.assertFalse(test_post.has_original_text)
        self.assertEqual(test_post.has_enough_content, True)

        # test when has_enough_content should be false
        # when the original text is present
        kwargs = self.post_kwargs.copy()
        kwargs.update(
            title='AITA Too little content')

        self.assertEqual(
            post.Post(**kwargs).has_enough_content,
            False)
        # when the original text is not present
        kwargs = self.post_kwargs.copy()
        kwargs.update(
            title='AITA A post with little content',
            selftext='Too little content.',
            comments=[])

        test_post = post.Post(**kwargs)
        self.assertFalse(test_post.has_original_text)
        self.assertEqual(test_post.has_enough_content, False)

    def test_has_good_label_scores(self):
        # test when has_good_label_scores should be true
        kwargs = self.post_kwargs.copy()

        test_post = post.Post(**kwargs)
        self.assertTrue(test_post.label_scores.is_good)
        self.assertEqual(test_post.has_good_label_scores, True)

        # test when has_good_label_scores should be false
        kwargs = self.post_kwargs.copy()
        kwargs.update(comments=[])

        test_post = post.Post(**kwargs)
        self.assertFalse(test_post.label_scores.is_good)
        self.assertEqual(test_post.has_good_label_scores, False)

    def test_is_spam(self):
        # test when is_spam should be false
        kwargs = self.post_kwargs.copy()
        kwargs.update(
            title='AITA The title must be long enough so that the post'
                  ' has sufficient content to be good.')

        self.assertEqual(
            post.Post(**kwargs).is_spam,
            False)

        # test when is_spam should be true
        # when is_self is false
        kwargs = self.post_kwargs.copy()
        kwargs.update(is_self=False)

        self.assertEqual(
            post.Post(**kwargs).is_spam,
            True)
        # iterate through cached properties and the truth values that
        # should make is_spam true
        for attribute, truth_value in [
                ('has_empty_selftext', True),
                ('is_deleted', True),
                ('has_deleted_author', True),
                ('has_post_type', False),
                ('is_meta', True),
                ('has_enough_content', False)
        ]:
            kwargs = self.post_kwargs.copy()

            test_post = post.Post(**kwargs)

            # patch test_post with the bad truth value for the cached
            # property. We'll patch where the cached_property decorator
            # will look for the cached value, since the attribute itself
            # is read-only. See scruples.data.utils.cached_property
            # for how the cached properties work.
            object.__setattr__(test_post, f'_{attribute}', truth_value)

            self.assertEqual(test_post.is_spam, True)

    def test_is_good(self):
        # test when is_good should be true
        kwargs = self.post_kwargs.copy()
        kwargs.update(
            title='AITA The title must be long enough so that the post'
                  ' has sufficient content to be good.')

        self.assertEqual(
            post.Post(**kwargs).is_good,
            True)

        # test when is_good should be false
        #
        # iterate through cached properties and the truth values that
        # should make is_good false
        for attribute, truth_value in [
                ('is_spam', True),
                ('has_original_text', False),
                ('has_good_label_scores', False)
        ]:
            kwargs = self.post_kwargs.copy()

            test_post = post.Post(**kwargs)

            # patch test_post with the bad truth value for the cached
            # property. We'll patch where the cached_property decorator
            # will look for the cached value, since the attribute itself
            # is read-only. See scruples.data.utils.cached_property
            # for how the cached properties work.
            object.__setattr__(test_post, f'_{attribute}', truth_value)

            self.assertEqual(test_post.is_good, False)

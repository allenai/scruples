"""Tests for socialnorms.data.post_types."""

import unittest

from socialnorms.data import post_types


class PostTypeTestCase(unittest.TestCase):
    """Test socialnorms.data.post_types.PostType."""

    POST_TYPE_TO_PHRASES = {
        post_types.PostType.HISTORICAL: [
            'am I the asshole',
            'am I the ahole',
            'am I the a-hole'
        ],
        post_types.PostType.HYPOTHETICAL: [
            'would I be the asshole',
            'would I be the ahole',
            'would I be the a-hole'
        ],
        post_types.PostType.META: [
            '[META]'
        ]
    }
    """A dictionary mapping post types to phrases that signify them."""

    # test extract_from_title

    def test_extract_from_title_on_initialisms(self):
        for post_type in post_types.PostType:
            # when the initialism starts the text
            self.assertEqual(
                post_types.PostType.extract_from_title(
                    f'{post_type.reddit_name} if...'),
                post_type)
            # when the initialism ends the text
            self.assertEqual(
                post_types.PostType.extract_from_title(
                    f'If I... then {post_type.reddit_name}'),
                post_type)
            # when the initialism is in the middle of the text
            self.assertEqual(
                post_types.PostType.extract_from_title(
                    f"If I... {post_type.reddit_name}, what do you think?"),
                post_type)
            # "meta" is too common a word so it's abbreviation is only
            # used to identify a post if the abbreviation is all caps.
            # Skip the capitalization tests if the post_type is META.
            if post_type != post_types.PostType.META:
                # when the initialism is uppercased
                self.assertEqual(
                    post_types.PostType.extract_from_title(
                        f'If I... {post_type.reddit_name.upper()}?'),
                    post_type)
                # when the initialism is lowercased
                self.assertEqual(
                    post_types.PostType.extract_from_title(
                        f'If I... {post_type.reddit_name.lower()}?'),
                    post_type)
                # when the initialism is capitalized
                self.assertEqual(
                    post_types.PostType.extract_from_title(
                        f'If I... {post_type.reddit_name.lower().capitalize()}?'),
                    post_type)

    def test_extract_from_title_on_phrases(self):
        for post_type, phrases in self.POST_TYPE_TO_PHRASES.items():
            for phrase in phrases:
                # when the phrase starts the text
                self.assertEqual(
                    post_types.PostType.extract_from_title(
                        f"{phrase} if I..."),
                    post_type)
                # when the phrase ends the text
                self.assertEqual(
                    post_types.PostType.extract_from_title(
                        f"If I... then {phrase}"),
                    post_type)
                # when the phrase is in the middle of the text
                self.assertEqual(
                    post_types.PostType.extract_from_title(
                        f"If I... {phrase}, what do you think?"),
                    post_type)
                # when the phrase is uppercased
                self.assertEqual(
                    post_types.PostType.extract_from_title(
                        f'If I... {phrase.upper()}?'),
                    post_type)
                # when the phrase is lowercased
                self.assertEqual(
                    post_types.PostType.extract_from_title(
                        f'If I... {phrase.lower()}?'),
                    post_type)
                # when the phrase is capitalized
                self.assertEqual(
                    post_types.PostType.extract_from_title(
                        f'If I... {phrase.lower().capitalize()}?'),
                    post_type)

    def test_extract_from_title_on_ambiguous_cases(self):
        self.assertEqual(
            post_types.PostType.extract_from_title(
                "AITA/WIBTA if..."),
            None)

    def test_extract_from_title_doesnt_extract_spurious_labels(self):
        self.assertEqual(
            post_types.PostType.extract_from_title("I'd like to know..."),
            None)

    # test in_

    def test_in__on_initialisms(self):
        for post_type1 in post_types.PostType:
            for post_type2 in post_types.PostType:
                if post_type1 == post_type2:
                    continue
                # when the initialism starts the text
                self.assertTrue(
                    post_type1.in_(f'{post_type1.reddit_name} if I...'))
                self.assertFalse(
                    post_type2.in_(f'{post_type1.reddit_name} if I...'))
                # when the initialism ends the text
                self.assertTrue(
                    post_type1.in_(f'If I... {post_type1.reddit_name}'))
                self.assertFalse(
                    post_type2.in_(f'If I... {post_type1.reddit_name}'))
                # when the initialism is in the middle of the text
                self.assertTrue(
                    post_type1.in_(
                        f'If I... {post_type1.reddit_name}, what do you'
                        f' think?'))
                self.assertFalse(
                    post_type2.in_(
                        f'If I... {post_type1.reddit_name}, what do you'
                        f' think?'))
                # "meta" is too common a word so it's abbreviation is only
                # used to identify a post if the abbreviation is all caps.
                # Skip the capitalization tests if post_type1 is META.
                if post_type1 != post_types.PostType.META:
                    # when the initialism is uppercased
                    self.assertTrue(
                        post_type1.in_(
                            f'If I... {post_type1.reddit_name.upper()}?'))
                    self.assertFalse(
                        post_type2.in_(
                            f'If I... {post_type1.reddit_name.upper()}?'))
                    # when the initialism is lowercased
                    self.assertTrue(
                        post_type1.in_(
                            f'If I... {post_type1.reddit_name.lower()}?'))
                    self.assertFalse(
                        post_type2.in_(
                            f'If I... {post_type1.reddit_name.lower()}?'))
                    # when the initialism is capitalized
                    self.assertTrue(
                        post_type1.in_(
                            f'If I...'
                            f' {post_type1.reddit_name.lower().capitalize()}?'))
                    self.assertFalse(
                        post_type2.in_(
                            f'If I...'
                            f' {post_type1.reddit_name.lower().capitalize()}?'))

    def test_in__on_phrases(self):
        for post_type1 in post_types.PostType:
            for post_type2 in post_types.PostType:
                if post_type1 == post_type2:
                    continue
                for phrase in self.POST_TYPE_TO_PHRASES[post_type1]:
                    # when the phrase starts the text
                    self.assertTrue(
                        post_type1.in_(f"{phrase} if I..."))
                    self.assertFalse(
                        post_type2.in_(f"{phrase} if I..."))
                    # when the phrase ends the text
                    self.assertTrue(
                        post_type1.in_(f"If I... {phrase}"))
                    self.assertFalse(
                        post_type2.in_(f"If I... {phrase}"))
                    # when the phrase is in the middle of the text
                    self.assertTrue(
                        post_type1.in_(
                            f"If I... {phrase}, what do you think?"))
                    self.assertFalse(
                        post_type2.in_(
                            f"If I... {phrase}, what do you think?"))
                    # when the phrase is uppercased
                    self.assertTrue(
                        post_type1.in_(f'If I... {phrase.upper()}?'))
                    self.assertFalse(
                        post_type2.in_(f'If I... {phrase.upper()}?'))
                    # when the phrase is lowercased
                    self.assertTrue(
                        post_type1.in_(f'If I... {phrase.lower()}?'))
                    self.assertFalse(
                        post_type2.in_(f'If I... {phrase.lower()}?'))
                    # when the phrase is capitalized
                    self.assertTrue(
                        post_type1.in_(
                            f'If I... {phrase.lower().capitalize()}?'))
                    self.assertFalse(
                        post_type2.in_(
                            f'If I... {phrase.lower().capitalize()}?'))

    def test_in__doesnt_return_true_on_spurious_titles(self):
        for post_type in post_types.PostType:
            self.assertFalse(post_type.in_('A random title.'))

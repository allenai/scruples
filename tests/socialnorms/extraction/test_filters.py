"""Tests for socialnorms.extraction.filters."""

import unittest

from socialnorms.extraction import filters


class EmptyStringFilterTestCase(unittest.TestCase):
    """Test socialnorms.extraction.filters.EmptyStringFilter."""

    def test_instantiates_without_arguments(self):
        filters.EmptyStringFilter()

    def test_is_true_on_empty_string(self):
        self.assertTrue(filters.EmptyStringFilter()(''))

    def test_is_false_on_non_empty_string(self):
        self.assertFalse(filters.EmptyStringFilter()('Not empty.'))

    def test_is_false_on_non_string_input(self):
        # on None
        self.assertFalse(filters.EmptyStringFilter()(None))

        # on booleans
        self.assertFalse(filters.EmptyStringFilter()(True))
        self.assertFalse(filters.EmptyStringFilter()(False))

        # on integers
        self.assertFalse(filters.EmptyStringFilter()(0))
        self.assertFalse(filters.EmptyStringFilter()(1))
        self.assertFalse(filters.EmptyStringFilter()(-1))


class TooFewCharactersFilterTestCase(unittest.TestCase):
    """Test socialnorms.extraction.filters.TooFewCharactersFilter."""

    def test_requires_min_chars_argument(self):
        with self.assertRaises(TypeError):
            filters.TooFewCharactersFilter()

    def test_accepts_min_chars_argument(self):
        filters.TooFewCharactersFilter(min_chars=10)

    def test_returns_true_if_fewer_than_min_chars(self):
        self.assertTrue(filters.TooFewCharactersFilter(min_chars=10)('x' * 9))

    def test_returns_false_if_as_many_or_more_than_min_chars(self):
        self.assertFalse(filters.TooFewCharactersFilter(min_chars=7)('x' * 7))
        self.assertFalse(filters.TooFewCharactersFilter(min_chars=7)('x' * 8))


class TooFewWordsFilterTestCase(unittest.TestCase):
    """Test socialnorms.extraction.filters.TooFewWordsFilter."""

    def test_requires_min_words_argument(self):
        with self.assertRaises(TypeError):
            filters.TooFewWordsFilter()

    def test_accepts_min_words_argument(self):
        filters.TooFewWordsFilter(min_words=10)

    def test_returns_true_if_fewer_than_min_words(self):
        self.assertTrue(
            filters.TooFewWordsFilter(min_words=3)('two words'))

    def test_returns_false_if_as_many_or_more_than_min_words(self):
        self.assertFalse(
            filters.TooFewWordsFilter(min_words=3)('Three words here'))
        self.assertFalse(
            filters.TooFewWordsFilter(min_words=3)(
                'More than three words here'))


class PrefixFilterTestCase(unittest.TestCase):
    """Test socialnorms.extraction.filters.PrefixFilter."""

    def test_requires_prefix_argument(self):
        with self.assertRaises(TypeError):
            filters.PrefixFilter()

    def test_accepts_prefix_argument(self):
        filters.PrefixFilter(prefix='aaa')

    def test_accepts_optional_case_sensitive_argument(self):
        filters.PrefixFilter(prefix='aaa')
        filters.PrefixFilter(prefix='aaa', case_sensitive=False)

    def test_matches_empty_prefix(self):
        self.assertTrue(filters.PrefixFilter(prefix='')('a word'))

    def test_matches_strings_correctly(self):
        prefix_filter = filters.PrefixFilter(prefix='abc')

        # when prefix matches exactly
        self.assertTrue(prefix_filter('abc'))

        # when prefix matches the input's prefix
        self.assertTrue(prefix_filter('abcdef'))

        # when prefix does not match at all
        self.assertFalse(prefix_filter('ieikdkfk'))

        # when prefix matches suffix
        self.assertFalse(prefix_filter('1abc'))

        # when string is a partial match to the prefix
        self.assertFalse(prefix_filter('ab'))

        # when run against the empty string
        self.assertFalse(prefix_filter(''))

    def test_case_sensitive_matching(self):
        # when prefix and input are capitalized the same
        self.assertTrue(filters.PrefixFilter(
            prefix='Abc',
            case_sensitive=True)('Abcd'))
        self.assertTrue(filters.PrefixFilter(
            prefix='abc', case_sensitive=True)('abcd'))
        self.assertTrue(filters.PrefixFilter(
            prefix='ABC', case_sensitive=True)('ABCD'))

        # when prefix is capitalized but input is not
        self.assertFalse(filters.PrefixFilter(
            prefix='Abc', case_sensitive=True)('abcd'))
        self.assertFalse(filters.PrefixFilter(
            prefix='ABC', case_sensitive=True)('abcd'))

        # when input is capitalized but prefix is not
        self.assertFalse(filters.PrefixFilter(
            prefix='abc', case_sensitive=True)('Abcd'))
        self.assertFalse(filters.PrefixFilter(
            prefix='abc', case_sensitive=True)('ABCD'))

    def test_case_insensitive_matching(self):
        # when prefix and input are capitalized the same
        self.assertTrue(filters.PrefixFilter(prefix='Abc')('Abcd'))
        self.assertTrue(filters.PrefixFilter(prefix='abc')('abcd'))
        self.assertTrue(filters.PrefixFilter(prefix='ABC')('ABCD'))

        # when prefix is capitalized but input is not
        self.assertTrue(filters.PrefixFilter(prefix='Abc')('abcd'))
        self.assertTrue(filters.PrefixFilter(prefix='ABC')('abcd'))

        # when input is capitalized but prefix is not
        self.assertTrue(filters.PrefixFilter(prefix='abc')('Abcd'))
        self.assertTrue(filters.PrefixFilter(prefix='abc')('ABCD'))


class StartsWithGerundFilterTestCase(unittest.TestCase):
    """Test socialnorms.extraction.filters.StartsWithGerundFilter."""

    def test_instantiates_without_arguments(self):
        filters.StartsWithGerundFilter()

    def test_is_false_if_gerund_is_in_first_three_words(self):
        self.assertFalse(filters.StartsWithGerundFilter()(
            'going to the store.'))
        self.assertFalse(filters.StartsWithGerundFilter()(
            'always going to the store.'))
        self.assertFalse(filters.StartsWithGerundFilter()(
            'always not going to the store.'))

    def test_is_true_if_no_gerund_in_first_three_words(self):
        self.assertTrue(filters.StartsWithGerundFilter()(''))
        self.assertTrue(filters.StartsWithGerundFilter()(
            'foo.'))
        self.assertTrue(filters.StartsWithGerundFilter()(
            'foo bar.'))
        self.assertTrue(filters.StartsWithGerundFilter()(
            'foo bar baz.'))

    def test_does_not_filter_dashed_gerunds(self):
        self.assertFalse(filters.StartsWithGerundFilter()(
            're-taking a test at school'))
        self.assertFalse(filters.StartsWithGerundFilter()(
            'never re-taking a test at school'))

    def test_does_not_filter_quoted_gerunds(self):
        self.assertFalse(filters.StartsWithGerundFilter()(
            "'running' down the hallway"))
        self.assertFalse(filters.StartsWithGerundFilter()(
            "not 'running' while crossing the street"))
        self.assertFalse(filters.StartsWithGerundFilter()(
            "'not running' while crossing the street"))
        self.assertFalse(filters.StartsWithGerundFilter()(
            '"running" down the hallway'))
        self.assertFalse(filters.StartsWithGerundFilter()(
            'not "running" while crossing the street'))
        self.assertFalse(filters.StartsWithGerundFilter()(
            '"not running" while crossing the street'))

    def test_does_not_filter_when_preceding_words_are_quoted(self):
        self.assertFalse(filters.StartsWithGerundFilter()(
            '"not" running down the hallway'))
        self.assertFalse(filters.StartsWithGerundFilter()(
            '"always" not running down the hallway'))
        self.assertFalse(filters.StartsWithGerundFilter()(
            '"always not" running down the hallway'))
        self.assertFalse(filters.StartsWithGerundFilter()(
            "'not' running down the hallway"))
        self.assertFalse(filters.StartsWithGerundFilter()(
            "'always' not running down the hallway"))
        self.assertFalse(filters.StartsWithGerundFilter()(
            "'always not' running down the hallway"))

    def test_does_not_filter_when_preceding_words_are_bracketed(self):
        self.assertFalse(filters.StartsWithGerundFilter()(
            '(not) running down the hallway'))
        self.assertFalse(filters.StartsWithGerundFilter()(
            '(always) not running down the hallway'))
        self.assertFalse(filters.StartsWithGerundFilter()(
            '(always not) running down the hallway'))
        self.assertFalse(filters.StartsWithGerundFilter()(
            '[not] running down the hallway'))
        self.assertFalse(filters.StartsWithGerundFilter()(
            '[always] not running down the hallway'))
        self.assertFalse(filters.StartsWithGerundFilter()(
            '[always not] running down the hallway'))
        self.assertFalse(filters.StartsWithGerundFilter()(
            '{not} running down the hallway'))
        self.assertFalse(filters.StartsWithGerundFilter()(
            '{always} not running down the hallway'))
        self.assertFalse(filters.StartsWithGerundFilter()(
            '{always not} running down the hallway'))


class WhWordFilterTestCase(unittest.TestCase):
    """Test socialnorms.extraction.filters.WhWordFilter."""

    def test_instantiates_without_arguments(self):
        filters.WhWordFilter()

    def test_is_true_if_starts_with_wh_word(self):
        self.assertTrue(filters.WhWordFilter()('why foo'))
        self.assertTrue(filters.WhWordFilter()('why foo bar'))
        self.assertTrue(filters.WhWordFilter()('who foo'))
        self.assertTrue(filters.WhWordFilter()('who foo bar'))
        self.assertTrue(filters.WhWordFilter()('which foo'))
        self.assertTrue(filters.WhWordFilter()('which foo bar'))
        self.assertTrue(filters.WhWordFilter()('what foo'))
        self.assertTrue(filters.WhWordFilter()('what foo bar'))
        self.assertTrue(filters.WhWordFilter()('where foo'))
        self.assertTrue(filters.WhWordFilter()('where foo bar'))
        self.assertTrue(filters.WhWordFilter()('when foo'))
        self.assertTrue(filters.WhWordFilter()('when foo bar'))
        self.assertTrue(filters.WhWordFilter()('how foo'))
        self.assertTrue(filters.WhWordFilter()('how foo bar'))

    def test_wh_words_are_matched_case_insensitively(self):
        # when the word is capitalized
        self.assertTrue(filters.WhWordFilter()('Why foo'))
        self.assertTrue(filters.WhWordFilter()('Why foo bar'))
        self.assertTrue(filters.WhWordFilter()('Who foo'))
        self.assertTrue(filters.WhWordFilter()('Who foo bar'))
        self.assertTrue(filters.WhWordFilter()('Which foo'))
        self.assertTrue(filters.WhWordFilter()('Which foo bar'))
        self.assertTrue(filters.WhWordFilter()('What foo'))
        self.assertTrue(filters.WhWordFilter()('What foo bar'))
        self.assertTrue(filters.WhWordFilter()('Where foo'))
        self.assertTrue(filters.WhWordFilter()('Where foo bar'))
        self.assertTrue(filters.WhWordFilter()('When foo'))
        self.assertTrue(filters.WhWordFilter()('When foo bar'))
        self.assertTrue(filters.WhWordFilter()('How foo'))
        self.assertTrue(filters.WhWordFilter()('How foo bar'))

        # when the word is all caps
        self.assertTrue(filters.WhWordFilter()('WHY foo'))
        self.assertTrue(filters.WhWordFilter()('WHY foo bar'))
        self.assertTrue(filters.WhWordFilter()('WHO foo'))
        self.assertTrue(filters.WhWordFilter()('WHO foo bar'))
        self.assertTrue(filters.WhWordFilter()('WHICH foo'))
        self.assertTrue(filters.WhWordFilter()('WHICH foo bar'))
        self.assertTrue(filters.WhWordFilter()('WHAT foo'))
        self.assertTrue(filters.WhWordFilter()('WHAT foo bar'))
        self.assertTrue(filters.WhWordFilter()('WHERE foo'))
        self.assertTrue(filters.WhWordFilter()('WHERE foo bar'))
        self.assertTrue(filters.WhWordFilter()('WHEN foo'))
        self.assertTrue(filters.WhWordFilter()('WHEN foo bar'))
        self.assertTrue(filters.WhWordFilter()('HOW foo'))
        self.assertTrue(filters.WhWordFilter()('HOW foo bar'))

    def test_is_false_if_does_not_start_with_wh_word(self):
        self.assertFalse(filters.WhWordFilter()('foo'))
        self.assertFalse(filters.WhWordFilter()('foo bar'))
        self.assertFalse(filters.WhWordFilter()('This string is an example'))

    def test_is_false_even_if_first_word_has_wh_word_like_prefix(self):
        self.assertFalse(filters.WhWordFilter()('howling at the moon'))
        self.assertFalse(filters.WhWordFilter()('wholly singing my heart out'))

    def test_matches_quoted_wh_words(self):
        self.assertTrue(filters.WhWordFilter()('"who" foo'))
        self.assertTrue(filters.WhWordFilter()("'who' foo"))

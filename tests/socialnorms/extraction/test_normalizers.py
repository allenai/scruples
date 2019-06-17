"""Tests for socialnorms.extraction.normalizers."""

import unittest
from unittest import mock

from socialnorms.extraction import normalizers


class ComposedNormalizerTestCase(unittest.TestCase):
    """Test socialnorms.extraction.normalizers.ComposedNormalizer."""

    def test_requires_normalizers_argument(self):
        with self.assertRaises(TypeError):
            normalizers.ComposedNormalizer()

    def test_accepts_normalizers_argument(self):
        normalizers.ComposedNormalizer(normalizers=[])

    def test_acts_as_identity_with_empty_normalizers_list(self):
        identity = normalizers.ComposedNormalizer(normalizers=[])

        self.assertEqual(identity(''), '')
        self.assertEqual(identity('foo'), 'foo')
        self.assertEqual(identity('bar'), 'bar')

    def test_composes_functions(self):
        mock1 = mock.MagicMock(return_value='bar')
        mock2 = mock.MagicMock(return_value='baz')

        composed_normalizer = normalizers.ComposedNormalizer(
            normalizers=[mock1, mock2])

        self.assertEqual(composed_normalizer('foo'), 'baz')

        mock1.assert_called_with('foo')
        mock2.assert_called_with('bar')


class FixTextNormalizerTestCase(unittest.TestCase):
    """Test socialnorms.extraction.normalizers.FixTextNormalizer."""

    def test_calls_fix_text(self):
        with mock.patch(
                'socialnorms.extraction.normalizers.ftfy.fix_text',
                return_value='mocked'
        ):
            self.assertEqual(
                normalizers.FixTextNormalizer()('some text'),
                'mocked')


class GonnaGottaWannaNormalizerTestCase(unittest.TestCase):
    """Test socialnorms.extraction.normalizers.GonnaGottaWannaNormalizer."""

    def test_leaves_normal_text_alone(self):
        f = normalizers.GonnaGottaWannaNormalizer()

        self.assertEqual(f(''), '')
        self.assertEqual(f('foo'), 'foo')
        self.assertEqual(f('foo bar'), 'foo bar')
        self.assertEqual(
            f('Some normal text, i.e. a typical example.'),
              'Some normal text, i.e. a typical example.')

    def test_makes_replacements(self):
        f = normalizers.GonnaGottaWannaNormalizer()

        # gonna
        self.assertEqual(
            f("I'm gonna go to the store"),
              "I'm going to go to the store")
        self.assertEqual(
            f("I'm gonna run across the room"),
              "I'm going to run across the room")
        # gotta
        self.assertEqual(
            f("I gotta go to the store"),
              'I got to go to the store')
        self.assertEqual(
            f("I gotta run across the room"),
              'I got to run across the room')
        # wanna
        self.assertEqual(
            f('I wanna go to the store'),
              'I want to go to the store')
        self.assertEqual(
            f('I wanna run across the room'),
              'I want to run across the room')
        # coulda
        self.assertEqual(
            f('I coulda gone to the store'),
              'I could have gone to the store')
        self.assertEqual(
            f('I coulda run across the room'),
              'I could have run across the room')
        # woulda
        self.assertEqual(
            f('I woulda gone to the store'),
              'I would have gone to the store')
        self.assertEqual(
            f('I woulda run across the room'),
              'I would have run across the room')
        # shoulda
        self.assertEqual(
            f('I shoulda gone to the store'),
              'I should have gone to the store')
        self.assertEqual(
            f('I shoulda run across the room'),
              'I should have run across the room')

    def test_makes_multiple_simultaneous_replacements(self):
        f = normalizers.GonnaGottaWannaNormalizer()

        self.assertEqual(
            f("I coulda gone to the store, but I didn't so I'm gonna."),
              "I could have gone to the store, but I didn't so I'm going to.")
        self.assertEqual(
            f("I gotta do it so I'm gonna do it."),
              "I got to do it so I'm going to do it.")


class RemoveAgeGenderMarkersNormalizerTestCase(unittest.TestCase):
    """Test socialnorms.extraction.normalizers.RemoveAgeGenderMarkersNormalizer."""

    def test_leaves_normal_text_alone(self):
        f = normalizers.RemoveAgeGenderMarkersNormalizer()

        self.assertEqual(f(''), '')
        self.assertEqual(f('foo'), 'foo')
        self.assertEqual(f('foo bar'), 'foo bar')
        self.assertEqual(
            f('Some normal text, i.e. a typical example.'),
              'Some normal text, i.e. a typical example.')

    def test_leaves_parenthesis_without_age_gender_markers_alone(self):
        f = normalizers.RemoveAgeGenderMarkersNormalizer()

        # single word in parentheses
        self.assertEqual(f('foo (bar)'), 'foo (bar)')

        # multiple words in parentheses
        self.assertEqual(f('foo (bar baz)'), 'foo (bar baz)')

    def test_leaves_non_age_numbers_alone(self):
        f = normalizers.RemoveAgeGenderMarkersNormalizer()

        # common years (without parentheses)
        self.assertEqual(f('1970'), '1970')
        self.assertEqual(f('2019'), '2019')

        # common years (with parentheses)
        self.assertEqual(f('(1970)'), '(1970)')
        self.assertEqual(f('(2019)'), '(2019)')

    def test_removes_ages_in_parentheses(self):
        f = normalizers.RemoveAgeGenderMarkersNormalizer()

        self.assertEqual(f('foo (20)'), 'foo')
        self.assertEqual(f('bar [19] baz'), 'bar baz')
        self.assertEqual(f('foo {7} and bar {4}'), 'foo and bar')

    def test_removes_genders_in_parentheses(self):
        f = normalizers.RemoveAgeGenderMarkersNormalizer()

        self.assertEqual(f('foo (m)'), 'foo')
        self.assertEqual(f('bar [f] baz'), 'bar baz')
        self.assertEqual(f('foo {m} and bar {f}'), 'foo and bar')

    def test_removes_age_and_gender_in_parentheses(self):
        f = normalizers.RemoveAgeGenderMarkersNormalizer()

        # age then gender
        #   upper case gender
        self.assertEqual(f('foo (27M) and bar (39F).'), 'foo and bar.')
        #   lower case gender
        self.assertEqual(f('foo (27m) and bar (39f).'), 'foo and bar.')
        #   alternative brackets
        self.assertEqual(f('foo {27M} and bar [39F].'), 'foo and bar.')
        #   optional space or punctuation
        self.assertEqual(f('foo (27 M) and bar (39 F).'), 'foo and bar.')
        self.assertEqual(f('foo (27, M) and bar (39, F).'), 'foo and bar.')
        self.assertEqual(f('foo (27.M) and bar (39.F).'), 'foo and bar.')
        self.assertEqual(f('foo (27:M) and bar (39:F).'), 'foo and bar.')

        # gender then age
        #   upper case gender
        self.assertEqual(f('foo (M27) and bar (F39).'), 'foo and bar.')
        #   lower case gender
        self.assertEqual(f('foo (m27) and bar (f39).'), 'foo and bar.')
        #   alternative brackets
        self.assertEqual(f('foo {M27} and bar [F39].'), 'foo and bar.')
        #   optional space or punctuation
        self.assertEqual(f('foo (M 27) and bar (F 39).'), 'foo and bar.')
        self.assertEqual(f('foo (M, 27) and bar (F, 39).'), 'foo and bar.')
        self.assertEqual(f('foo (M.27) and bar (F.39).'), 'foo and bar.')
        self.assertEqual(f('foo (M:27) and bar (F:39).'), 'foo and bar.')

    def test_removes_age_and_gender_without_parentheses(self):
        f = normalizers.RemoveAgeGenderMarkersNormalizer()

        # age then gender
        #   upper case gender
        self.assertEqual(f('foo 27M and bar 39F.'), 'foo and bar.')
        #   lower case gender
        self.assertEqual(f('foo 27m and bar 39f.'), 'foo and bar.')

        # gender then age
        #   upper case gender
        self.assertEqual(f('foo M27 and bar F39.'), 'foo and bar.')
        #   lower case gender
        self.assertEqual(f('foo m27 and bar f39.'), 'foo and bar.')


class StripWhitespaceAndPunctuationNormalizerTestCase(unittest.TestCase):
    """Test socialnorms.extraction.normalizers.StripWhitespaceAndPunctuationNormalizer."""

    def test_leaves_normal_text_alone(self):
        f = normalizers.StripWhitespaceAndPunctuationNormalizer()

        self.assertEqual(f('foo'), 'foo')
        self.assertEqual(f('foo bar'), 'foo bar')

    def test_leaves_internal_whitespace_alone(self):
        f = normalizers.StripWhitespaceAndPunctuationNormalizer()

        self.assertEqual(
            f('Some text  with   odd\n internal whitespace'),
              'Some text  with   odd\n internal whitespace')

    def test_strips_whitespace(self):
        f = normalizers.StripWhitespaceAndPunctuationNormalizer()

        # leading whitespace
        self.assertEqual(f(' foo'), 'foo')
        self.assertEqual(f(' foo bar'), 'foo bar')
        self.assertEqual(f('\tfoo'), 'foo')
        self.assertEqual(f('\tfoo bar'), 'foo bar')

        # trailing whitespace
        self.assertEqual(f('foo '), 'foo')
        self.assertEqual(f('foo bar '), 'foo bar')
        self.assertEqual(f('foo\t'), 'foo')
        self.assertEqual(f('foo bar\t'), 'foo bar')

        # leading and trailing whitespace
        self.assertEqual(f(' foo '), 'foo')
        self.assertEqual(f(' foo bar '), 'foo bar')
        self.assertEqual(f(' foo\t'), 'foo')
        self.assertEqual(f(' foo bar\t'), 'foo bar')

    def test_strips_punctuation(self):
        f = normalizers.StripWhitespaceAndPunctuationNormalizer()

        # leading punctuation
        self.assertEqual(f('-foo'), 'foo')
        self.assertEqual(f(':foo bar'), 'foo bar')
        self.assertEqual(f('_foo'), 'foo')
        self.assertEqual(f('.foo bar'), 'foo bar')

        # trailing punctuation
        self.assertEqual(f('foo.'), 'foo')
        self.assertEqual(f('foo bar!'), 'foo bar')
        self.assertEqual(f('foo?'), 'foo')
        self.assertEqual(f('foo bar@'), 'foo bar')

        # leading and trailing punctuation
        self.assertEqual(f('-foo.'), 'foo')
        self.assertEqual(f(':foo bar!'), 'foo bar')
        self.assertEqual(f('#foo&'), 'foo')
        self.assertEqual(f('^foo bar$'), 'foo bar')

    def test_strips_whitespace_and_punctuation(self):
        f = normalizers.StripWhitespaceAndPunctuationNormalizer()

        # leading whitespace and punctuation
        self.assertEqual(f('- foo'), 'foo')
        self.assertEqual(f(': foo bar'), 'foo bar')
        self.assertEqual(f(' _\tfoo'), 'foo')
        self.assertEqual(f(' . foo bar'), 'foo bar')

        # trailing punctuation
        self.assertEqual(f('foo. '), 'foo')
        self.assertEqual(f('foo bar !'), 'foo bar')
        self.assertEqual(f('foo ? '), 'foo')
        self.assertEqual(f('foo bar\t@ '), 'foo bar')

        # leading and trailing punctuation
        self.assertEqual(f('- foo.'), 'foo')
        self.assertEqual(f(' :foo bar !\t\t'), 'foo bar')
        self.assertEqual(f('#foo\t&'), 'foo')
        self.assertEqual(f(' ^  foo bar    $'), 'foo bar')

    def test_does_not_strip_matched_punctuation(self):
        f = normalizers.StripWhitespaceAndPunctuationNormalizer()

        # quotes
        #   leading and trailing
        #     " character
        self.assertEqual(f('"foo"'), '"foo"')
        self.assertEqual(f('"foo bar"'), '"foo bar"')
        #     ' character
        self.assertEqual(f("'foo'"), "'foo'")
        self.assertEqual(f("'foo bar'"), "'foo bar'")
        #   leading only
        #     " character
        self.assertEqual(f('"foo" bar'), '"foo" bar')
        #     ' character
        self.assertEqual(f("'foo' bar"), "'foo' bar")
        #   trailing only
        #     " character
        self.assertEqual(f('foo "bar"'), 'foo "bar"')
        #     ' character
        self.assertEqual(f("foo 'bar'"), "foo 'bar'")

        # parentheses
        #   leading and trailing
        #     () characters
        self.assertEqual(f('(foo)'), '(foo)')
        self.assertEqual(f('(foo bar)'), '(foo bar)')
        #     [] characters
        self.assertEqual(f("[foo]"), "[foo]")
        self.assertEqual(f("[foo bar]"), "[foo bar]")
        #   leading only
        #     () characters
        self.assertEqual(f('(foo) bar'), '(foo) bar')
        #     [] characters
        self.assertEqual(f("[foo] bar"), "[foo] bar")
        #   trailing only
        #     () characters
        self.assertEqual(f('foo (bar)'), 'foo (bar)')
        #     [] characters
        self.assertEqual(f("foo [bar]"), "foo [bar]")


class StripMatchedPunctuationNormalizerTestCase(unittest.TestCase):
    """Test socialnorms.extraction.normalizers.StripMatchedPunctuationNormalizer."""

    def test_leaves_normal_text_alone(self):
        f = normalizers.StripMatchedPunctuationNormalizer()

        self.assertEqual(f('foo'), 'foo')
        self.assertEqual(f('foo bar'), 'foo bar')

    def test_leaves_leading_and_trailing_punctuation_alone(self):
        f = normalizers.StripMatchedPunctuationNormalizer()

        # leading
        self.assertEqual(f(':foo'), ':foo')
        self.assertEqual(f(';foo bar'), ';foo bar')

        # trailing
        self.assertEqual(f('foo.'), 'foo.')
        self.assertEqual(f('foo bar!'), 'foo bar!')

        # leading and trailing
        self.assertEqual(f('.foo,'), '.foo,')
        self.assertEqual(f('- foo bar!'), '- foo bar!')

    def test_leaves_unmatched_punctuation_alone(self):
        f = normalizers.StripMatchedPunctuationNormalizer()

        # quotes
        #   " character
        self.assertEqual(f('"foo'), '"foo')
        self.assertEqual(f('foo"'), 'foo"')
        self.assertEqual(f('"foo bar'), '"foo bar')
        self.assertEqual(f('foo bar"'), 'foo bar"')
        #   ' character
        self.assertEqual(f("'foo"), "'foo")
        self.assertEqual(f("foo'"), "foo'")
        self.assertEqual(f("'foo bar"), "'foo bar")
        self.assertEqual(f("foo bar'"), "foo bar'")

        # brackets
        #   () characters
        self.assertEqual(f('(foo'), '(foo')
        self.assertEqual(f('foo)'), 'foo)')
        self.assertEqual(f('(foo bar'), '(foo bar')
        self.assertEqual(f('foo bar)'), 'foo bar)')
        #   [] characters
        self.assertEqual(f('[foo'), '[foo')
        self.assertEqual(f('foo]'), 'foo]')
        self.assertEqual(f('[foo bar'), '[foo bar')
        self.assertEqual(f('foo bar]'), 'foo bar]')
        #   {} characters
        self.assertEqual(f('{foo'), '{foo')
        self.assertEqual(f('foo}'), 'foo}')
        self.assertEqual(f('{foo bar'), '{foo bar')
        self.assertEqual(f('foo bar}'), 'foo bar}')

    def test_leaves_matched_punctuation_not_wrapping_text_alone(self):
        f = normalizers.StripMatchedPunctuationNormalizer()

        # quotes
        #   " character
        self.assertEqual(f('"foo" bar'), '"foo" bar')
        self.assertEqual(f('foo "bar"'), 'foo "bar"')
        #   ' character
        self.assertEqual(f("'foo' bar"), "'foo' bar")
        self.assertEqual(f("foo 'bar'"), "foo 'bar'")

        # brackets
        #   () characters
        self.assertEqual(f('(foo) bar'), '(foo) bar')
        self.assertEqual(f('foo (bar)'), 'foo (bar)')
        #   [] characters
        self.assertEqual(f('[foo] bar'), '[foo] bar')
        self.assertEqual(f('foo [bar]'), 'foo [bar]')
        #   {} characters
        self.assertEqual(f('{foo} bar'), '{foo} bar')
        self.assertEqual(f('foo {bar}'), 'foo {bar}')

    def test_removes_matched_punctuation(self):
        f = normalizers.StripMatchedPunctuationNormalizer()

        # quotes
        #   " character
        self.assertEqual(f('"foo"'), 'foo')
        self.assertEqual(f('"foo bar"'), 'foo bar')
        #   ' character
        self.assertEqual(f("'foo'"), 'foo')
        self.assertEqual(f("'foo bar'"), 'foo bar')

        # brackets
        #   () characters
        self.assertEqual(f('(foo)'), 'foo')
        self.assertEqual(f('(foo bar)'), 'foo bar')
        #   [] characters
        self.assertEqual(f('[foo]'), 'foo')
        self.assertEqual(f('[foo bar]'), 'foo bar')
        #   {} characters
        self.assertEqual(f('{foo}'), 'foo')
        self.assertEqual(f('{foo bar}'), 'foo bar')


class StripLeadingAndTrailingParentheticalsNormalizerTestCase(unittest.TestCase):
    """Test socialnorms.extraction.normalizers.StripLeadingAndTrailingParentheticalsNormalizer."""

    def test_leaves_normal_text_alone(self):
        f = normalizers.StripLeadingAndTrailingParentheticalsNormalizer()

        self.assertEqual(f('foo'), 'foo')
        self.assertEqual(f('foo bar'), 'foo bar')

    def test_leaves_internal_parentheticals_alone(self):
        f = normalizers.StripLeadingAndTrailingParentheticalsNormalizer()

        # with whitespace
        self.assertEqual(f('foo (bar) baz'), 'foo (bar) baz')

        # without whitespace
        self.assertEqual(f('foo (bar)baz'), 'foo (bar)baz')
        self.assertEqual(f('foo(bar) baz'), 'foo(bar) baz')
        self.assertEqual(f('foo(bar)baz'), 'foo(bar)baz')

    def test_when_strip_leading_is_false(self):
        f = normalizers.StripLeadingAndTrailingParentheticalsNormalizer(
            strip_leading=False)

        # test that it does not strip leading
        self.assertEqual(f('(foo) bar'), '(foo) bar')
        self.assertEqual(f('(foo bar) baz'), '(foo bar) baz')
        self.assertEqual(f('(foo)bar'), '(foo)bar')
        self.assertEqual(f('(foo bar)baz'), '(foo bar)baz')

        # test that it does strip trailing
        self.assertEqual(f('foo (bar)'), 'foo')
        self.assertEqual(f('foo (bar baz)'), 'foo')
        self.assertEqual(f('foo(bar)'), 'foo')
        self.assertEqual(f('foo(bar baz)'), 'foo')

    def test_when_strip_trailing_is_false(self):
        f = normalizers.StripLeadingAndTrailingParentheticalsNormalizer(
            strip_trailing=False)

        # test that it does strip leading
        self.assertEqual(f('(foo) bar'), 'bar')
        self.assertEqual(f('(foo bar) baz'), 'baz')
        self.assertEqual(f('(foo)bar'), 'bar')
        self.assertEqual(f('(foo bar)baz'), 'baz')

        # test that it does not strip trailing
        self.assertEqual(f('foo (bar)'), 'foo (bar)')
        self.assertEqual(f('foo (bar baz)'), 'foo (bar baz)')
        self.assertEqual(f('foo(bar)'), 'foo(bar)')
        self.assertEqual(f('foo(bar baz)'), 'foo(bar baz)')

    def test_strips_leading_parenthetical(self):
        f = normalizers.StripLeadingAndTrailingParentheticalsNormalizer()

        # with whitespace
        self.assertEqual(f('(foo) bar'), 'bar')
        self.assertEqual(f('(foo bar) baz'), 'baz')

        # without whitespace
        self.assertEqual(f('(foo)bar'), 'bar')
        self.assertEqual(f('(foo bar)baz'), 'baz')

    def test_strips_trailing_parenthetical(self):
        f = normalizers.StripLeadingAndTrailingParentheticalsNormalizer()

        # with whitespace
        self.assertEqual(f('foo (bar)'), 'foo')
        self.assertEqual(f('foo (bar baz)'), 'foo')

        # without whitespace
        self.assertEqual(f('foo(bar)'), 'foo')
        self.assertEqual(f('foo(bar baz)'), 'foo')

    def test_strips_leading_and_trailing_parentheticals(self):
        f = normalizers.StripLeadingAndTrailingParentheticalsNormalizer()

        self.assertEqual(f('(foo) bar (baz)'), 'bar')
        self.assertEqual(f('(foo bar) baz (qux)'), 'baz')
        self.assertEqual(f('(foo) bar (baz qux)'), 'bar')
        self.assertEqual(f('(foo bar) baz (qux quux)'), 'baz')

    def test_strips_repeated_parentheticals(self):
        f = normalizers.StripLeadingAndTrailingParentheticalsNormalizer()

        # leading
        self.assertEqual(f('(foo) (bar) baz'), 'baz')
        self.assertEqual(f('(foo bar) (baz) qux'), 'qux')
        # trailing
        self.assertEqual(f('foo (bar) (baz)'), 'foo')
        self.assertEqual(f('foo (bar baz) (qux)'), 'foo')
        # leading and trailing
        self.assertEqual(f('(foo) (bar) baz (qux) (quux)'), 'baz')


class RemovePostTypeNormalizerTestCase(unittest.TestCase):
    """Test socialnorms.extraction.normalizers.RemovePostTypeNormalizer."""

    def test_leaves_normal_text_alone(self):
        f = normalizers.RemovePostTypeNormalizer()

        self.assertEqual(f('foo'), 'foo')
        self.assertEqual(f('foo bar'), 'foo bar')

    def test_removes_aita_post_type(self):
        f = normalizers.RemovePostTypeNormalizer()

        # uppercase
        self.assertEqual(f('AITA foo bar'), ' foo bar')
        # capitalized
        self.assertEqual(f('Aita foo bar'), ' foo bar')
        # lowercase
        self.assertEqual(f('aita foo bar'), ' foo bar')
        # with an insertion
        self.assertEqual(f('aitaa foo bar'), ' foo bar')
        # with a deletion
        self.assertEqual(f('aia foo bar'), ' foo bar')
        # with a substitution
        self.assertEqual(f('atta foo bar'), ' foo bar')

    def test_removes_wibta_post_type(self):
        f = normalizers.RemovePostTypeNormalizer()

        # uppercase
        self.assertEqual(f('WIBTA foo bar'), ' foo bar')
        # capitalized
        self.assertEqual(f('Wibta foo bar'), ' foo bar')
        # lowercase
        self.assertEqual(f('wibta foo bar'), ' foo bar')
        # with an insertion
        self.assertEqual(f('wibtaa foo bar'), ' foo bar')
        # with a deletion
        self.assertEqual(f('wiba foo bar'), ' foo bar')
        # with a substitution
        self.assertEqual(f('wibba foo bar'), ' foo bar')


class RemoveExpandedPostTypeNormalizerTestCase(unittest.TestCase):
    """Test socialnorms.extraction.normalizers.RemoveExpandedPostTypeNormalizer."""

    def test_leaves_normal_text_alone(self):
        f = normalizers.RemoveExpandedPostTypeNormalizer()

        self.assertEqual(f('foo'), 'foo')
        self.assertEqual(f('foo bar'), 'foo bar')

    def test_removes_aita_post_type(self):
        f = normalizers.RemoveExpandedPostTypeNormalizer()

        # normal case
        self.assertEqual(
            f('Am I the asshole for foo bar'),
              'for foo bar')
        # lowercased
        self.assertEqual(
            f('am i the asshole for foo bar'),
              'for foo bar')
        # uppercase
        self.assertEqual(
            f('AM I THE ASSHOLE for foo bar'),
              'for foo bar')
        # alternative determiners
        self.assertEqual(
            f('Am I a asshole for foo bar'),
              'for foo bar')
        self.assertEqual(
            f('Am I an asshole for foo bar'),
              'for foo bar')
        # past tense
        self.assertEqual(
            f('Was I the asshole for foo bar'),
              'for foo bar')
        # plural
        self.assertEqual(
            f('Are we the assholes for foo bar'),
              'for foo bar')
        # plural past tense
        self.assertEqual(
            f('Were we the assholes for foo bar'),
              'for foo bar')
        # a-hole variants
        self.assertEqual(
            f('Am I the ahole for foo bar'),
              'for foo bar')
        self.assertEqual(
            f('Am I the a-hole for foo bar'),
              'for foo bar')

    def test_removes_wibta_post_type(self):
        f = normalizers.RemoveExpandedPostTypeNormalizer()

        # normal case
        self.assertEqual(
            f('Would I be the asshole for foo bar'),
              'for foo bar')
        # lowercased
        self.assertEqual(
            f('would i be the asshole for foo bar'),
              'for foo bar')
        # uppercase
        self.assertEqual(
            f('WOULD I BE THE ASSHOLE for foo bar'),
              'for foo bar')
        # alternative determiners
        self.assertEqual(
            f('Would I be a asshole for foo bar'),
              'for foo bar')
        self.assertEqual(
            f('Would I be an asshole for foo bar'),
              'for foo bar')
        # past tense
        self.assertEqual(
            f('Would I have been the asshole for foo bar'),
              'for foo bar')
        # plural
        self.assertEqual(
            f('Would we be the assholes for foo bar'),
              'for foo bar')
        # plural past tense
        self.assertEqual(
            f('Would we have been the assholes for foo bar'),
              'for foo bar')
        # a-hole variants
        self.assertEqual(
            f('Would I be the ahole for foo bar'),
              'for foo bar')
        self.assertEqual(
            f('Would I be the a-hole for foo bar'),
              'for foo bar')

    def test_removes_partial_post_type_signifiers(self):
        f = normalizers.RemoveExpandedPostTypeNormalizer()

        self.assertEqual(
            f('a asshole for foo bar'),
            'for foo bar')
        self.assertEqual(
            f('an asshole for foo bar'),
            'for foo bar')
        self.assertEqual(
            f('the asshole for foo bar'),
            'for foo bar')
        self.assertEqual(
            f('asshole for foo bar'),
            'for foo bar')
        self.assertEqual(
            f('ahole for foo bar'),
            'for foo bar')
        self.assertEqual(
            f('a-hole for foo bar'),
            'for foo bar')
        self.assertEqual(
            f('assholes for foo bar'),
            'for foo bar')
        self.assertEqual(
            f('aholes for foo bar'),
            'for foo bar')
        self.assertEqual(
            f('a-holes for foo bar'),
            'for foo bar')


class WhitespaceNormalizerTestCase(unittest.TestCase):
    """Test socialnorms.extraction.normalizers.WhitespaceNormalizer."""

    def test_leaves_normal_text_alone(self):
        f = normalizers.WhitespaceNormalizer()

        self.assertEqual(f('foo'), 'foo')
        self.assertEqual(f('foo bar'), 'foo bar')

    def test_strips_leading_whitespace(self):
        f = normalizers.WhitespaceNormalizer()

        # space character
        self.assertEqual(f(' foo'), 'foo')
        self.assertEqual(f('  foo'), 'foo')
        self.assertEqual(f('   foo'), 'foo')
        self.assertEqual(f(' foo bar'), 'foo bar')
        self.assertEqual(f('  foo bar'), 'foo bar')
        self.assertEqual(f('   foo bar'), 'foo bar')
        # tab character
        self.assertEqual(f('\tfoo'), 'foo')
        self.assertEqual(f('\t\tfoo'), 'foo')
        self.assertEqual(f('\t\t\tfoo'), 'foo')
        self.assertEqual(f('\tfoo bar'), 'foo bar')
        self.assertEqual(f('\t\tfoo bar'), 'foo bar')
        self.assertEqual(f('\t\t\tfoo bar'), 'foo bar')
        # newline character
        self.assertEqual(f('\nfoo'), 'foo')
        self.assertEqual(f('\n\nfoo'), 'foo')
        self.assertEqual(f('\n\n\nfoo'), 'foo')
        self.assertEqual(f('\nfoo bar'), 'foo bar')
        self.assertEqual(f('\n\nfoo bar'), 'foo bar')
        self.assertEqual(f('\n\n\nfoo bar'), 'foo bar')

    def test_strips_trailing_whitespace(self):
        f = normalizers.WhitespaceNormalizer()

        # space character
        self.assertEqual(f('foo '), 'foo')
        self.assertEqual(f('foo  '), 'foo')
        self.assertEqual(f('foo   '), 'foo')
        self.assertEqual(f('foo bar '), 'foo bar')
        self.assertEqual(f('foo bar  '), 'foo bar')
        self.assertEqual(f('foo bar   '), 'foo bar')
        # tab character
        self.assertEqual(f('foo\t'), 'foo')
        self.assertEqual(f('foo\t\t'), 'foo')
        self.assertEqual(f('foo\t\t\t'), 'foo')
        self.assertEqual(f('foo bar\t'), 'foo bar')
        self.assertEqual(f('foo bar\t\t'), 'foo bar')
        self.assertEqual(f('foo bar\t\t\t'), 'foo bar')
        # newline character
        self.assertEqual(f('foo\n'), 'foo')
        self.assertEqual(f('foo\n\n'), 'foo')
        self.assertEqual(f('foo\n\n\n'), 'foo')
        self.assertEqual(f('foo bar\n'), 'foo bar')
        self.assertEqual(f('foo bar\n\n'), 'foo bar')
        self.assertEqual(f('foo bar\n\n\n'), 'foo bar')

    def test_replaces_internal_whitespace(self):
        f = normalizers.WhitespaceNormalizer()

        # space character
        self.assertEqual(f('foo bar'), 'foo bar')
        self.assertEqual(f('foo  bar'), 'foo bar')
        self.assertEqual(f('foo   bar'), 'foo bar')
        # tab character
        self.assertEqual(f('foo\tbar'), 'foo bar')
        self.assertEqual(f('foo\t\tbar'), 'foo bar')
        self.assertEqual(f('foo\t\t\tbar'), 'foo bar')
        # newline character
        self.assertEqual(f('foo\nbar'), 'foo bar')
        self.assertEqual(f('foo\n\nbar'), 'foo bar')
        self.assertEqual(f('foo\n\n\nbar'), 'foo bar')

    def test_replaces_mixed_whitespace(self):
        f = normalizers.WhitespaceNormalizer()

        # leading
        self.assertEqual(f('\n\t foo'), 'foo')
        self.assertEqual(f('\n \tfoo bar'), 'foo bar')
        # trailing
        self.assertEqual(f('foo \n\n'), 'foo')
        self.assertEqual(f('foo bar \t\n '), 'foo bar')
        # leading, trailing, and internal
        self.assertEqual(f('\n\t foo \n\n'), 'foo')
        self.assertEqual(f('\t \nfoo \t \nbar \n\t '), 'foo bar')


class CapitalizationNormalizerTestCase(unittest.TestCase):
    """Test socialnorms.extraction.normalizers.CapitalizationNormalizer."""

    def test_leaves_normal_text_alone(self):
        f = normalizers.CapitalizationNormalizer()

        self.assertEqual(f('foo'), 'foo')
        self.assertEqual(f('foo bar'), 'foo bar')

    def test_leaves_place_names_capitalized(self):
        f = normalizers.CapitalizationNormalizer()

        self.assertEqual(f('NYC'), 'NYC')
        self.assertEqual(f('New York City'), 'New York City')
        self.assertEqual(f('he flew to Seattle.'), 'he flew to Seattle.')

    def test_lower_cases_sentence_starts(self):
        f = normalizers.CapitalizationNormalizer()

        self.assertEqual(f('Hello, world!'), 'hello, world!')
        self.assertEqual(
            f('He drove to the airport.'),
              'he drove to the airport.')

    def test_fixes_title_style_capitalization(self):
        f = normalizers.CapitalizationNormalizer()

        self.assertEqual(
            f('For Driving My Car'),
              'for driving my car')
        self.assertEqual(
            f('Being Mad at My Brother for Not Hanging Out More'),
              'being mad at my brother for not hanging out more')

    def test_capitalizes_I_correctly(self):
        f = normalizers.CapitalizationNormalizer()

        self.assertEqual(
            f('I went to the store'),
              'I went to the store')
        self.assertEqual(
            f('The other day, I went to the store.'),
              'the other day, I went to the store.')
        self.assertEqual(
            f('The other day, i went to the store.'),
              'the other day, I went to the store.')

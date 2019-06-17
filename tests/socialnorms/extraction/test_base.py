"""Tests for socialnorms.extraction.base."""

import unittest

from socialnorms.extraction import base


class LoggedCallableTestCase(unittest.TestCase):
    """Test socialnorms.extraction.base.LoggedCallable."""

    # classes for testing LoggedCallable

    class Add1(base.LoggedCallable):
        def apply(self, y):
            return y + 1

    class AddX(base.LoggedCallable):
        def __init__(self, x, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self.x = x

        def apply(self, y):
            return self.x + y

    # tests

    def test_apply_raises_not_implemented(self):
        logged_callable = base.LoggedCallable()

        with self.assertRaises(NotImplementedError):
            logged_callable.apply()

        with self.assertRaises(NotImplementedError):
            logged_callable()

    def test_subclass_without_args(self):
        add1 = self.Add1()

        self.assertEqual(add1(0), 1)
        self.assertEqual(add1(1), 2)
        self.assertEqual(add1(5), 6)

        self.assertEqual(
            add1.call_log,
            [
                (
                    # inputs
                    (
                        # args
                        (0,),
                        # kwargs
                        {}
                    ),
                    # outputs
                    1
                ),
                (
                    # inputs
                    (
                        # args
                        (1,),
                        # kwargs
                        {}
                    ),
                    # outputs
                    2
                ),
                (
                    # inputs
                    (
                        # args
                        (5,),
                        # kwargs
                        {}
                    ),
                    # outputs
                    6
                )
            ])

    def test_subclass_with_args(self):
        add5 = self.AddX(x=5)

        self.assertEqual(add5(0), 5)
        self.assertEqual(add5(1), 6)
        self.assertEqual(add5(5), 10)

        self.assertEqual(
            add5.call_log,
            [
                (
                    # inputs
                    (
                        # args
                        (0,),
                        # kwargs
                        {}
                    ),
                    # outputs
                    5
                ),
                (
                    # inputs
                    (
                        # args
                        (1,),
                        # kwargs
                        {}
                    ),
                    # outputs
                    6
                ),
                (
                    # inputs
                    (
                        # args
                        (5,),
                        # kwargs
                        {}
                    ),
                    # outputs
                    10
                )
            ])

    def test_when_log_calls_is_false(self):
        add1 = self.Add1(log_calls=False)

        self.assertEqual(add1(0), 1)
        self.assertEqual(add1(1), 2)
        self.assertEqual(add1(5), 6)

        self.assertEqual(add1.call_log, None)

        add5 = self.AddX(x=5, log_calls=False)

        self.assertEqual(add5(0), 5)
        self.assertEqual(add5(1), 6)
        self.assertEqual(add5(5), 10)

        self.assertEqual(add5.call_log, None)


class CaseTestCase(unittest.TestCase):
    """Test socialnorms.extraction.base.Case."""

    class SomeToNoneCase(base.Case):
        def match(self, x):
            return (x, x is not None)

        def transform(self, x):
            return None

        def filter(self, x):
            return False

    # tests

    def test_subclass_when_case_matches(self):
        self.assertEqual(
            self.SomeToNoneCase()(1),
            (None, True))
        self.assertEqual(
            self.SomeToNoneCase()('a'),
            (None, True))
        self.assertEqual(
            self.SomeToNoneCase()(''),
            (None, True))

    def test_subclass_when_case_does_not_match(self):
        self.assertEqual(
            self.SomeToNoneCase()(None),
            (None, False))

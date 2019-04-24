"""Tests for socialnorms.data.utils."""

import unittest

import attr

from socialnorms.data import utils


class InstantiateAttrsWithExtraKwargsTestCase(unittest.TestCase):
    """Test instantiate_attrs_with_extra_kwargs."""

    class NonAttrsClass:
        def __init__(self, foo):
            self.foo = foo

    @attr.s
    class AttrsClass:
        foo = attr.ib()

    def test_raises_error_on_non_attrs_classes(self):
        with self.assertRaisesRegex(
                ValueError,
                r'.* must be decorated with @attr\.s'
        ):
            utils.instantiate_attrs_with_extra_kwargs(
                self.NonAttrsClass, foo=1)

    def test_instantiates_class_with_no_extra_kwargs(self):
        instance = utils.instantiate_attrs_with_extra_kwargs(
            self.AttrsClass, foo=1)

        self.assertEqual(instance.foo, 1)

    def test_instantiates_class_with_extra_kwargs(self):
        instance = utils.instantiate_attrs_with_extra_kwargs(
            self.AttrsClass, foo='a', bar='b')

        self.assertEqual(instance.foo, 'a')


class CachedPropertyTestCase(unittest.TestCase):
    """Test cached_property."""

    def test_makes_method_into_property(self):
        class Foo:
            @utils.cached_property
            def bar(self):
                return 1

        foo = Foo()

        self.assertEqual(foo.bar, 1)

    def test_caches_property_from_method(self):
        class Foo:
            @utils.cached_property
            def bar(self):
                # this method (if not cached) will increment it's return
                # value based on the number of times it has been called
                self.num_calls = 1 + getattr(self, 'num_calls', 0)

                return self.num_calls

        foo = Foo()

        # calling bar once should return 1
        self.assertEqual(foo.bar, 1)
        # if bar is called more than once (i.e., the method is not
        # cached) then it will return something greater than one
        self.assertEqual(foo.bar, 1)

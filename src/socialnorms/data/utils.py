"""Utilities for working with data."""

from functools import wraps
from typing import (
    Any,
    Callable,
    Dict)

import attr


def instantiate_attrs_with_extra_kwargs(
        cls: Any,
        **kwargs: Dict[str, Any]
):
    """Return ``cls`` instantiated with ``kwargs`` ignoring extra kwargs.

    Parameters
    ----------
    cls : Object
        An object that has been decorated with ``@attr.s``.
    **kwargs : Dict[str, Any]
        Any keyword arguments to use when instantiating ``cls``. Extra
        keyword arguments will be ignored.
    """
    if not attr.has(cls):
        raise ValueError(f'{cls} must be decorated with @attr.s')

    attr_names = attr.fields_dict(cls).keys()
    return cls(**{
        k: kwargs[k]
        for k in attr_names
    })


def cached_property(method: Callable):
    """Decorate a method to act as a cached property.

    This decorator converts a method into a cached property. It is
    intended to only be used on the methods of classes decorated with
    ``@attr.s`` where ``frozen=True``. This decorator works analogously
    to ``@property`` except it caches the computed value.

    Parameters
    ----------
    method : Callable, required
        The method to decorate. ``method`` should take only one
        argument: ``self``.

    Returns
    -------
    Callable
        The decoratored method.

    Notes
    -----
    When used on a frozen attrs class, values for the property may
    safely be cached because the object is intended to be
    immutable. Additionally, the best place to store these cached values
    is on the object itself, so that they can be garbage collected when
    the object is.
    """
    @wraps(method)
    def wrapper(self):
        cached_name = f'_{method.__name__}'
        if not hasattr(self, cached_name):
            value = method(self)

            # To get around the immutability of the instance, we have to
            # use __setattr__ from object.
            object.__setattr__(self, cached_name, value)

        return getattr(self, cached_name)
    return property(wrapper)

"""Base classes and utilities for extracting data from text."""

import copy
from typing import Any, Tuple


class LoggedCallable:
    """A base class for callable objects that should log their calls.

    ``LoggedCallable`` should not be instantiated directly, rather it
    should only be subclassed. Children must implement the ``apply``
    method which defines the main logic for when the object is
    called. Direct calls to the ``apply`` method are not logged.

    When a child of ``LoggedCallable`` is called, the input-output pair
    is appended to the ``call_log`` attribute. The input and output
    objects are copied using ``copy.deepcopy`` from the standard library
    to try and preserve the objects as they were at call time. Calls are
    recorded in the order they were made.

    Attributes
    ----------
    call_log : Optional[List[Tuple[Any, Any]]]
        If ``None``, then calls are not being logged, otherwise a list
        of input-output pairs from all calls to the object. Inputs are
        stored as tuples of arguments and keyword arguments,
        ``(args, kwargs)``.

    Parameters
    ----------
    log_calls : bool, optional (default=True)
        If ``True``, then calls will be logged, otherwise calls will not
        be logged.

    Example
    -------
    To create a logged callable, you would subclass ``LoggedCallable``
    and override the ``apply`` method. If overriding ``__init__`` or
    ``__call__``, you must call the method on the super class::

        class AddX(LoggedCallabe):
            def __init__(self, x, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.x = x

            def apply(self, y):
                return self.x + y

        add2 = AddX(2)
        add2(-1)  # returns 1
        add2.call_log  # is [(((-1,), {}), 1)]

    """

    def __init__(
            self,
            log_calls: bool = True
    ):
        self.call_log = [] if log_calls else None

    def apply(self, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        output = self.apply(*args, **kwargs)

        if self.call_log is not None:
            self.call_log.append((
                (
                    copy.deepcopy(args),
                    copy.deepcopy(kwargs)
                ),
                copy.deepcopy(output)
            ))

        return output


class Case(LoggedCallable):
    """A base class for cases for applying different extraction rules.

    A case for applying a set of extraction rules consists of three
    components:

      1. a matching method that identifies when the case applies.
      2. a transforming method that transforms the input into the
         desired output.
      3. a filter method that identifies bad inputs that have not been
         successfully transformed into the proper output.

    ``Case`` should be subclassed to create new cases. Children must
    implement the ``match``, ``transform``, and ``filter`` methods
    described `Methods`_ section.

    When a case is called, first ``match`` is run on the input and if
    the match is successful then the input is transformed (with
    ``transform``) and possibly filtered out if ``filter`` returns
    ``True`` on the transformed data.

    Methods
    -------
    match(self, x: Any) -> Tuple[Any, bool]
        Return a tuple whose first element is (optionally) some data
        extracted from the input and whose second element is a boolean
        clarifying whether or not the input matches the case. If the
        case does not match, by convention the first element should be
       ``None``.
    transform(self, x: Any) -> Any
        Return a transformed version of ``x``.
    filter(self, x: Any) -> bool
        Return ``True`` if the case output from ``transform`` should be
        considered a failed case.

    Returns
    -------
    output : Any, success : bool
        Return the output corresponding to the cases input, with a
        boolean signifying whether or not the case completed
        successfully (i.e., matched the input and did not filter the
        output out).

    Example
    -------
    To create a case, subclass ``Case``, implement the ``match``,
    ``transform``, and ``filter`` methods and then instantiate and call
    the object from the subclass::

        class NonnegativeToProbability(Case):
            def match(self, x: float) -> Tuple[float, bool]:
                if x >= 0:
                    return (x, True)
                else:
                    return (None, False)

            def transform(self, x: float) -> float:
                return x / (1 + x)

            def filter(self, x: float) -> bool:
                return math.isnan(x)

        nonnegative_to_probability = NonnegativeToProbability()
        nonnegative_to_probability(1)  # is (0.5, True)
        nonnegative_to_probability(-1)  # is (None, False)

    """
    def match(
            self,
            x: Any
    ) -> Tuple[Any, bool]:
        raise NotImplementedError()

    def transform(
            self,
            x: Any
    ) -> Any:
        raise NotImplementedError()

    def filter(
            self,
            x: Any
    ) -> bool:
        raise NotImplementedError()

    def apply(
            self,
            x: Any
    ) -> Tuple[Any, bool]:
        """Return the results of applying this case to ``x``.

        Return a tuple, ``(output, success)``, where output is
        potentially a value computed from ``x`` and ``success`` is a
        boolean signifying whether or not ``x`` matches this case.

        Parameters
        ----------
        x : Any
            The input to process with this case.

        Returns
        -------
        output : Any, success : bool
            A tuple containing the value computed from ``x`` (output),
            and whether or not the case matched ``x`` successfully
            (success).
        """
        x, success = self.match(x)

        if not success:
            return (x, False)

        x = self.transform(x)

        if self.filter(x):
            return (x, False)

        return (x, True)

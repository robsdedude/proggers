import abc
from collections import defaultdict
from copy import deepcopy
from typing import (
    DefaultDict,
    Dict,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from .data_sources import T_DB_TUPLES
from .query import Query

T = TypeVar("T")
T_HISTOGRAM = DefaultDict[Tuple[str, ...], int]


class ErrorMetric(abc.ABC):
    """Base class for error metrics used in the result cache."""
    def __init__(self):
        self._cache = None
        self._histogram = None  # type: Optional[T_HISTOGRAM]

    @property
    def histogram(self):
        return self._histogram

    @histogram.setter
    def histogram(self, value):
        self._histogram = value
        self._cache = None

    @classmethod
    @abc.abstractmethod
    def calculate_error(cls, histogram):
        pass

    @property
    def error(self):
        if self._cache is None:
            self._cache = self.__class__.calculate_error(self.histogram)
        return self._cache


class NormStdError(ErrorMetric):
    @classmethod
    def calculate_error(cls, histogram: Optional[T_HISTOGRAM]):
        # Can only be used if more than one value exists else error would be
        # estimated to be 0
        if histogram is None:  # or len(histogram) <= 1:
            return float('inf')
        n = sum(v for v in histogram.values())
        if not n or len(histogram) <= 1 and n < 1000:
            return float('inf')
        p_estimate = {k: v/n for k, v in histogram.items()}
        error_estimate = {
            k: 1/p*(p*(1-p)/n)*.5
            for k, p in p_estimate.items()
        }
        return sum(error_estimate.values())


T_ERROR_METRIC_TYPE = Type[ErrorMetric]
T_ERROR_METRIC = TypeVar("T_ERROR_METRIC", bound=ErrorMetric)
T_RC_ENTRY = Tuple[T_HISTOGRAM, T_ERROR_METRIC]


class ResultCache:
    """Result cache builds the progressive histogram and error estimate."""
    def __init__(self, error_metric: T_ERROR_METRIC_TYPE = NormStdError):
        """
        :param error_metric: the error estimator to be used (class).
        """
        self._store = {}  # type: Dict[Query, T_RC_ENTRY]
        self.error_metric = error_metric

    def process_tuples(self, tuples: T_DB_TUPLES, q: Query):
        """
        Update histogram and error estimate with a tuple of tuples.

        :param tuples: tuples of tuples that were retrieved.
        :param q: the query the tuples where retrieved for.
        """
        q = q.canonical()
        queries = [Query((g,), q.where).canonical() for g in q.groups]
        if len(q.groups) > 1:
            queries.append(q)
        for query in queries:
            if query not in self._store:
                self._store[query] = defaultdict(int), self.error_metric()
            else:
                self._store[query] = (deepcopy(self._store[query][0]),
                                      self.error_metric())
            for tuple_ in tuples:
                assert (all(attr in tuple_ for attr in query.groups)
                        or print(tuple_, query))
                values = tuple(tuple_[attr] for attr in query.groups)
                self._store[query][0][values] += 1
            self._store[query][1].histogram = self._store[query][0]

    def __str__(self):
        return "ResultCache " + str(self._store)

    def __getitem__(self, q: Query):
        return deepcopy(self._store[q.canonical()])

    def get(self, q: Query, default: T = (None, None)) -> Union[T, T_RC_ENTRY]:
        """Get current histogram and error estimate for a query.

        :param q: the query to look the results up for.
        :param default: what to return if no results are available.

        :returns: default if no results are available, else a tuple
            `(histogram, error_estimate)`.
        """
        res = self._store.get(q.canonical(), None)
        if res is None:
            return default
        else:
            return deepcopy(res)

import threading
import time
from typing import (
    Dict,
    Optional,
    Set,
    TypeVar,
)

from .data_sources import DataSource, T_DB_TUPLES
from .query import Query
from .result_cache import ResultCache
from .tail_index import TailIndex

T = TypeVar("T")
TCursor = Optional[T]

TCursorsDict = Dict[Query, TCursor]

MIN_TUPLES_PER_CYCLE = 1
MAX_TUPLES_PER_CYCLE = 10000
GOAL_CYCLE_S = 1  # 100 ms update cycle desired but Python is too slow
CYCLE_DECREASE_FACTOR = 2
CYCLE_INCREASE_FACTOR = 0.8


class QueryEngine(threading.Thread):
    """Query engine manages queries and streams tuples to the result cache."""

    daemon = True  # daemon thread dies with its parent

    def __init__(self, data_source: DataSource,
                 tail_index: bool = True, or_support: bool = True,
                 *thread_args, **thread_kwargs):
        """
        :param data_source: the data source to explore.
        :param tail_index: whether to use a tail index or stream all tuples
            directly from the data source.
        :param or_support: whether to support arbitrary disjunctions in queries
            or only disjunctions of selections of pairwise disjoint
            subpopulations.
        :param thread_args: passed to `threading.Thread.__init__`.
        :param thread_kwargs: passed to `threading.Thread.__init__`.
        """
        super().__init__(*thread_args, **thread_kwargs)
        self.ds = data_source
        self.cursors = dict()  # type: TCursorsDict
        self.tuples_per_cycle = dict()  # type: Dict[Query, int]
        self.running_queries = set()  # type: Set[Query]
        self.finished_queries = set()  # type: Set[Query]
        print("tail_index:", tail_index)
        if tail_index:
            print("or_support:", or_support)
            self.tail_index = TailIndex(data_source, or_support=or_support)
        else:
            self.tail_index = None
        self.queries_lock = threading.RLock()
        self.result_cache = ResultCache()
        self._stopped = False

    def start_query(self, q: Query):
        """Add a query and start streaming its results to the result cache."""
        q = q.canonical()
        with self.queries_lock:
            if q in self.finished_queries:
                return
            if q in self.cursors:
                return self.resume_query(q)
            self.running_queries.add(q)
            if self.tail_index is not None:
                self.tail_index.add_query(q)
            self.cursors[q] = None
            self.tuples_per_cycle[q] = MIN_TUPLES_PER_CYCLE

    def stop_query(self, q: Query):
        """Stop a query (cannot be restarted)."""
        # TODO: could free resources instead
        q = q.canonical()
        if q in self.finished_queries:
            return
        return self.pause_query(q)

    def pause_query(self, q: Query):
        """Pause a query (can be restarted)."""
        q = q.canonical()
        with self.queries_lock:
            if q in self.finished_queries:
                return
            assert q in self.cursors
            if self.tail_index is not None:
                self.tail_index.pause_query(q)
            self.running_queries -= {q}

    def resume_query(self, q: Query):
        """Restart a previously paused query."""
        q = q.canonical()
        with self.queries_lock:
            if q in self.finished_queries:
                return
            assert q in self.cursors
            if self.tail_index is not None:
                self.tail_index.resume_query(q)
            self.running_queries.add(q)

    @staticmethod
    def _adjust_tuples_per_cycle(n: int, s: float) -> int:
        """
        Adaptive sample size.

        Calculate the number of requested tuples in the next iteration of
        depending on the number of tuples returned this time and the time took
        to do so.

        :param n: number of tuples returned this iteration.
        :param s: time (in seconds) took to return `n` tuples this iteration.
        """
        if s < GOAL_CYCLE_S:
            res = int((GOAL_CYCLE_S / s) * CYCLE_INCREASE_FACTOR * n)
        else:
            res = int((GOAL_CYCLE_S / (s * CYCLE_DECREASE_FACTOR)) * n)
        return min(MAX_TUPLES_PER_CYCLE, max(res, MIN_TUPLES_PER_CYCLE))

    def _get_tuples(self, query: Query, n: int, cursor: TCursor,
                    timeout: Optional[float]) \
            -> (T_DB_TUPLES, TCursor):
        """
        Helper method to get tuples.

        Takes care of using the tail index or not and manages the cursors.
        """
        if self.tail_index is not None:
            ds = self.tail_index[query.where]
            return ds.get_tuples(n, cursor, timeout=timeout)
        res = tuple()
        while True:
            new_tuples, cursor = self.ds.get_tuples(n - len(res), cursor,
                                                    timeout=timeout)
            if query.where is True:
                res += new_tuples
            elif query.where is False:
                return res, None
            else:
                res += tuple(t for t in new_tuples if query.where.eval(t))
            if cursor is None or len(res) == n:
                return res, cursor

    def stop(self):
        """Stop the query engine (cannot be restarted)"""
        self._stopped = True
        with self.queries_lock:
            if self.tail_index is not None:
                self.tail_index.stop()

    def run(self):
        """Iterative function to stream tuples into the result cache."""
        while not self._stopped:
            with self.queries_lock:
                num_queries = len(self.running_queries)
                for query in self.running_queries:
                    cursor = self.cursors[query]
                    n = self.tuples_per_cycle[query]
                    n_fraction = max(int(n / num_queries), MIN_TUPLES_PER_CYCLE)
                    start = time.time()
                    tuples, cursor = self._get_tuples(
                        query, n_fraction, cursor,
                        timeout=(GOAL_CYCLE_S / num_queries)
                    )
                    elapsed = (time.time() - start)
                    self.cursors[query] = cursor
                    if not tuples:
                        time.sleep(
                            max((GOAL_CYCLE_S / num_queries) - elapsed, 0)
                        )
                    self.result_cache.process_tuples(tuples, query)
                    if cursor is None:
                        self.finished_queries.add(query)
                        del self.tuples_per_cycle[query]
                        del self.cursors[query]
                    if not tuples or cursor is None:
                        continue
                    virtual_elapsed = num_queries * elapsed
                    self.tuples_per_cycle[query] = \
                        QueryEngine._adjust_tuples_per_cycle(len(tuples),
                                                             virtual_elapsed)
            self.running_queries -= self.finished_queries
            if not num_queries:
                # no active waiting for queries
                time.sleep(.5)

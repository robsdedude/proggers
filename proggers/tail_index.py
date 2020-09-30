from abc import (
    ABC,
    abstractmethod,
)
from collections import defaultdict
from copy import deepcopy
from threading import (
    Condition,
    Thread,
    Lock,
    RLock,
)
import sys
import time
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Iterable,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
from numpy.random import multinomial

from .data_sources import (
    DataSource,
    EmptyDataSource,
    T_DS,
    T_CURSOR,
    T_DB_TUPLE,
    T_DB_TUPLES,
)
from .query import (
    Where,
    WhereLiteral,
    WhereOp,
    Query,
)
from .util import (
    acquire_deadlock_check,
    join_clauses,
    lock_ctx_deadlock_check,
    powerset,
    ReadWriteLock,
    TemporalCache,
)

T_STR_TUPLE = Tuple[str, ...]

DUMMY = object()  # placeholder value for tail index structure

# list for indexed tuples, int for count of unindexed tuples
T_INDEX_DICT = Dict[str, Union[List[T_DB_TUPLE], int]]
T_ENTRY_FILTER = Set[T_STR_TUPLE]
# [attribute][value]
# value is DUMMY for template sub-index
T_SUB_INDEX_DICT = Dict[str, Dict[Union[object, str], "TailIndexEntry"]]

# TODO: free resources of long time not used TailIndexEntries when close to
#       system's memory limit.

# attribute values rarer than this frequency will be indexed
RARE_THRESHOLD = 1/100


def with_vm_tree_structure_lock(inner):
    """
    Decorator for protecting a TailIndexEntry method with its value manager's
    tree_structure_lock.
    """
    def outer(self: "TailIndexEntry", *args, **kwargs):
        if self.vm is not None:
            with lock_ctx_deadlock_check(self.vm.tree_structure_lock):
                return inner(self, *args, **kwargs)
        else:
            return inner(self, *args, **kwargs)
    return outer


class TailIndexEntry:
    """
    One entry in the tail index.

    It represents a node that tracks a subpopulation belonging to one attribute.
    The `index` attribute is a dictionary that contains the indexed tuples keyed
    by their value. The values of `index` are lists containing the tuples if the
    attribute value is index (rare enough), or the number of seen tuples, if
    it's not indexed.
    The tail index is a tree, so the node has child nodes and a parent (unless
    it's the root). The children are in `sub_indexes` accessedd as
    `sub_indexes[attribute_of_subindex][value_of_this_index]`.
    For more details read the master's thesis that belongs to this code.
    """
    def __init__(self, where: Optional[Where], attribute: str,
                 value_manager: Optional["ValueManager"]):
        """
        :param where: filter that describes what tuples are indexed in this
            node. Only `None` for the root node (i.e. all tuples are indexed).
        :param attribute: The attribute this index indexes.
        :param value_manager: The value manager for this tail index tree.
            It propagates newly discovered values of attributes through the
            whole tree structure.
        """
        assert (where is None
                or where.is_wrapped_and_clause()
                or print(repr(where)))
        self.where = where
        self._vm = None  # type: Optional[ValueManager]
        self._attr = attribute
        self.vm = value_manager
        # only increases when parent's where matches
        self._total_count_filtered = 0
        # increases when ever white list matches
        self._total_count = 0
        # only increases when own where matches
        self._match_count = 0
        # will be an int if index is disabled because a value is too common
        self.index = {}  # type: T_INDEX_DICT
        # self.sub_indexes[attribute][value]
        # value is None for template sub-index
        self.sub_indexes = {}  # type: T_SUB_INDEX_DICT
        self.parent_index = None  # type: Optional[TailIndexEntry]
        self.complete = False
        self._root = None
        # Only guards the index entries.
        # Frequency estimates might be inconsistent
        rwlock = ReadWriteLock()
        # self.index_lock = RLock()
        self.index_lock = rwlock.write_lock
        self.index_read_lock = rwlock.read_lock

    @property
    def attr(self):
        return self._attr

    @attr.setter
    def attr(self, attr: str):
        if self.vm is not None:
            self.vm.unsubscribe(self, self.attr)
        self._attr = attr
        if self.vm is not None:
            self.vm.subscribe(self, attr)

    @property
    def vm(self):
        return self._vm

    @vm.setter
    def vm(self, vm: Optional["ValueManager"]):
        if self._vm is not None:
            self._vm.unsubscribe(self, self.attr)
        self._vm = vm
        if self._vm is not None:
            self._vm.subscribe(self, self.attr)

    def __hash__(self):
        return hash((self.where, self.attr))

    def __eq__(self, other: "TailIndexEntry"):
        return other.where == self.where and other.attr == self.attr

    @property
    def root(self):
        """Indicate if this node is the root node."""
        if self.parent_index is None:
            return self
        if self._root is None:
            self._root = self.parent_index.root
        return self._root

    def _index_tuples(self, tuples_: T_DB_TUPLES,
                      white_list: Optional[T_ENTRY_FILTER],
                      new_path: T_STR_TUPLE,
                      absolute_tuple_count: Optional[int],
                      relative_tuple_count: Optional[int]):
        """
        Recursively add `tuples` to the index.

        - filter the tuples by `self.where`
        - sort them into their bins of `self.index`
        - drop bins that correspond to too common subpopulations
        - update the internal counts used for frequency estimation

        :param tuples_: the to be analyzed tuples.
        :param white_list: passed recursively to `index_tuples` of the child
            nodes.
        :param new_path: passed recursively to `index_tuples` of the child
            nodes.
        :param absolute_tuple_count: number of tuples that are added the tail
            index (not only this node).
        :param relative_tuple_count: like `absolute_tuple_count`, but with a
            correction mechanism for increasing rarity further down the tree.

            [Galakatos et al., Section 4.1 second to last paragraph]
        """
        if self.where is not None:
            matches = tuple(t for t in tuples_ if self.where.eval(t))
        else:
            matches = tuples_
        for tup in matches:
            val = tup[self.attr]
            with lock_ctx_deadlock_check(self.index_lock, timeout=15):
                if val not in self.index:
                    # new value found
                    if self.vm is not None:
                        self.vm.notify_new_value(self.attr, val)
                    else:
                        self.notify_new_value(val)
                bucket = self.index[val]
                if isinstance(bucket, list):
                    bucket.append(tup)
                else:
                    self.index[val] += 1
        with lock_ctx_deadlock_check(self.index_read_lock):
            if absolute_tuple_count is None:
                absolute_tuple_count = len(tuples_)
            if relative_tuple_count is None:
                relative_tuple_count = len(tuples_)
            self._total_count += absolute_tuple_count
            self._total_count_filtered += relative_tuple_count
            # waiting until estimates are somewhat reliable™
            # TODO: argue/experiment with error estimate
            if self._total_count_filtered > 1 / RARE_THRESHOLD * 10:
                # enough samples to start forgetting common values
                estimates = self.frequency_estimate
                with lock_ctx_deadlock_check(self.index_lock):
                    for value in self.index.keys():
                        if (isinstance(self.index[value], list)
                                and estimates[value] > RARE_THRESHOLD):
                            self.index[value] = len(self.index[value])
            for attr, sub_indexes in self.sub_indexes.items():
                for val, sub_index in sub_indexes.items():
                    if not isinstance(val, str):  # placeholder entry
                        assert False
                    if isinstance(self.index[val], list):
                        rtc = len(matches)
                    else:
                        # adjust for unindexd parent attribute value
                        # cf. Galakatos et al., sec. 4.1 second to last
                        # paragraph
                        rtc = relative_tuple_count
                    sub_index.index_tuples(
                        matches, white_list=white_list, path=new_path,
                        absolute_tuple_count=absolute_tuple_count,
                        relative_tuple_count=rtc
                    )

    def _pass_down(self, tuples_: T_DB_TUPLES,
                   white_list: Optional[T_ENTRY_FILTER],
                   new_path: T_STR_TUPLE,
                   absolute_tuple_count: Optional[int],
                   relative_tuple_count: Optional[int]):
        """
        Like `_index_tuples` but this index node is left untouched.
        """
        def matches():
            nonlocal _matches
            if not _matches:
                if self.where is None:
                    _matches = tuples_
                else:
                    _matches = tuple(t for t in tuples_ if self.where.eval(t))
            return _matches

        _matches = None

        for val in {tup[self.attr] for tup in tuples_}:
            with lock_ctx_deadlock_check(self.index_lock, timeout=15):
                if val not in self.index:
                    # new value found
                    if self.vm is not None:
                        self.vm.notify_new_value(self.attr, val)
                    else:
                        self.notify_new_value(val)

        with lock_ctx_deadlock_check(self.index_read_lock):
            for sub_indexes in self.sub_indexes.values():
                for val, sub_index in sub_indexes.items():
                    if not isinstance(val, str):
                        assert False
                    if isinstance(self.index[val], int):
                        # adjust for unindexd parent attribute value
                        # cf. Galakatos et al., sec. 4.1 second to last
                        # paragraph
                        sub_index.index_tuples(
                            tuples_, white_list=white_list, path=new_path,
                            absolute_tuple_count=absolute_tuple_count,
                            relative_tuple_count=relative_tuple_count,
                        )
                    else:
                        sub_index.index_tuples(
                            matches(), white_list=white_list, path=new_path,
                            absolute_tuple_count=absolute_tuple_count,
                            relative_tuple_count=relative_tuple_count
                        )

    def index_tuples(self, tuples_: T_DB_TUPLES,
                     white_list: Optional[T_ENTRY_FILTER] = None,
                     path: T_STR_TUPLE = tuple(),
                     absolute_tuple_count: Optional[int] = None,
                     relative_tuple_count: Optional[int] = None):
        """
        Recursively add tuples to the tail index.

        For external use, this function should only be called on root nodes
        with `absolute_tuple_count`, `absolute_tuple_count`, and `path` left
        as default params.

        :param tuples_: the tuples to be indexed.
        :param white_list: specify a set of attribute paths. Every node that
            is not at the end of any of that paths will not be updated
            (`_pass_down` will be called). Node that are whitelisted are
            updated (`_index_tuples` will be used). `None` will cause all nodes
            to be updated.
        :param path: internal tracking of the path during recursion.
        :param absolute_tuple_count: internal counts during recursion.
        :param relative_tuple_count: internal counts during recursion.
        """
        self_filtered_out = False
        new_path = (*path, self.attr)
        if not tuples_:
            return
        if white_list is not None:
            self_filtered_out = new_path not in white_list
            # TODO: can prune this tree traversal if new_path is not a prefix
            #       of any white-listed path
        if self.complete or self_filtered_out:
            self._pass_down(tuples_, white_list, new_path,
                            absolute_tuple_count, relative_tuple_count)
        else:
            self._index_tuples(tuples_, white_list, new_path,
                               absolute_tuple_count, relative_tuple_count)

    def notify_new_value(self, value: str):
        """
        Process new value of `self._attr` was discovered.

        Mainly called from a ValueManager.
        """
        with lock_ctx_deadlock_check(self.index_lock):
            if value in self.index:
                return
            self.index[value] = []
            self._add_value_to_all_sub_indexes(value)

    def _add_value_to_all_sub_indexes(self, value: str):
        with lock_ctx_deadlock_check(self.index_lock):
            for attr in self.sub_indexes:
                self._add_value_to_sub_index(value, attr)

    def _add_value_to_sub_index(self, value: str, attribute: str):
        def where_convert(where: Optional[Where]) -> Optional[Where]:
            if where is None:
                return None
            assert where.is_wrapped_and_clause()
            for p in where.parts[0].parts:
                assert not p.neg or print(where)
                if p.attr == self.attr:
                    p.val = value
            return where

        with lock_ctx_deadlock_check(self.index_lock):
            assert attribute in self.sub_indexes
            assert value not in self.sub_indexes[attribute]
            if (len(self.sub_indexes[attribute]) == 1
                    and next(iter(self.sub_indexes[attribute])) is DUMMY):
                # only template sub_index exists
                if self.where is None:
                    new_where = Where(parts=(WhereLiteral(self.attr, value),),
                                      op=WhereOp.AND).canonical()
                else:
                    new_where = Where(parts=((*self.where.parts[0].parts,
                                              WhereLiteral(self.attr, value))),
                                      op=WhereOp.AND).canonical()
                    assert (new_where
                            and new_where.is_wrapped_and_clause()
                            and (len(new_where.parts[0].parts)
                                 == len(self.where.parts[0].parts) + 1)
                            or print(self.where, new_where))
                dummy_sub_index = self.sub_indexes[attribute][DUMMY]
                dummy_sub_index.where = new_where
                dummy_sub_index.vm = self.vm
                self.sub_indexes[attribute][value] = dummy_sub_index
                del self.sub_indexes[attribute][DUMMY]
            else:
                any_sub_index_key = next(iter(self.sub_indexes[attribute]))
                any_sub_index = self.sub_indexes[attribute][any_sub_index_key]
                self.sub_indexes[attribute][value] = \
                    any_sub_index.clear_deep_copy(keep_completed=False,
                                                  where_convert=where_convert)

    def add_sub_index(self, attribute: str) -> bool:
        """
        Expand the tail index: add a new indexed attribute to this node.

        :param attribute: the new attribute for which to create a sub index.

        :returns: `True` if a new sub index had to be created or `False` if it
            already existed.
        """
        if (self.where is not None
                and attribute in map(lambda p: p.attr,
                                     self.where.parts[0].parts)):
            raise ValueError('attribute "%s" already indexed upstream: %s' %
                             (attribute, self.where))
        if attribute == self.attr:
            raise ValueError('attribute "%s" is currently being indexed.' %
                             attribute)
        if attribute not in self.sub_indexes:
            template_sub_index = TailIndexEntry(None, attribute, None)
            template_sub_index.parent_index = self
            self.sub_indexes[attribute] = {DUMMY: template_sub_index}
            for val in self.index:
                self._add_value_to_sub_index(val, attribute)
            return True
        return False

    def add_sub_indexes(self, attribute_path: T_STR_TUPLE) -> \
            Set[T_STR_TUPLE]:
        """
        Recursively create all needed sub indexes for a path of attributes.

        :param attribute_path: the attribute path to create in the tree.
            e.g. `("sex", "age", "education")`

        :returns: set of created paths.
            e.g. `{("sex", "age"), ("sex", "age", "education")}` of "sex" was
            already indexed as root node.
        """
        ret = set()
        if not attribute_path:
            return ret

        if self.add_sub_index(attribute_path[0]):
            ret.add((attribute_path[0],))
        sub_index = self.sub_indexes[attribute_path[0]]
        returns = []
        for value in sub_index:
            returns.append(sub_index[value].add_sub_indexes(attribute_path[1:]))
        if returns:
            assert (all(r == returns[0]for r in returns)
                    or print(returns))
            ret.update({
                (attribute_path[0], *sub_path) for sub_path in returns[0]
            })
        return ret

    def _frequency_estimate(self, total):
        class EstimateDict(dict):
            def __missing__(self, key):
                nonlocal lengths, total
                if not total:
                    return None
                if key not in lengths:
                    return 0
                return lengths[key] / total

        with lock_ctx_deadlock_check(self.index_lock):
            lengths = {k: len(v) if isinstance(v, list) else v
                       for k, v in self.index.items()}
        return EstimateDict()

    # different frequency estimates:
    #  - global for merging and clauses in queries
    #  - relative to decide if value is rare enough to be indexed
    @property
    def global_frequency_estimate(self):
        return self._frequency_estimate(self._total_count)

    @property
    def frequency_estimate(self):
        return self._frequency_estimate(self._total_count_filtered)

    @property
    def total_count_filtered(self):
        return self._total_count_filtered

    @property
    def total_count(self):
        return self._total_count

    def clear(self, keep_completed=True):
        """Reset the state of this tail index node but remember seen values."""
        if not (self.complete and keep_completed):
            self._total_count_filtered = 0
            self._match_count = 0
            self.index = {k: [] for k in self.index}
            self.complete = False
        for sub_indexes in self.sub_indexes.values():
            for sub_index in sub_indexes.values():
                sub_index.clear(keep_completed=keep_completed)

    @with_vm_tree_structure_lock
    def clear_deep_copy(
            self, keep_completed: bool = True,
            where_convert: Optional[Callable[[Where], Where]] = None) -> \
            "TailIndexEntry":
        """
        Create an empty copy of the tail index tree.

        Deep copy the tail index tree without indexed tuples.
        This will not deep copy parent node but set it nevertheless.

        :param keep_completed: will not reset completed nodes in the copy if
            set to `True` (i.e. notes that have index all tuples of the dataset
            already).
        :param where_convert: optional function that takes a filter (Where) and
            transforms it. The function will be applied to the `where` attribute
            of all nodes when copying them.

        :returns: the root node of the tree copy.
        """
        new_where = deepcopy(self.where)
        if where_convert is not None:
            new_where = where_convert(new_where)
        with lock_ctx_deadlock_check(self.index_lock):
            new = TailIndexEntry(new_where, self.attr, self.vm)
            new.parent_index = self.parent_index
            if self.complete and keep_completed:
                new._total_count_filtered = self._total_count_filtered
                new._match_count = self._match_count
                new.complete = True
                new.index = deepcopy(self.index)
            else:
                for val in self.index:
                    new.index[val] = []

            for attr, sub_indexes in self.sub_indexes.items():
                if attr not in new.sub_indexes:
                    new.sub_indexes[attr] = {}
                for value, sub_index in sub_indexes.items():
                    new.sub_indexes[attr][value] = sub_index.clear_deep_copy(
                        keep_completed=keep_completed,
                        where_convert=where_convert
                    )
                    new.sub_indexes[attr][value].parent_index = new

        return new

    def mark_complete(self, white_list: Optional[T_ENTRY_FILTER] = None,
                      path: T_STR_TUPLE = tuple()):
        """Mark the node as complete (i.e. all tuples have been scanned)."""
        new_path = (*path, self.attr)
        if white_list is None or new_path in white_list:
            self.complete = True
        for sub_indexes in self.sub_indexes.values():
            for sub_index in sub_indexes.values():
                sub_index.mark_complete(white_list=white_list, path=new_path)


# attribute -> subscribed tail index entries
T_VM_SUBSCRIBERS = DefaultDict[str, Set[TailIndexEntry]]
T_VM_SUB_LOCKS = DefaultDict[str, Lock]


# THESIS: subscriber model + locking to propagate new attr. values throughout
#         the tree
class ValueManager:
    """
    Subscription manager to propagate newly discovered values through the tail
    index tree.
    """
    def __init__(self):
        self._subscribers = defaultdict(set)  # type: T_VM_SUBSCRIBERS
        self._subscriber_locks = defaultdict(Lock)  # type: T_VM_SUB_LOCKS
        # self.tree_structure_lock = ReadWriteLock()
        self.tree_structure_lock = RLock()

    def subscribe(self, entry: TailIndexEntry, attr: str):
        """Subscribe a tail index node to an attribute."""
        with lock_ctx_deadlock_check(self._subscriber_locks[attr]):
            assert (entry not in self._subscribers[attr]
                    or print(self._subscribers, attr, entry))
            self._subscribers[attr].add(entry)

    def unsubscribe(self, entry: TailIndexEntry, attr: str):
        """Unsubscribe a tail index node to an attribute."""
        with lock_ctx_deadlock_check(self._subscriber_locks[attr]):
            assert (entry in self._subscribers[attr]
                    or print(self._subscribers, attr, entry))
            self._subscribers[attr].remove(entry)

    def notify_new_value(self, attr: str, value: str):
        """
        Notify all tail index nodes that are subscribed to an attribute of a
        new value.
        """
        with lock_ctx_deadlock_check(self._subscriber_locks[attr]), \
             lock_ctx_deadlock_check(self.tree_structure_lock):
            for s in self._subscribers[attr]:
                s.notify_new_value(value)


T_ENTRIES_DICT = Dict[str, TailIndexEntry]


INDEXERS = {}


class IndexingProcess(Thread):
    """Process that stream tuples from the data source into the tail index."""
    daemon = True

    def __init__(self, ds: T_DS, entries: T_ENTRIES_DICT,
                 update_white_list: Optional[T_ENTRY_FILTER] = None):
        """
        :param ds: the data set from which to stream the tuples.
        :param entries: a `dict` that contains the root node to the tail index
            tree belonging to the top-level indexed attribute.
        :param update_white_list: The set of paths this indexing process is
            responsible for. Nodes in the tail index that are not on this path
            are left untouched.
        """
        super().__init__()
        self.ds = ds
        self.entries = entries
        self._paused = False
        self._stopped = False
        self.processed = 0
        self.complete = False
        self.white_list = update_white_list
        self.lock = Lock()
        self.condition = Condition()
        for path in update_white_list:
            INDEXERS[path] = self.getName()

    def run(self):
        cursor = tuples = None
        if self.entries:
            while True:
                with self.condition:
                    while self._paused:
                        self.condition.wait()
                # print('before lock', self.white_list)
                with lock_ctx_deadlock_check(self.lock):
                    # print('with lock', self.white_list)
                    if self._stopped:
                        break
                    # print('before get_tuples', self.white_list)
                    tuples, cursor = self.ds.get_tuples(1000, cursor)
                    # print('before sorting tuples', self.white_list)
                    for attr in self.entries:
                        self.entries[attr].index_tuples(
                            tuples, white_list=self.white_list
                        )
                    # print('after sorting tuples', self.white_list)
                    self.processed += len(tuples)
                if cursor is None:
                    break
        # print('after lock', self.white_list)
        if tuples is not None and cursor is None:
            self.complete = True
            for attr in self.entries:
                self.entries[attr].mark_complete(white_list=self.white_list)
        self._stopped = True

    def stop(self):
        self._stopped = True
        self.resume()

    def pause(self):
        # if self._stopped:
        #     raise RuntimeError("Cannot pause a stopped indexing process.")
        with self.condition:
            self._paused = True

    def resume(self):
        # if self._stopped:
        #     raise RuntimeError("Cannot resume a stopped indexing process.")
        with self.condition:
            if self._paused:
                self._paused = False
                self.condition.notify()


# look up all the stuff by sub_index, value
T_CURSOR_KEY = Tuple[TailIndexEntry, str]
T_CURSOR_DONE = DefaultDict[T_CURSOR_KEY, bool]
T_CURSOR_USED_INDEX_ENTRIES = DefaultDict[T_CURSOR_KEY, Set[Tuple]]
T_CURSOR_USING_INDEX = DefaultDict[T_CURSOR_KEY, bool]
T_CURSOR_CURSORS = DefaultDict[T_CURSOR_KEY, T_CURSOR]


class TailIndexCursor:
    """Cursor class used by `TailIndexDataSource`."""
    def __init__(self):
        self.done = defaultdict(lambda: False)  # type: T_CURSOR_DONE
        # THESIS: Storing all tuples drawn from an index so that they can
        #         be excluded if index gets dropped (bin too frequent) and
        #         tuples are drawn from data source (with new cursor)
        self.used_index_entries = \
            defaultdict(set)  # type: T_CURSOR_USED_INDEX_ENTRIES
        self.using_index = \
            defaultdict(lambda: False)  # type: T_CURSOR_USING_INDEX
        self.cursors = defaultdict(lambda: None)  # type: T_CURSOR_CURSORS
        self.stall_cache = dict()  # type Dict


T_INDEX_PICKS = Dict[Where, Tuple[TailIndexEntry, Any]]
# tuple of tuples of (and_clause, sub_indexes, values_to_pick, probability)
T_PROP_SORTING = Tuple[Tuple[Where, Set[TailIndexEntry], Set[str], float], ...]
T_PROP_SORTING_L = List[List[
    Union[Where, Set[TailIndexEntry], Set[str], float]
]]
T_PATH_CACHE = Dict[Where, Tuple[Tuple[str, Set[Tuple[str, bool]]], ...]]
T_SUB_INDEX_PICKS_CACHE = Dict[Where, Tuple[Set[TailIndexEntry], Set[str]]]


class _StallInterrupt(Exception):
    """
    Internal exception.

    Raised to indicate that `TailIndexDataSource.get_tuples` needs to stall
    because not enough tuples can be retrieved withing the demanded time window.
    """
    pass


class TailIndexDataSource(DataSource):
    """
    Data source that retrieves tuples for a query utilizing the tail index.
    """
    def __init__(self, index: "T_TAIL_INDEX", where: Where,
                 or_support: bool = False):
        """
        :param index: the tail index to use for retrievieng tuples.
        :param where: the selection of the query to retrieve tuples for.
        :param or_support: `True` if arbitrary or support is enabled.
        """
        self.index = index
        self.or_support = or_support
        self.where = where.canonical()
        assert not isinstance(self.where, bool)
        self.path_cache = {}  # type: T_PATH_CACHE
        self._sub_index_picks_cache = \
            TemporalCache()  # type: T_SUB_INDEX_PICKS_CACHE

        assert self.or_support or len(where.parts) <= 1 or where.or_exclusive()

        if not self.or_support:
            clauses = self.where.parts
        else:
            clauses = (
                join_clauses(clauses_)
                for clauses_ in powerset(self.where.parts, min_size=1)
            )
            clauses = (c for c in clauses if c is not False)

        for and_clause in clauses:
            path = index.get_path_read(and_clause)
            assert (len({p.attr for p in and_clause.parts}) == len(path)
                    or print("This might have been caused by using OR "
                             "queries without having or_support enabled."))
            eq = defaultdict(set)
            for p in and_clause.parts:
                eq[p.attr].add((p.val, p.neg))
            self.path_cache[and_clause] = tuple((attr, eq[attr])
                                                for attr in path)

    def pk_names(self) -> T_STR_TUPLE:
        return self.index.ds.pk_names

    def __get_sub_index_picks(self, and_clause: Where) -> \
            (Set[TailIndexEntry], Set[str]):
        """Select tail index nodes and values to fulfill the clause."""
        sub_indexes = {
            self.index._entries_read[self.path_cache[and_clause][0][0]]
        }  # type: Set[TailIndexEntry]
        for (_, val_neg), (attr, _) in zip(
                self.path_cache[and_clause][:-1],
                self.path_cache[and_clause][1:]
                ):
            any_val, any_neg = next(iter(val_neg))
            # either all are positive or all are negative.
            # Where._simplify_and_clause assures this
            # Furthermore, if one is positive there only is one value,
            # else the clause was unsatisfiable
            assert all(vn[1] == any_neg for vn in val_neg)
            if not any_neg:
                assert len(val_neg) == 1
                values = {any_val}
            else:
                # negated value means to pick from all other values.
                # this can't be cached as new values might be discovered later
                # while the indexing is still running.
                excluded_values = {vn[0] for vn in val_neg}
                any_sub_index = next(iter(sub_indexes))
                values = {
                    v
                    for v in any_sub_index.sub_indexes[attr].keys()
                    if v not in excluded_values and v is not DUMMY
                }
                assert all(ev not in values for ev in excluded_values)
                assert all((set(idx.sub_indexes[attr].keys())
                            == set(any_sub_index.sub_indexes[attr].keys()))
                           for idx in sub_indexes)
                # if not values:
                #     return set(), set()
            sub_indexes = {idx.sub_indexes[attr][v]
                           for idx in sub_indexes for v in values
                           # could be an not yet seen value
                           if (v in idx.sub_indexes[attr]
                               and idx.sub_indexes[attr][v].total_count)
                           }
        _, val_neg = self.path_cache[and_clause][-1]
        any_val, any_neg = next(iter(val_neg))
        assert all(vn[1] == any_neg for vn in val_neg)
        if not any_neg:
            assert len(val_neg) == 1
            values = {any_val}
        elif sub_indexes:
            excluded_values = (vn[0] for vn in val_neg)
            any_sub_index = next(iter(sub_indexes))
            values = {v for v in any_sub_index.index.keys()
                      if v not in excluded_values}
            assert all(set(idx.index.keys()) == set(any_sub_index.index.keys())
                       for idx in sub_indexes)
            # if not values:
            #     return set(), set()
        return sub_indexes, values

    def _get_sub_index_picks(self, and_clause: Where) -> \
            (Set[TailIndexEntry], Set[str]):
        """Like `__get_sub_index_picks`, but cached."""
        if and_clause in self._sub_index_picks_cache:
            return self._sub_index_picks_cache[and_clause]
        res = self.__get_sub_index_picks(and_clause)
        self._sub_index_picks_cache[and_clause] = res
        return res

    def _prop_sorted(self, cursor: TailIndexCursor) -> T_PROP_SORTING:
        """
        Get the tail index entries to draw from ordered by their weight.

        :param cursor: cursor that keeps track of the state.

        :returns: tuple of tuples of
            (and_clause, sub_indexes, values_to_pick, weight)
        """
        res = []  # type: T_PROP_SORTING_L
        for clause in self.where.parts:
            sub_indexs, values = self._get_sub_index_picks(clause)
            probs = []
            for si in sub_indexs:
                for v in values:
                    if si.global_frequency_estimate[v] == 0 and si.complete:
                        # empty sub population
                        cursor.done[si, v] = True
                    probs.append(0 if cursor.done[si, v]
                                 else si.global_frequency_estimate[v])
            prob = sum(p for p in probs if p is not None)
            res.append([clause, sub_indexs, values, prob])
        res.sort(key=lambda x: x[3])

        total_prob = 0
        for i, r in enumerate(res):
            clause, sub_indexs, values, prob = r
            if i and self.or_support:
                # i == 0 => nothing to exclude for first clause
                prev_clauses = [r[0] for r in res[:i]]
                for idxs in powerset(range(i-1)):
                    prev_clauses = [res[i][0] for i in idxs]
                    # intersection by joining (=conjugating) conjunctive clauses
                    joined_clause = join_clauses([*prev_clauses, clause])
                    if joined_clause is False:
                        # empty intersection => nothing to in- or exclude
                        continue
                    if len(idxs) % 2 == 1:
                        excl_sub_indexes, excl_values = \
                            self._get_sub_index_picks(joined_clause)
                        prob -= sum(
                            idx.global_frequency_estimate[val]
                            for idx in excl_sub_indexes for val in excl_values
                        )
                    else:
                        incl_sub_indexes, incl_values = \
                            self._get_sub_index_picks(joined_clause)
                        prob += sum(
                            idx.global_frequency_estimate[val]
                            for idx in incl_sub_indexes for val in incl_values
                        )
                r[3] = prob
            total_prob += prob
        # at this point the order might not follow the probability anymore.
        if total_prob:  # can be zero if no matching tuples have been found yet
            for i in range(len(res)):
                res[i][3] /= total_prob
        return tuple(map(tuple, res))

    def _get_tuples_sub_index(
            self, n: int, sub_index: TailIndexEntry,
            value: str, filter_clauses: Sequence[Where],
            cursor: TailIndexCursor,
            timeout: Optional[float]) -> \
            (T_DB_TUPLES, TailIndexCursor, bool):
        """
        Get tuples from a single sub index node or the data source if the
        desired subpopulation is not indexed.

        :param n: try to retrieve `n` tuples. Can return less.
        :param sub_index: the sub index entry to use.
        :param value: the value for which to retrieve the tuple
        :param filter_clauses: filter out all tuples that satisfy and of these
            filters.
        :param cursor: cursor that keeps track of the state.
        :param timeout: see `DataSource.get_tuples`.

        :returns:
            - Tuples of tuples that have been received
            - The new cursor that keeps track of the state
            - `True` if this function requests a stall because not enough tuples
               are available yet (already found tuples will be cached in
               cursor), `False` else.
        """
        # timeout None: blocking call, timeout > 0: with timout in seconds,
        # timeout -1: do 1 iteration
        res_tuples = tuple()
        acquire_deadlock_check(sub_index.index_lock)
        start = time.time()
        try:
            if isinstance(sub_index.index[value], int):
                # look up in data source
                # TODO: future work: if sub-population is too common move up
                #       the index tree and use the lowest existing bucket +
                #       filtering
                sub_index.index_lock.release()
                bucket_literal = WhereLiteral(sub_index.attr, value)
                if sub_index.where is None:
                    sub_index_clause = Where(parts=(bucket_literal,))
                else:
                    sub_index_clause = Where(parts=tuple(sorted(
                        {*sub_index.where.parts[0].parts, bucket_literal}
                    )))

                if cursor.using_index[sub_index, value]:
                    cursor.using_index[sub_index, value] = False
                    cursor.cursors[sub_index, value] = None
                tries = 0
                while timeout in (None, -1) or time.time() - start < timeout:
                    tries += 1
                    tuples, cursor.cursors[sub_index, value] = \
                        self.index.ds.get_tuples(
                            n - len(res_tuples),
                            cursor.cursors[sub_index, value]
                        )
                    # Discard clause matches of more frequent clause if where
                    # matches less frequent condition.
                    res_tuples += tuple(
                        t for t in tuples
                        if (sub_index_clause.eval(t)
                            and not any(c.eval(t) for c in filter_clauses)
                            and self.index.ds.to_pk(t) not in
                            cursor.used_index_entries[sub_index, value])
                    )
                    if cursor.cursors[sub_index, value] is None:
                        cursor.done[sub_index, value] = True
                        break
                    if len(res_tuples) >= n or timeout == -1:
                        break
            else:
                # use index
                if cursor.cursors[sub_index, value] is None:
                    cursor.using_index[sub_index, value] = True
                    sub_index_cursor = 0
                else:
                    sub_index_cursor = cursor.cursors[sub_index, value]
                from_ = sub_index_cursor
                to = sub_index_cursor + n
                while (not isinstance(sub_index.index[value], int)
                       and (timeout in (None, -1)
                            or time.time() - start < timeout)):
                    tuples = tuple(sub_index.index[value][from_:to])
                    sub_index.index_lock.release()
                    from_ += len(tuples)
                    cursor.cursors[sub_index, value] = from_
                    res_tuples += tuple(
                        t for t in tuples
                        if (not any(c.eval(t) for c in filter_clauses))
                    )
                    if (sub_index.complete
                            and from_ >= len(sub_index.index[value])):
                        cursor.done[sub_index, value] = True
                        break
                    if len(res_tuples) == n or timeout == -1:
                        break
                    # give time for more tuples to be streamed into tail-index
                    # concurrently
                    time.sleep(0.1)
                    acquire_deadlock_check(sub_index.index_lock)
                else:
                    sub_index.index_lock.release()
                # Keeping track of used index entries in case index entry gets
                # deleted if estimated frequency exceeds threshold
                # mid-computation.
                cursor.used_index_entries[sub_index, value].update(
                    self.index.ds.to_pk(res_tuples)
                )
            stall = len(res_tuples) < n and not cursor.done[sub_index, value]
            return res_tuples, cursor, stall
        finally:
            try:
                sub_index.index_lock.release()
            except RuntimeError:
                # lock is not acquired or acquired by another thread
                pass

    def _get_tuples_clause(
            self, n: int, clause: Where, filter_clauses: Sequence[Where],
            cursor: TailIndexCursor,
            timeout: Optional[float]) -> \
            (T_DB_TUPLES, TailIndexCursor, bool):
        """
        Get tuples for a single and clause.

        :param n: try to retrieve `n` tuples. Can return less.
        :param clause: the clause to retrieve tuples for.
        :param filter_clauses: filter out all tuples that satisfy and of these
            filters.
        :param cursor: cursor that keeps track of the state.
        :param timeout: see `DataSource.get_tuples`.

        :returns:
            - Tuples of tuples that have been received
            - The new cursor that keeps track of the state
            - `True` if this function requests a stall because not enough tuples
               are available yet (already found tuples will be cached in
               cursor), `False` else.
        """
        assert clause.is_and_clause()
        stall_cache_key = ("_get_tuples_clause", clause)
        try:
            tuples, buckets, dist, start_i = \
                cursor.stall_cache.pop(stall_cache_key)
        except KeyError:
            tuples = tuple()  # type: T_DB_TUPLES
            sub_indexes, values = self._get_sub_index_picks(clause)
            # each combination of sub_index, value is a bucket
            # buckets are pairwise disjoint (empty intersection)
            # => no PIE needed
            buckets = [
                [sub_index, value, sub_index.global_frequency_estimate[value]]
                for sub_index in sub_indexes for value in values
            ]  # type: List[List[Union[TailIndexEntry, str, float]], ...]
            buckets = [bucket for bucket in buckets if bucket[2] is not None]
            total_prop = sum(b[2] for b in buckets)
            if not total_prop:
                # no matches found yet
                assert not tuples
                return tuples, cursor
            for bucket in buckets:
                bucket[2] /= total_prop
            dist = multinomial(n, [b[2] for b in buckets])
            start_i = 0
        start_t = None
        for i in range(start_i, len(buckets)):
            sub_index, value, _ = buckets[i]
            n_ = dist[i]
            if not n_:
                continue
            if timeout is not None and timeout > 0:
                if start_t is None:
                    start_t = time.time()
                else:
                    timeout = time.time() - start_t
            sub_index_tuples, cursor, stall = self._get_tuples_sub_index(
                n_, sub_index, value, filter_clauses, cursor, timeout
            )
            tuples += sub_index_tuples
            if stall:
                dist[i] -= len(sub_index_tuples)
                cursor.stall_cache[stall_cache_key] = (tuples, buckets, dist, i)
                return (), cursor, True
            missing_tuples = n_ - len(sub_index_tuples)
            if missing_tuples and cursor is None and i < len(buckets):
                # sub index is empty: get tuples from next sub indexes

                left_props = [b[2] for b in buckets[(i + 1):]]
                left_sum = sum(left_props)
                left_normalized = [p / left_sum for p in left_props]
                dist[(i + 1):] += multinomial(missing_tuples, left_normalized)
        return tuples, cursor, False

    def get_tuples(self, n: int, cursor: T_CURSOR = None,
                   timeout: Optional[float] = -1) -> \
            (T_DB_TUPLES, T_CURSOR):
        """
        Get tuples from this tail index data source.

        :param n: try to retrieve `n` tuples. Can return less.
        :param cursor: cursor that keeps track of the state.
        :param timeout: see `DataSource.get_tuples`.

        :returns:
            - Tuples of tuples that have been received
            - The new cursor that keeps track of the state
        """
        # drawing from least frequent clause first
        tuples = tuple()
        if cursor is None:
            cursor = TailIndexCursor()
        with self._sub_index_picks_cache:
            stall_cache_key = "get_tuples"
            try:
                prop_sorted, start_i, dist, clause_probabilities, tuples = \
                    cursor.stall_cache.pop(stall_cache_key)
            except KeyError:
                prop_sorted = self._prop_sorted(cursor)
                clause_probabilities = np.array([x[3] for x in prop_sorted])
                if not clause_probabilities.sum():
                    # no tuples found yet => cannot estimate frequencies
                    if all(cursor.done[si, v]
                           for clause in self.where.parts
                           for si in self._get_sub_index_picks(clause)[0]
                           for v in self._get_sub_index_picks(clause)[1]):
                        # no more tuples can be found
                        return tuples, None
                    return tuples, cursor
                # Using sample from multinomial distribution. Other option would
                # be to roll a die for each sample and then decide which clause
                # to fulfill. This would cause up to n times connecting to the
                # db.
                start_i = 0
                dist = multinomial(n, clause_probabilities)

            clauses = [x[0] for x in prop_sorted]
            start_t = None
            for i in range(start_i, len(prop_sorted)):
                n_ = dist[i]
                if not n_:
                    continue
                clause, sub_index, value = prop_sorted[i][:3]
                if not n_:
                    continue
                if timeout is not None and timeout > 0:
                    if start_t is None:
                        start_t = time.time()
                    else:
                        timeout = time.time() - start_t
                filter_clauses = clauses[(i + 1):] if self.or_support else []
                clause_tuples, cursor, stall = self._get_tuples_clause(
                    n_, clause, filter_clauses, cursor, timeout
                )
                tuples += clause_tuples
                if stall:
                    cursor.stall_cache[stall_cache_key] = \
                        (prop_sorted, i, dist, clause_probabilities, tuples)
                    return (), cursor
                missing_tuples = n_ - len(clause_tuples)
                if missing_tuples and i + 1 < len(prop_sorted):
                    left_props = clause_probabilities[(i + 1):]
                    left_sum = sum(left_props)
                    left_normalized = [p/left_sum for p in left_props]
                    dist[(i + 1):] += multinomial(missing_tuples,
                                                  left_normalized)
            if len(tuples) >= n:
                return tuples, cursor
            else:
                # _get_tuples_clause is blocking. If no tuples are returned
                # the data sources have been fully scanned.
                return tuples, None


class TailIndex:
    """
    Behold! The Tail Index!

    This manages the tail index trees and indexing processes for queries.
    """
    def __init__(self, data_source: DataSource, or_support: bool = True):
        """
        :param data_source: the data source the tail index is based on.
        :param or_support: if `True` the tail index supports arbitrary
            disjunctions in the queries it manages. If `False`, only
            disjunctions of different values of the same attribute (basic or
            support) is supported.
        """
        self._ds = data_source
        self._vm = ValueManager()
        # _entries contains the root nodes for all tail index trees
        # key is the attribute the root node idexes
        self._entries = {}  # type: T_ENTRIES_DICT
        self._path_cache = {}  # type: Dict[Where, T_STR_TUPLE]
        self.or_support = or_support
        self._queries = set()  # type: Set[Query]
        self._paused_queries = set()  # type: Set[Query]
        self._providers = {}  # type: Dict[T_STR_TUPLE, IndexingProcess]
        self._provider_paths = \
            {}  # type: Dict[IndexingProcess, Set[T_STR_TUPLE]]
        self._subscribers = \
            defaultdict(set)  # type: DefaultDict[T_STR_TUPLE, Set[Query]]
        self._indexing_processes = set()  # type: Set[IndexingProcess]

    @property
    def ds(self):
        return self._ds

    def _get_path(self, attrs: Set[str], sub_index: TailIndexEntry) -> \
            T_STR_TUPLE:
        """
        Recursive helper to find the longest existing path in the tail in that
        only contains `attrs` attributes.


        :param attrs: the desired attributes of the path
        :param sub_index: the node from wich to start the search

        :returns: the path as tuple of attribute names
        """
        res = ()
        for attr in attrs:
            s = sub_index.sub_indexes.get(attr, None)
            if s is not None:
                if s:
                    p = ((attr,) +
                         self._get_path(attrs - {attr}, next(iter(s.values()))))
                else:
                    # yet unpopulated sub-indexes
                    p = (attr,)
                if len(p) > len(res):
                    res = p
        return res

    @property
    def _entries_write(self) -> T_ENTRIES_DICT:
        return self._entries

    _entries_read = _entries_write

    @property
    def _path_cache_write(self) -> Dict[Where, T_STR_TUPLE]:
        return self._path_cache

    _path_cache_read = _path_cache_write

    def get_path(self, and_clause: Where, entries: T_ENTRIES_DICT,
                 cache: Dict[Where, T_STR_TUPLE]) -> T_STR_TUPLE:
        """
        Get the longest existing attribute path in the tail index that belongs
        to an and clause.

        :param and_clause: the and clause to search the path for.
        :param entries: a dict that maps indexed attributes to all root nodes
            in the tail index forest.
        :param cache: a cache that will be used to look up the longest existing
            path for `and_clause`. If not found, it will be computed and stored
            in the cache.

        :returns: the path as tuple of attribute names
        """
        assert and_clause.is_and_clause() or print(repr(and_clause))
        res = cache.get(and_clause)
        if res is not None:
            return res
        res = tuple()
        attrs = {p.attr for p in and_clause.parts}
        for attr in entries:
            if attr in attrs:
                p = attr, *self._get_path(attrs - {attr}, entries[attr])
                if len(p) > len(res):
                    res = p
        cache[and_clause] = res
        return res

    def get_path_read(self, and_clause: Where) -> T_STR_TUPLE:
        return self.get_path(and_clause,
                             self._entries_read, self._path_cache_read)

    def get_path_write(self, and_clause: Where) -> T_STR_TUPLE:
        return self.get_path(and_clause,
                             self._entries_write, self._path_cache_write)

    @property
    def _indexing_processes_to_lock(self) -> Iterable[IndexingProcess]:
        """
        All indexing processes that need to be locked when updating the tail
        index forest's structure.
        """
        return self._indexing_processes

    @property
    def _indexing_process_lock(self):
        """
        A one-time-use environment that locks all indexing processes returned by
        `self._indexing_processes_to_lock`.
        """
        other = self

        class Locker:
            def __init__(self):
                self._state = 0
                self._acquired_locks = []

            def __enter__(self):
                assert self._state == 0 and "don't reuse the index locker"
                self._state = 1
                for ip in other._indexing_processes_to_lock:
                    if not ip.is_alive():
                        continue
                    ip.lock.acquire()
                    self._acquired_locks.append(ip.lock)

            def __exit__(self, *args, **kwargs):
                assert self._state == 1 and "don't reuse the index locker"
                self._state = 2
                for lock in self._acquired_locks:
                    lock.release()

        return Locker()

    def _extend_path(self,
                     extensions: Iterable[Tuple[T_STR_TUPLE, T_STR_TUPLE]]) -> \
            Set[T_STR_TUPLE]:
        """
        Extends the tail index forest to contain the given paths.

        :param extensions: iterable of tuples `(existing_path, to_create)` where
            `existing_path` is a tuple of attribute strings that describes the
            existing path in the forest that shall be extended. `to_create` is a
            tuple of attribute strings that describes the path that shall be
            created.

        :returns: the set of newly created paths as tuple of attribute names.
        """
        new_paths = set()
        if not extensions:
            return new_paths

        with self._indexing_process_lock:
            for extension in extensions:
                existing_path, to_create = extension
                assert to_create
                if existing_path:
                    new_paths.update(
                        existing_path[:1] + new_path
                        for new_path in
                        self._entries_write[existing_path[0]].add_sub_indexes(
                            existing_path[1:] + to_create
                        )
                    )
                else:
                    if to_create[0] in self._entries_write:
                        entry = self._entries_write[to_create[0]]
                    else:
                        entry = TailIndexEntry(None, to_create[0], self._vm)
                        self._entries_write[to_create[0]] = entry
                        new_paths.add(to_create[:1])
                    new_paths.update(
                        to_create[:1] + new_path
                        for new_path in entry.add_sub_indexes(to_create[1:])
                    )

        return new_paths

    def _iter_clauses(self, q: Query) -> Iterator[Where]:
        """
        Iterate over all clauses or all combinations of clauses (if or
        support is enabled) in a query.
        """
        q = q.canonical()
        if q.where is True:
            return (Where(parts=tuple()) for _ in [1])
        elif q.where is False:
            return (_ for _ in [])
        if not self.or_support:
            clauses = q.where.parts
        else:
            # Or support means indexing all clause combinations to be able to
            # estimate the likelihood of intersections which is needed for in-
            # and exclusion principle. We only need the superset modulo sorted
            # prefix. Question: how much is left? Still O(exp)? Also argue with
            # limited amount of attributes inspected simultaneously.
            clauses = (
                join_clauses(clauses_)
                for clauses_ in powerset(q.where.parts, min_size=1)
            )
            clauses = (c for c in clauses if c is not False)

        return clauses

    def add_query(self, q: Query):
        """
        Register a query. Updated tail index forest structure and starts an
        indexing process.

        :param q: the query to add.
        """
        q = q.canonical()
        if q in self._queries:
            self.resume_query(q)
        if (not self.or_support
                and not isinstance(q.where, bool)
                and len(q.where.parts) > 1
                and not q.where.or_exclusive()):
            raise ValueError(
                "Or support must be enabled when indexing queries "
                "with OR in WHERE statement: %s" % str(q))

        self._queries.add(q)
        path_extensions = set()  # type: Set[Tuple[T_STR_TUPLE, T_STR_TUPLE]]

        clauses = tuple(self._iter_clauses(q))

        for clause in clauses:
            for group in q.groups:
                path = self.get_path_write(clause)
                if len(path) < len({p.attr for p in clause.parts}):
                    missing = (set(map(lambda p: p.attr, clause.parts))
                               - set(path))  # type: Set[str]
                    path_extensions.add((path, tuple(missing) + (group,)))
                    del self._path_cache_write[clause]
                else:
                    path_extensions.add((path, (group,)))

        new_paths = self._extend_path(path_extensions)

        if new_paths:
            indexing_process = self._start_indexing(tuple(new_paths))
            for new_path in new_paths:
                assert new_path not in self._providers
                self._providers[new_path] = indexing_process
            self._provider_paths[indexing_process] = new_paths

        self._paused_queries.add(q)
        self.resume_query(q)  # make sure all providers are running

    def _start_indexing(self, new_paths: Sequence[Sequence[str]]) -> \
            IndexingProcess:
        """
        Create an indexing process responsible for updating all final nodes
        of `new_paths`.

        :param new_paths: The paths to the nodes the new indexing process is
            responsible for.
        """
        white_list = {tuple(path) for path in new_paths}
        indexing_process = IndexingProcess(self._ds, self._entries,
                                           update_white_list=white_list)
        self._indexing_processes.add(indexing_process)
        indexing_process.start()
        return indexing_process

    def pause_query(self, q: Query):
        """'
        Pause a query and its indexing process if no other query depends on the
        indexer.

        :param q: the query to be paused.

        :raises ValueError: if the query hasn't been registered with `add_query`
            before.
        """
        q = q.canonical()
        if q not in self._queries:
            raise ValueError("Cannot pause query before started: %s" % str(q))
        if q in self._paused_queries:
            return
        paths = set()
        for clause in self._iter_clauses(q):
            path = self.get_path_write(clause)
            assert len(path) == len({p.attr for p in clause.parts})
            paths |= {path + (group,) for group in q.groups}
            if path:
                paths.add(path)

        for path in paths:
            subscribed_queries = self._subscribers[path]
            subscribed_queries.remove(q)
            if not subscribed_queries:
                indexer = self._providers[path]
                indexer_paths = self._provider_paths[indexer]
                if all(not self._subscribers[p] for p in indexer_paths):
                    # provider has no subscribers left
                    indexer.pause()

        self._paused_queries.add(q)

    def pause_all_queries(self):
        """`pause_query` for all registered queries."""
        for q in self._queries:
            self.pause_query(q)
        assert all(not x for x in self._subscribers.values())

    def resume_query(self, q: Query):
        """
        Resume a query and restart its indexing process if paused.

        :param q: the query to be resumed.

        :raises ValueError: if the query hasn't been registered with `add_query`
            before.
        """
        q = q.canonical()
        if q not in self._queries:
            raise ValueError("Cannot pause query before started: %s" % str(q))
        if q not in self._paused_queries:
            return
        for clause in self._iter_clauses(q):
            path = self.get_path_write(clause)
            assert len(path) == len({p.attr for p in clause.parts})
            for group in q.groups:
                g_path = path + (group,)
                self._subscribers[g_path].add(q)
                self._providers[g_path].resume()
            if path:
                self._subscribers[path].add(q)
                self._providers[path].resume()
        self._paused_queries.remove(q)

    def stop(self):
        """Stop all indexing processes"""
        for indexer in self._provider_paths:
            indexer.stop()
            indexer.join(10)
            if indexer.is_alive():
                print("Couldn't stop indexer :/", file=sys.stderr)

    def __getitem__(self, item: Where) -> T_DS:
        """Get a `DataSource` for the filter provided by `item`."""
        if isinstance(item, Where):
            item = item.canonical()
        if item is True:
            return self.ds
        elif item is False:
            return EmptyDataSource()
        if (not self.or_support and len(item.parts) > 1
                and not item.or_exclusive()):
            raise ValueError(
                "Or support must be enabled when executing queries with OR in "
                "WHERE statement: %s" % repr(item)
            )
        return TailIndexDataSource(self, item, or_support=self.or_support)


T_TAIL_INDEX = TypeVar("T_TAIL_INDEX", bound=TailIndex)

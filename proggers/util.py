from collections import defaultdict
from contextlib import contextmanager
from itertools import chain, combinations
import sys
from threading import (
    Condition,
    get_ident,
    RLock,
)
from time import monotonic as _time
import traceback
from typing import (
    Iterable,
    Optional,
    Tuple,
    TypeVar,
)

from .query import (
    Where,
    WhereOp,
)


T = TypeVar("T")


# https://docs.python.org/3/library/itertools.html#itertools-recipes
def powerset(iterable: Iterable[T], min_size: int = 0,
             max_size: Optional[int] = None) -> \
        Iterable[Tuple[T, ...]]:
    """Computes the powerset of an iterable (each combination of items).

    Examaple:
        powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)

    :param iterable: the iterable of which to build the powerset.
    :min_size: skip combinations that are smaller than `min_size`.
        `0` will skip nothing.
    :max_size: skip combinations that are larger than `max_size`.
        `None` will skip nothing.
    """
    s = list(iterable)
    if max_size is None:
        max_size = len(s)
    return chain.from_iterable(combinations(s, r)
                               for r in range(min_size, max_size + 1))


def join_clauses(clauses: Iterable[Where]) -> Optional[Where]:
    """Turn multiple conjunctive clauses into one big conjunctive clause"""
    clauses = tuple(clauses)
    assert all(c.is_and_clause() for c in clauses)
    res = Where(
        parts=tuple(part for clause in clauses for part in clause.parts),
        op=WhereOp.AND
    ).canonical()
    if res is False:
        # unsatisfiable clause
        return False
    return res.parts[0]


class TemporalCache(dict):
    """
    Dictionary that provide an "active"-environment.

    The `dict` will ignore entries if the env. is not active and will clear
    itself when the env. is left.

    Example:
         tc = TemporalCache()
         tc["foo"] = "bar"  # -> tc == {}
         with tc:
            tc["foo"] = "baz"  # -> tc == {"foo": "baz"}
         tc == {}  # -> True
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._active = False

    def __enter__(self):
        self._active = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._active = False
        self.clear()

    def __setitem__(self, key, value):
        if self._active:
            super().__setitem__(key, value)

    def update(self, *args, **kwargs):
        if self._active:
            super().update(*args, **kwargs)

    def setdefault(self, *args, **kwargs):
        if self._active:
            super().setdefault(*args, **kwargs)


class ReadWriteLock:
    """
    Lock that allows multiple concurrent readers but only one writer at a time.

    This implementation has a writer prioritization: as soon as one writer wants
    to acquire the lock, no new readers can acquire lock until no more writers
    are waiting for the lock.
    """
    class ReadLock:
        def __init__(self, rw_lock):
            self._rw_lock = rw_lock

        def acquire(self, *args, **kwargs):
            return self._rw_lock.acquire_read(*args, **kwargs)

        def release(self, *args, **kwargs):
            return self._rw_lock.release_read(*args, **kwargs)

        def __enter__(self):
            return self.acquire()

        def __exit__(self, exc_type, exc_val, exc_tb):
            return self.release()

    class WriteLock:
        def __init__(self, rw_lock):
            self._rw_lock = rw_lock

        def acquire(self, *args, **kwargs):
            return self._rw_lock.acquire_write(*args, **kwargs)

        def release(self, *args, **kwargs):
            return self._rw_lock.release_write(*args, **kwargs)

        def __enter__(self):
            return self.acquire()

        def __exit__(self, exc_type, exc_val, exc_tb):
            return self.release()

    def __init__(self):
        self._condition = Condition(RLock())
        self._readers = defaultdict(int)
        self._writers_queued = defaultdict(int)
        self.read_lock = ReadWriteLock.ReadLock(self)
        self.write_lock = ReadWriteLock.WriteLock(self)

    def acquire_read(self, blocking=True, timeout=-1):
        if not blocking and timeout is not None:
            raise ValueError("can't specify timeout for non-blocking acquire")
        rc = False
        endtime = None
        with self._condition:
            while set(self._writers_queued) - {get_ident()}:
                if not blocking:
                    break
                if timeout >= 0:
                    if endtime is None:
                        endtime = _time() + timeout
                    else:
                        timeout = endtime - _time()
                        if timeout <= 0:
                            break
                self._condition.wait(None if timeout < 0 else timeout)
            else:
                self._readers[get_ident()] += 1
                rc = True
        return rc

    def release_read(self):
        with self._condition:
            if self._readers[get_ident()] > 1:
                self._readers[get_ident()] -= 1
            else:
                del self._readers[get_ident()]
            if not self._readers:
                self._condition.notifyAll()

    def acquire_write(self, blocking=True, timeout=-1):
        if not blocking and timeout is not None:
            raise ValueError("can't specify timeout for non-blocking acquire")
        if not self._condition.acquire(blocking=blocking, timeout=timeout):
            return False
        endtime = None
        self._writers_queued[get_ident()] += 1
        while set(self._readers) - {get_ident()}:
            if not blocking:
                break
            if timeout >= 0:
                if endtime is None:
                    endtime = _time() + timeout
                else:
                    timeout = endtime - _time()
                    if timeout <= 0:
                        break
            self._condition.wait(None if timeout < 0 else timeout)
        else:
            return True
        self._writers_queued[get_ident()] -= 1
        return False

    def release_write(self):
        if self._writers_queued[get_ident()] > 1:
            self._writers_queued[get_ident()] -= 1
        else:
            del self._writers_queued[get_ident()]
        self._condition.notifyAll()
        self._condition.release()


def acquire_deadlock_check(lock, timeout=5):
    """Wrapper for lock.acquire to print a warning if lock is unresponsive."""
    result = lock.acquire(timeout=timeout)
    if not result:
        try:
            raise RuntimeError('This might be a dead lock')
        except RuntimeError:
            f = [f[0] for _, f in zip(range(4), traceback.walk_stack(None))][-1]
            traceback.print_stack(f, file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
        lock.acquire()


@contextmanager
def lock_ctx_deadlock_check(lock, timeout=5):
    acquire_deadlock_check(lock, timeout=timeout)
    yield
    lock.release()

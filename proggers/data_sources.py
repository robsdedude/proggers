import abc
import base64
import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor
from typing import (
    Iterator,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)
import uuid


T = TypeVar("T")
T_CURSOR = Optional[T]
T_DB_TUPLE = dict
T_DB_TUPLES = Tuple[T_DB_TUPLE, ...]


class DataSource(abc.ABC):
    """Base class for all data sources."""
    @abc.abstractmethod
    def get_tuples(self, n: int, cursor: T_CURSOR = None,
                   timeout: Optional[float] = -1) -> \
            (T_DB_TUPLES, T_CURSOR):
        """
        Get up to n tuples from the data source. To be implemented.

        :param n: how many tuples to retrieve at most.
        :param cursor: the cursor returned from this function's last call or
            `None` if called for the first time. The cursor contains the
            internal state used to make sure every tuples is retrieved exactly
            once.
        :param timeout: `-1` to try once to retrieve `n` tuples and return how
            ever many were available.
            `None` to repeat the retrieving until `n` tuples are available.
            Any positive `float` to specify a rough upper time limit in
            seconds. Some implementations might ignore this parameter.


        :returns 1) Tuple of retrieved tuples (in dict form)
            2) The new cursor or None if the data source was fully scanned
        """
        pass

    @property
    @abc.abstractmethod
    def pk_names(self) -> Tuple[str, ...]:
        """The key or keys that make the primary key. To be implemented."""
        pass

    def to_pk(self, tuples: Union[Sequence[T_DB_TUPLE], T_DB_TUPLE]) -> \
            Tuple[tuple, ...]:
        """
        Extract primary key values for one or more tuples.

        :param tuples: A tuple or a sequence of tuples (in dict form).

        :returns A tuple of values or sequence of tuples of values. The values
            are the ones corresponding to the keys returned by `pk_names`.
        """
        if isinstance(tuples, dict):
            tuples = (tuples,)
        return tuple(map(lambda t: tuple(t[pk] for pk in self.pk_names),
                         tuples))


class EmptyDataSource(DataSource):
    """Data source that contains no tuples."""
    def get_tuples(self, n: int, cursor: T_CURSOR = None,
                   timeout: Optional[float] = -1) -> \
            (T_DB_TUPLES, T_CURSOR):
        return tuple(), None

    @property
    def pk_names(self):
        return tuple()


class PostgresSource(DataSource):
    """Data source that draws tuples from on table of a postgres data base."""
    class PGCursor:
        """Cursor class used by `PostgresSource`."""
        def __init__(self, con, table):
            self._name = base64.b64encode(uuid.uuid4().bytes) \
                               .decode('ascii').replace("=", "_")
            self._cursor = con.cursor(self._name, scrollable=False,
                                      cursor_factory=RealDictCursor,
                                      withhold=True)
            self._cursor.execute(
                # shuffle rows on the fly for testing. Would be more efficient
                # to do in preprocessing.
                sql.SQL("SELECT * FROM {} ORDER BY RANDOM()")
                   .format(sql.Identifier(table))
            )
            self.__iter__ = self._cursor.__iter__
            self.fetchmany = self._cursor.fetchmany

        def __del__(self):
            self._cursor.close()

    def __init__(self, db_host: str, db_name: str, db_user: str, db_pass: str,
                 table: str):
        self._con = psycopg2.connect(database=db_name, user=db_user,
                                     host=db_host, password=db_pass)
        self._con.autocommit = True
        self._table = table
        self._table_id = sql.Identifier(table)

        cur = self._con.cursor(cursor_factory=RealDictCursor)
        cur.execute(sql.SQL("SELECT * FROM {} LIMIT 1;").format(self._table_id))
        self.__pk = tuple(sorted(map(str, cur.fetchone().keys())))

    @property
    def pk_names(self) -> Tuple[str, ...]:
        return self.__pk

    def get_tuples(self, n: int, cursor: T_CURSOR = None,
                   timeout: Optional[float] = -1) -> \
            (T_DB_TUPLES, T_CURSOR):
        if cursor is None:
            cursor = PostgresSource.PGCursor(self._con, self._table)
        rows = tuple({k: str(v) for k, v in row.items()}
                     for row in cursor.fetchmany(n))
        if len(rows) < n:
            del cursor
            return rows, None
        return rows, cursor


T_DS = TypeVar("T_DS", bound=DataSource)


def get_data(n: int, ds: T_DS, *args, **kwargs) -> Iterator[T_DB_TUPLES]:
    """
    Helper method to draw all tuples from a given data source.

    :param n: the desired chunk size to retrieve tuples in.
    :param ds: the data source to draw tuples from.
    :param args: passed to `ds.get_tuples`.
    :param kwargs: passed to `ds.get_tuples`.

    :returns: Iterator that yields a tuple of tuples (in dict form)
    """
    res, cursor = ds.get_tuples(n)
    if not cursor:
        return
    while cursor:
        yield res
        res, cursor = ds.get_tuples(n, cursor, *args, **kwargs)

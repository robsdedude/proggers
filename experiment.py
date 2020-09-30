import datetime
from contextlib import ExitStack
import csv
import functools
import os
import time
import sys

import psycopg2
from psycopg2 import sql
import scipy.stats

import proggers
from proggers.query import parse_where as p_where

from .db_settings import DB_HOST
from .db_settings import DB_USER
from .db_settings import DB_PASSWORD
from .db_settings import DB_NAME
from .db_settings import DB_TABLE_NAME


print = functools.partial(print, flush=True)

ds = proggers.data_sources.PostgresSource(
    DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_TABLE_NAME
)


# Galakatos et al. do not mention a confidence level or
# error rate (i.e. how likely it is that the true error is not within
# the given threshold). Their paper assumes a factor of 1 which corresponds to a
# error rate of (1 - scipy.stats.norm.cdf(1)) * 2 = 0.31731050786291415
ERROR_RATE = 0.31731050786291415
# the paper calls this confidence of 3.5 sigma
# 1 - (scipy.stats.norm.cdf(3.5) - scipy.stats.norm.cdf(-3.5))
THRESHOLD = 0.0004652581580710802

ERROR_THRESHOLD = THRESHOLD / scipy.stats.norm.ppf(1 - .5 * ERROR_RATE)

# ERROR_THRESHOLD = 1 - 0.999534741841929  # sigma = 3.5
# ERROR_THRESHOLD = 1 - 0.682689492  # sigma = 1


SEX_MALE = 'sex=Male'
SEX_FEMALE = 'sex=Female'
SEX_NOT_FEMALE = 'sex<>Female'
EDU_PHD = 'education="Doctorate degree(PhD EdD)"'
AGE_20_TO_40 = '(age=20 OR age=30)'


queries = (
    proggers.Query(groups=('sex',)),
    proggers.Query(groups=('education',)),
    proggers.Query(groups=('education',), where=p_where(SEX_FEMALE)),
    proggers.Query(groups=('education',), where=p_where(SEX_MALE)),
    proggers.Query(groups=('sex', 'education')),
    proggers.Query(groups=('sex',), where=p_where(EDU_PHD)),
    proggers.Query(groups=('income',)),
    proggers.Query(groups=('income',), where=p_where(EDU_PHD)),
    proggers.Query(groups=('income', 'sex')),
    proggers.Query(groups=('income',), where=p_where(SEX_FEMALE)),
    proggers.Query(groups=('income',)),
    proggers.Query(groups=('income',), where=p_where(SEX_FEMALE)),
    proggers.Query(groups=('income',), where=p_where(SEX_NOT_FEMALE)),
    (
        proggers.Query(groups=('income',),
                       where=p_where(' AND '.join((SEX_FEMALE, EDU_PHD)))),
        proggers.Query(groups=('income',),
                       where=p_where(' AND '.join((SEX_NOT_FEMALE, EDU_PHD))))
    ),
    proggers.Query(groups=('age',)),
    (
        proggers.Query(groups=('income',), where=p_where(' AND '.join((
            AGE_20_TO_40, SEX_FEMALE, EDU_PHD
        )))),
        proggers.Query(groups=('income',), where=p_where(' AND '.join((
            AGE_20_TO_40, SEX_NOT_FEMALE, EDU_PHD
        )))),
    ),
)


def get_exact_results(query_groups):
    def transform_where(where):
        if isinstance(where, proggers.query.WhereLiteral):
            return ("{}" + ("<>" if where.neg else "=") + "%s",
                    [where.attr], [where.val])
        elif isinstance(where, proggers.query.Where):
            strings, f_args, q_args = \
                zip(*(transform_where(p) for p in where.parts))
            concat = (" " + where.op.name + " ").join
            return "(" + concat(strings) + ")", sum(f_args, []), sum(q_args, [])
        else:
            print(type(where))
            assert False

    def query_to_sql(query):
        format_args = []
        query_args = []
        sql_ = "SELECT " + ", ".join(["{}"] * len(query.groups)) + ", COUNT(*) "
        format_args += sorted(query.groups)
        sql_ += "FROM {} "
        format_args.append(DB_TABLE_NAME)
        if query.where is not True:
            w_str, w_format_args, w_query_args = transform_where(query.where)
            sql_ += "WHERE " + w_str
            format_args += w_format_args
            query_args += w_query_args
        sql_ += "GROUP BY " + ", ".join(["{}"] * len(query.groups)) + ";"
        format_args += sorted(query.groups)
        format_args = tuple(map(sql.Identifier, format_args))
        return sql.SQL(sql_).format(*format_args), query_args

    con = psycopg2.connect(database=DB_NAME, user=DB_USER,
                           host=DB_HOST, password=DB_PASSWORD)
    cur = con.cursor()

    res = []
    for query_group in query_groups:
        group_res = []
        for query in query_group:
            q, q_args = query_to_sql(query)
            cur.execute(q, q_args)
            rows = [[tuple(map(str, row[:-1])), row[-1]]
                    for row in cur.fetchall()]
            sum_ = sum(r[-1] for r in rows)
            for i in range(len(rows)):
                rows[i][-1] /= sum_
            group_res.append(dict(rows))
        res.append(tuple(group_res))
    return tuple(res)


queries = tuple((q,) if isinstance(q, proggers.Query) else q for q in queries)
queries = tuple(tuple(q.canonical() for q in group) for group in queries)
exact_results = get_exact_results(queries)


def get_real_error(true_hist, approx_hist):
    sum_approx = sum(approx_hist.values())
    found_attrs = set()
    error = 0
    for attrs in true_hist.keys():
        if attrs not in approx_hist:
            p_approx = 0
        else:
            found_attrs.add(attrs)
            p_approx = approx_hist[attrs] / sum_approx
        error += abs(p_approx - true_hist[attrs])
    assert len(found_attrs) == len(approx_hist) or print(true_hist, approx_hist)
    error /= len(true_hist)
    return error


def print_index_processes(qe: proggers.QueryEngine):
    print("Indexers:",
          {path: {k: v for k, v in prov.__dict__.items()
                  if k in ('_paused', '_stopped',
                           'complete', 'processed')}
           for path, prov in qe.tail_index._providers.items()})


def experiment(qe, verbose, error_fn, time_fn, think_time=0):
    error_writer = time_writer = None
    with ExitStack() as stack:
        if error_fn:
            error_writer = csv.writer(stack.enter_context(open(error_fn, 'w',
                                                               1)))
        if time_fn:
            time_writer = csv.writer(stack.enter_context(open(time_fn, 'w', 1)))

        t = time.time()
        for query in queries[0]:
            qe.start_query(query)

        tmp = 0
        for i in range(len(queries)):
            error_times = ["time", i, ""]
            estimated_errors = [["estimate", i, j]
                                for j in range(len(queries[i]))]
            real_errors = [["real", i, j] for j in range(len(queries[i]))]
            while True:
                # tmp += verbose or error_writer is not None
                tmp += 1
                hists, errs = zip(*(qe.result_cache.get(q) for q in queries[i]))
                done = (
                    tmp == -5000
                    or all(q in qe.finished_queries for q in queries[i])
                    or (all(hist is not None for hist in hists)
                        and max(err.error for err in errs) < ERROR_THRESHOLD)
                )
                if done:
                    if tmp == -5000:
                        print("qe timed out!\n"
                              "queries: %s\n"
                              "queries finished: %s\n"
                              "errors: %s\n"
                              "histograms: %s\n" %
                              (queries[i], qe.finished_queries,
                               [err.error if hist is not None else None
                                for hist, err in zip(hists, errs)]
                               , hists),
                              file=sys.stderr)
                    took = time.time() - t
                    if verbose:
                        print()
                        print("=" * 80)
                        print("query(s) %i took %fs" % (i, took))
                        for j in range(len(hists)):
                            print('-' * 20)
                            print("query: %s" % queries[i][j])
                            print(hists[j])
                        print("=" * 80)
                        tmp = 0
                    if time_writer is not None:
                        time_writer.writerow([i, took])
                if verbose and tmp % 100 == 0:
                    print(tuple(sum(v for v in hist.values())
                                for hist in hists if hist is not None))
                    print(tuple(
                        err if err is None else (err.error - ERROR_THRESHOLD)
                        for err in errs
                    ))
                if (error_writer is not None and tmp % 10 == 0
                        or done):
                    error_times.append(
                        took if done else (time.time() - t)
                    )
                    for j in range(len(queries[i])):
                        if errs[j] is not None:
                            estimated_errors[j].append(errs[j].error)
                        else:
                            estimated_errors[j].append("")
                        if hists[j] is not None:
                            real_errors[j].append(
                                get_real_error(exact_results[i][j], hists[j])
                            )
                        else:
                            real_errors[j].append("")
                if done:
                    if think_time:
                        if verbose:
                            print("Emulating think time of %i seconds" %
                                  think_time)
                        time.sleep(think_time)
                        if verbose:
                            print("Enough thinking. Now work for your money!")
                    break
                time.sleep(0.05)
            if error_writer is not None:
                assert (len(estimated_errors)
                        == len(real_errors)
                        == len(queries[i]))
                error_writer.writerow(error_times)
                for j in range(len(estimated_errors)):
                    error_writer.writerow(estimated_errors[j])
                    error_writer.writerow(real_errors[j])
            for query in queries[i]:
                if verbose:
                    print("Pausing query %s" % query)
                qe.pause_query(query)
                if verbose:
                    print("Paused")

            if verbose and qe.tail_index is not None:
                time.sleep(2)
                print_index_processes(qe)
            if i < len(queries) - 1:
                t = time.time()
                for query in queries[i + 1]:
                    if verbose:
                        print("Staring query %s" % query)
                    qe.start_query(query)
                    if verbose:
                        print("Started")

            if verbose and qe.tail_index is not None:
                time.sleep(2)
                print_index_processes(qe)


def main(args_):
    verbose = args_.v
    repetitions = args_.n
    out_path = args_.o
    dump_error = args_.dump_error
    dump_time = args_.dump_time
    think_time = args.think_time
    use_tailindex = not args_.no_tailindex
    or_support = not args.no_or_support

    error_fn = time_fn = None
    for i in range(repetitions):
        print("Repetition %i" % i)
        qe = proggers.QueryEngine(ds, tail_index=use_tailindex,
                                  or_support=or_support)
        qe.start()
        if repetitions:
            exp_count_postfix = "_%%0%ii" % len(str(repetitions)) % i
        else:
            exp_count_postfix = ""
        if dump_error:
            error_fn = "error" + exp_count_postfix + ".csv"
            error_fn = os.path.join(out_path, error_fn)
        if dump_time:
            time_fn = "time" + exp_count_postfix + ".csv"
            time_fn = os.path.join(out_path, time_fn)
        experiment(qe, verbose, error_fn, time_fn, think_time=think_time)
        qe.stop()
        qe.join(120)
        if qe.isAlive():
            raise RuntimeError("Couldn't stop query engine :/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", action="store_true",
                        help="Give verbose output (slows down the computation)")
    parser.add_argument("-n", type=int, default=1,
                        help="Number of times to repeat the experiment")
    parser.add_argument("-o", type=str, default="./output",
                        help="where to output dumps defaults to CWD/output")
    parser.add_argument("--dump-error", action="store_true",
                        help="dump estimated error vs true error into csv file")
    parser.add_argument("--dump-time", action="store_true",
                        help="dump time per query into csv file. should not be "
                             "combined with -v for higher accuracy.")
    parser.add_argument("--no-tailindex", action="store_true",
                        help="disable the tailindex")
    parser.add_argument("--no-or-support", action="store_true",
                        help="disable the or-support")
    parser.add_argument("--think-time", type=float, default=0,
                        help="How long the simlated think time between queries "
                             "should be (default: 0).")

    args = parser.parse_args()

    now = datetime.datetime.now().replace(microsecond=0).isoformat()
    args.o = os.path.join(args.o, now)

    os.makedirs(args.o, exist_ok=True)

    main(args)

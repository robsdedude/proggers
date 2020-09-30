#!/usr/bin/python3
import argparse
import csv
import datetime
from operator import itemgetter
import os
import sys
import re
import tqdm
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT


THIS_DIR = os.path.dirname(os.path.realpath(__file__))


def create_table(con):
    with open(os.path.join(THIS_DIR, 'schema.db'), 'r') as fd:
        schema = fd.read()

    # drop all existing public table
    cur = con.cursor()
    try:
        cur.execute(sql.SQL("""
            select 'drop table if exists "' || tablename || '" cascade;' 
                from pg_table
            where schemaname = 'public';
        """))
    except psycopg2.ProgrammingError as e:
        if 'relation "pg_table" does not exist' in e.pgerror:
            pass
    else:
        del_query = ' '.join(map(itemgetter(0), cur))
        if del_query:
            cur.execute(sql.SQL(del_query))

    cur.execute(sql.SQL(schema))
    cur.close()


BIN_COLS = (
    # tuples of column index and size of bins
    # age (0-90)
    (0, 10),
    # wage per hour (0-9999)
    (5, 1000),
    # capital gains (0-99999)
    (16, 10000),
    # capital losses (0-4608)
    (17, 500),
    # dividends from stocks (0-99999)
    (18, 10000),
    # weeks worked in year (0-52)
    (39, 5),
)


def _read_csv_and_preprocess(path):
    row_len = None
    int_cols = []
    rows = []
    with open(path, 'r') as fd:
        for row in csv.reader(fd):
            if row_len is None:
                row_len = len(row)
                int_cols = [True] * row_len
                int_cols[24] = False
            else:
                assert len(row) == row_len
            row = [None if c[:16] in (' ?', ' Not in universe') else c.strip()
                   for c in row]
            for idx, bin_size in BIN_COLS:
                if row[idx] is not None:
                    row[idx] = int(row[idx])
                    row[idx] = (row[idx] // bin_size) * bin_size
            # instance weight
            if row[24] is not None:
                row[24] = float(row[24])
            for i in (i for i in range(row_len) if int_cols[i]):
                if row[i] is None:
                    continue
                try:
                    int(row[i])
                except ValueError:
                    int_cols[i] = False
            rows.append(row)
    rows = [[int(c) if int_cols[i] else c
             for i, c in enumerate(r)]
            for r in rows]
    return rows


def fill_table(con, path, row_chunks=10000):
    def execute(cur_, tuples_, table_):
        val_arg_template = '(' + ','.join(['%s'] * len(tuples_[0])) + ')'
        args_str = ','.join(cur_.mogrify(val_arg_template, t).decode('utf-8')
                            for t in tuples_)
        cur_.execute("INSERT INTO {} VALUES ".format(table_) + args_str)

    with open(os.path.join(THIS_DIR, 'schema.db'), 'r') as fd:
        schema = fd.read()
    tables = re.findall(r'CREATE\s+TABLE\s+(\S+)\s*\((.*?)\)\s*;', schema,
                        re.IGNORECASE | re.DOTALL)
    assert len(tables) == 1

    csv_path = os.path.join(path, 'census-income.data')
    rows = _read_csv_and_preprocess(csv_path)

    cur = con.cursor()

    table, schema = tables[0]

    fields = [f.split()[:2] for f in schema.split('\n') if f.strip()]
    with tqdm.tqdm(total=len(rows)) as p_bar:
        tuples = []
        for row in rows:
            p_bar.update(1)
            if len(fields) != len(row):
                print('\n'.join(map(str, zip(row, fields))))
                print('\n'.join((str(fields), str(row))))
                assert False
            tuples.append(row)
            if len(tuples) >= row_chunks:
                execute(cur, tuples, table)
                tuples = []
        if tuples:
            execute(cur, tuples, table)

    cur.close()


def main():
    sys.path = [THIS_DIR, *sys.path]
    import db_settings

    config = vars(db_settings)

    args = argparse.ArgumentParser()
    args.add_argument('path', metavar='PATH', nargs='?',
                      help="The directory where to find the data csv in. "
                           "Defaults to current working directory.",
                      default='.')
    args = args.parse_args()

    start_date = datetime.datetime.now()
    print("Import data from {}".format(args.path))

    db_name = config.get('DB_NAME', 'census')
    db_user = config.get('DB_USER', 'pguser')
    db_pass = config.get('DB_PW', 'pgpassword')
    db_host = config.get('DB_HOST', 'localhost')
    if config.get('DB_CREATE'):
        con = psycopg2.connect(database='postgres', user=db_user,
                               host=db_host, password=db_pass)
        con.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = con.cursor()
        cur.execute(sql.SQL(
            "DROP DATABASE IF EXISTS {0}").format(sql.Identifier(db_name))
        )
        cur.execute(sql.SQL(
            "CREATE DATABASE {0}").format(sql.Identifier(db_name))
        )
        con.close()

    con = psycopg2.connect(database=db_name, user=db_user,
                           host=db_host, password=db_pass)
    con.autocommit = True

    create_table(con)
    fill_table(con, args.path)

    con.close()
    end_date = datetime.datetime.now()
    seconds = (end_date - start_date).total_seconds()
    print("\nExecuted in {}s".format(seconds))


if __name__ == '__main__':
    main()

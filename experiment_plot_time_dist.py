import argparse
import csv
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


def main(args):

    fig, ax = plt.subplots()

    multi = len(args.group) > 1

    for group_i, group_spec in enumerate(args.group):
        g_name, q_num, g_path = map(lambda s: s.strip(), group_spec.split(":"))
        q_num = int(q_num)
        times = None
        paths = glob.glob(os.path.join(g_path, "*", "time_*.csv"))
        for fn_i, fn in enumerate(paths):
            with open(fn, "r") as fd:
                lines = list(csv.reader(fd))

            if times is None:
                if q_num == -2:
                    shape = (len(lines), len(paths))
                else:
                    shape = len(paths)
                times = np.empty(shape, dtype=np.double)
            a = np.array(lines, dtype=np.double)
            if q_num == -2:
                # all times
                times[:, fn_i] = a[:, 1]
            elif q_num == -1:
                # sum of all times per run
                times[fn_i] = a[:, 1].sum()
            else:
                a = a[a[:, 0] == q_num, 1]
                assert a.size or print("Query number %i not available in %s." %
                                       (q_num, fn), file=sys.stderr)
                times[fn_i] = a
        ax.hist(times.reshape(-1), bins=int(times.size / 10), label=g_name,
                density=True, alpha=0.7 if multi else 1)

    ax.set_ylabel("Density of queries")
    ax.set_xlabel("Time in seconds to complete")
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    if multi:
        ax.legend(loc="upper left")

    fig.tight_layout()
    out_path = args.out[0]
    plt.savefig(out_path + ".svg", format='svg')
    plt.savefig(out_path + ".pdf", format='pdf')
    plt.savefig(out_path + ".png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--title", type=str)
    parser.add_argument("out", metavar="PATH", nargs=1)
    parser.add_argument("group", metavar="NAME:QUERY_NUM:PATH", nargs="+")
    main(parser.parse_args())

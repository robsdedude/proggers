import csv
import os
import re

import matplotlib.pyplot as plt
import numpy as np

CSV_ERROR_RE = re.compile(r"^error(?:_\d+)?.csv$")
CSV_TIME_RE = re.compile(r"^time(?:_\d+)?.csv$")

NUM_ERROR_SAMPLES = 100


def plot_normals(xs, ysss, title, names=None, path='plots'):
    os.makedirs(path, exist_ok=True)
    plt.title(title)
    for i in range(len(ysss)):
        mean = np.nanmean(ysss[i], axis=0)
        std_err = np.nanstd(ysss[i], axis=0)
        kwargs = {} if names is None else {"label": names[i]}
        plt.plot(xs, mean, **kwargs)
        plt.fill_between(xs, mean - std_err, mean + std_err, alpha=0.5)
    if names is not None:
        plt.legend()
    plt.xlim(0, 1)
    plt.savefig(
        os.path.join(path, re.sub(r'\s+', '', title.lower()) + '.svg'),
        format='svg'
    )
    plt.savefig(
        os.path.join(path, re.sub(r'\s+', '', title.lower()) + '.pdf'),
        format='pdf'
    )
    plt.savefig(
        os.path.join(path, re.sub(r'\s+', '', title.lower()) + '.png')
    )
    plt.close()


def plot_normal(xs, yss, title, path='plots'):
    return plot_normals(xs, [yss], title, path=path)


def plot_error_group(root, files):
    def to_np_float_array(line_):
        return np.array(
            [float(x) if x else float('nan') for x in line_],
            dtype=np.double
        )

    groups = []
    for exp_i, fn in enumerate(files):
        with open(os.path.join(root, fn), "r") as fd:
            lines = list(csv.reader(fd))
        experiments = exp = times = queries = query_i = None
        for line in lines:
            if line[0] == "time":
                group_i = int(line[1])
                if len(groups) <= group_i:
                    queries = []
                    groups.append(queries)
                else:
                    queries = groups[group_i]

                times = to_np_float_array(line[3:])
            elif line[0] == "estimate":
                assert group_i == int(line[1])
                query_i = int(line[2])
                assert queries == groups[group_i]
                if len(queries) <= query_i:
                    experiments = []
                    queries.append(experiments)
                else:
                    experiments = queries[query_i]
                exp = {"estimate": to_np_float_array(line[3:]), "times": times}
            elif line[0] == "real":
                assert group_i == int(line[1])
                assert query_i == int(line[2])
                assert queries == groups[group_i]
                exp["real"] = to_np_float_array(line[3:])
                exp["diff"] = exp["estimate"] - exp["real"]
                exp["ratio"] = exp["estimate"] / exp["real"]
                experiments.append(exp)
                assert len(groups[group_i][query_i]) == exp_i + 1
            else:
                print(line)
                assert False
    x_inter = np.linspace(0, 1, NUM_ERROR_SAMPLES)
    path = os.path.join(root, "plots")
    for group_i in range(len(groups)):
        for query_i in range(len(groups[group_i])):
            real = np.empty((len(groups[group_i][query_i]), NUM_ERROR_SAMPLES),
                            dtype=np.double)
            est = np.empty_like(real)
            diff = np.empty_like(real)
            ratio = np.empty_like(real)
            for exp_i in range(len(groups[group_i][query_i])):
                exp = groups[group_i][query_i][exp_i]
                x = exp["times"] / exp["times"][-1]
                real[exp_i, :] = np.interp(x_inter, x, exp["real"])
                est[exp_i, :] = np.interp(x_inter, x, exp["estimate"])
                diff[exp_i, :] = np.interp(x_inter, x, exp["diff"])
                ratio[exp_i, :] = np.interp(x_inter, x, exp["ratio"])
            real[np.isinf(real)] = np.nan
            est[np.isinf(est)] = np.nan
            diff[np.isinf(diff)] = np.nan
            ratio[np.isinf(ratio)] = np.nan
            plot_normal(x_inter, real,
                        "Real %i.%i" % (group_i + 1, query_i + 1), path)
            plot_normal(x_inter, est,
                        "Est %i.%i" % (group_i + 1, query_i + 1), path)
            plot_normal(x_inter, diff,
                        "Diff %i.%i" % (group_i + 1, query_i + 1), path)
            plot_normal(x_inter, ratio,
                        "Ratio %i.%i" % (group_i + 1, query_i + 1), path)
            plot_normals(x_inter, [real, est],
                         "Error %i.%i" % (group_i + 1, query_i + 1),
                         ["Real Error", "Estimated Error"], path)


def plot_time_group(root, files):
    array = None
    for i, fn in enumerate(files):
        with open(os.path.join(root, fn), "r") as fd:
            lines = list(csv.reader(fd))
            if array is None:
                array = np.empty((len(lines), len(files)), dtype=np.double)
            array[:, i] = np.array(lines, dtype=np.double)[:, 1]

    path = os.path.join(root, "times_avg.csv")
    mean = array.mean(axis=1)
    std_dev = array.std(axis=1)
    sum_ = array.sum(axis=0)
    n = len(files)
    with open(path, "w") as fd:
        writer = csv.writer(fd)
        assert len(mean) == len(std_dev)
        writer.writerow(["n", n])
        for group_i in range(len(mean)):
            writer.writerow([group_i, "mean", mean[group_i]])
            writer.writerow([group_i, "std_dev", std_dev[group_i]])
        writer.writerow(["all", "mean", array.mean()])
        writer.writerow(["all", "std_dev", array.std()])
        writer.writerow(["sum", "mean", sum_.mean()])
        writer.writerow(["sum", "std_dev", sum_.std()])


def main(args):
    error_groups = []
    time_groups = []

    for root, dirs, files in os.walk(args.path, topdown=True):
        depth = root[len(args.path) + len(os.path.sep):].count(os.path.sep)
        error_group = [f for f in files if CSV_ERROR_RE.match(f)]
        if error_group:
            error_groups.append((root, error_group))
        time_group = [f for f in files if CSV_TIME_RE.match(f)]
        if time_group:
            time_groups.append((root, time_group))
        if depth == args.max_depth:
            del dirs[:]

    for root, group in error_groups:
        plot_error_group(root, group)
    for root, group in time_groups:
        plot_time_group(root, group)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-depth", "-d", type=int, default=-1,
                        help="how deep to traverse the tree (default -1: inf)")
    parser.add_argument("path", nargs="?", type=str, default=".",
                        metavar="PATH", help="where to look")

    main(parser.parse_args())

import argparse
import csv
import glob
import os

import matplotlib.pyplot as plt
import numpy as np


def main(args):
    means = []
    means_all = []
    means_sum = []
    stds = []
    stds_all = []
    stds_sum = []
    names = []
    all_ = False
    sum_ = True
    for group_i, group_spec in enumerate(args.group):
        g_name, g_path = map(lambda s: s.strip(), group_spec.split(":"))
        means.append([])
        stds.append([])
        names.append(g_name)
        glob_path = os.path.join(g_path, "*", "times_avg.csv")
        with open(glob.glob(glob_path)[0], "r") as fd:
            for row in csv.reader(fd):
                if row[0] in ("all", "sum"):
                    if row[1] == "mean":
                        list_ = locals()["means_" + row[0]]
                    elif row[1] == "std_dev":
                        list_ = locals()["stds_" + row[0]]
                    assert len(list_) == group_i
                elif row[1] == "mean":
                    list_ = means[-1]
                    assert len(list_) == int(row[0])
                elif row[1] == "std_dev":
                    list_ = stds[-1]
                    assert len(list_) == int(row[0])
                else:
                    continue
                list_.append(np.array(float(row[2])))
        means[-1] = np.array(means[-1], dtype=float)
        stds[-1] = np.array(stds[-1], dtype=float)

    length = min(*(len(m) for l in (means, stds) for m in l))
    means = [np.concatenate((means[i][:length],
                             [means_all[i]] if all_ else [],
                             [means_sum[i]] if sum_ else []))
             for i in range(len(means))]
    stds = [np.concatenate((stds[i][:length],
                            [stds_all[i]] if all_ else [],
                            [stds_sum[i]] if sum_ else []))
            for i in range(len(means))]
    x_labels = ["Q%i" % (i + 1) for i in range(length)]
    if all_:
        x_labels.append("All")
    if sum_:
        x_labels.append("Sum")
    x = np.arange(len(x_labels))
    width = .7 / (len(means) + 1)  # the width of the bars

    kwargs = {"ncols": 1, "figsize": (6.4*1.5, 4.8)}
    if all_ or sum_:
        kwargs.update(
            ncols=2,
            gridspec_kw={"width_ratios": [length, all_ + sum_]}
        )
    fig, axs = plt.subplots(**kwargs)
    ax1 = axs[0]
    ax2 = None if len(axs) == 1 else axs[1]
    assert bool(ax2) == (all_ or sum_)
    ax1.set_ylabel("Time in seconds to complete")
    ax1.set_xlabel("Query")
    ax1.set_xticks(x[:-1])
    ax1.set_xticklabels(x_labels[:-1])
    if all_ or sum_:
        ax2.set_xticks(x[-1:])
        ax2.set_xticklabels(x_labels[-1:])

    rects1 = [
        ax1.bar((x + (i - ((len(means) - 1) / 2)) * width)[:-1],
                means[i][:-1], width, label=names[i], yerr=stds[i][:-1])
        for i in range(len(means))
    ]
    if all_ or sum_:
        rects2 = [
            ax2.bar((x + (i - ((len(means) - 1) / 2)) * width)[-1:],
                    means[i][-1:], width, label=names[i],
                    yerr=stds[i][-1:])
            for i in range(len(means))
        ]

    ax1.legend(loc="upper left")
    ax1.set_ylim(bottom=0)
    if all_ or sum_:
        ax2.set_ylim(bottom=0)

    fig.tight_layout()
    out_path = args.out[0]
    plt.savefig(out_path + ".svg", format='svg')
    plt.savefig(out_path + ".pdf", format='pdf')
    plt.savefig(out_path + ".png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--title", type=str)
    parser.add_argument("out", metavar="NAME:PATH", nargs=1)
    parser.add_argument("group", metavar="NAME:PATH", nargs="+")
    main(parser.parse_args())

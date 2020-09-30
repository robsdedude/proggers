#!/usr/bin/env bash
set -e

ROOT_PT="${1:-.}"

python3 experiment_plot.py "${ROOT_PT}"
mkdir -p "${ROOT_PT}/out_plots"
python3 ./experiment_plot_time_bars.py "${ROOT_PT}/out_plots/think_time" \
    "no think time:${ROOT_PT}/out_with_tailindex" \
    "1 second:${ROOT_PT}/out_with_1s_think" \
    "10 seconds:${ROOT_PT}/out_with_10s_think" \
    "100 seconds:${ROOT_PT}/out_with_100s_think"
python3 ./experiment_plot_time_bars.py "${ROOT_PT}/out_plots/tail_index_time" \
    "with tail index:${ROOT_PT}/out_with_tailindex" \
    "without tail index:${ROOT_PT}/out_without_tailindex"
python3 ./experiment_plot_time_bars.py "${ROOT_PT}/out_plots/or_support_time" \
    "basic or support:${ROOT_PT}/out_without_or_support" \
    "full or support:${ROOT_PT}/out_with_tailindex"

mkdir -p "${ROOT_PT}/out_plots/time_dist"
for ((i=-2; i<=15; i++)); do
    if [[ ${i} -gt -1 ]]; then
        name="Q$((i + 1))"
    elif [[ ${i} -eq -1 ]]; then
        name="Sum"
    else
        name="All"
    fi
    name_lower=$(echo "${name}" | tr '[:upper:]' '[:lower:]')
    python3 ./experiment_plot_time_dist.py \
        "${ROOT_PT}/out_plots/time_dist/time_dist_${name_lower}" \
        "${name}:${i}:${ROOT_PT}/out_with_tailindex"
done

#!/usr/bin/env bash

run_exp() {
    #run_exp(out_folder, name, args, n=100)
    out_folder="out_results/${1}"
    mkdir "${out_folder}"
    name=$2
    args=$3
    n=${4:-100}
    stdout_fn="${out_folder}/${name}_log.txt"
    stderr_fn="${out_folder}/${name}_error.txt"
    (python experiment.py -o "${out_folder}" ${args} -n "${n}" | tee "${stdout_fn}") 3>&1 1>&2 2>&3 | tee "${stderr_fn}"
}


for seconds in 100 10 1; do
    echo "time w/ ${seconds}s thinking time"
    run_exp out_with_${seconds}s_think time "--think-time ${seconds} --dump-time"
done

echo "error w/ tailindex"
run_exp out_with_tailindex error "--dump-error"
echo "time w/ tailindex"
run_exp out_with_tailindex time "--dump-time"

echo "time w/o or_support"
run_exp out_without_or_support time "--no-or-support --dump-time"

echo "time w/o tailindex"
run_exp out_without_tailindex time "--no-tailindex --dump-time"


echo "======================== Main experiments done! ========================"

for seconds in 100 10 1; do
    echo "error w/ ${seconds}s thinking time"
    run_exp out_with_${seconds}s_think error "--think-time ${seconds} --dump-error"
done

echo "error w/o or_support"
run_exp out_without_or_support error "--no-or-support --dump-error"

echo "error w/o tailindex"
run_exp out_without_tailindex error "--no-tailindex --dump-error"

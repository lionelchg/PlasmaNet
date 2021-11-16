#!/bin/bash

python perf_petsc.py \
    -n 101 201 401 801 2001 4001 5001 5501 \
    -ls_fn petsc/log/cart/36/rtol_1e-3/default/cg_boomeramg/36_procs/ \
    -cn cases/bench_V100/ -o figures/V100_perfs

python perf_petsc.py \
    -n 101 201 401 801 2001 4001 5001 5501 6001 \
    -ls_fn petsc/log/cart/128/rtol_1e-3/default/cg_boomeramg/128_procs/ \
    -cn cases/bench_A100/ -o figures/A100_perfs
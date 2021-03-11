#!/usr/bin/bash

# Generate datasets for "IRSPR" study (cf. ltloss_runs document on "DL POISSON" drive)

python rhs_random.py --case random_3.4_1.50 --npts 200 --nits 10000 --n_res 4  --device kraken --n_procs 128

python rhs_random.py --case random_3.4_1.40 --npts 160 --nits 10000 --n_res 4  --device kraken --n_procs 128

python rhs_random.py --case random_3.4_1.20 --npts 80  --nits 10000 --n_res 4  --device kraken --n_procs 128

python rhs_random.py --case random_3.4_1.10 --npts 40  --nits 10000 --n_res 4  --device kraken --n_procs 128

python rhs_random.py --case random_3.4_1.5  --npts 20  --nits 10000 --n_res 4  --device kraken --n_procs 128



python rhs_random.py --case random_3.6_1.50 --npts 300 --nits 10000 --n_res 6  --device kraken --n_procs 128

python rhs_random.py --case random_3.6_1.40 --npts 240 --nits 10000 --n_res 6  --device kraken --n_procs 128

python rhs_random.py --case random_3.6_1.20 --npts 120 --nits 10000 --n_res 6  --device kraken --n_procs 128

python rhs_random.py --case random_3.6_1.10 --npts 60  --nits 10000 --n_res 6  --device kraken --n_procs 128

python rhs_random.py --case random_3.6_1.5  --npts 30  --nits 10000 --n_res 6  --device kraken --n_procs 128



python rhs_random.py --case random_3.8_1.50 --npts 400 --nits 10000 --n_res 8  --device kraken --n_procs 128

python rhs_random.py --case random_3.8_1.40 --npts 320 --nits 10000 --n_res 8  --device kraken --n_procs 128

python rhs_random.py --case random_3.8_1.20 --npts 160 --nits 10000 --n_res 8  --device kraken --n_procs 128

python rhs_random.py --case random_3.8_1.10 --npts 80  --nits 10000 --n_res 8  --device kraken --n_procs 128

python rhs_random.py --case random_3.8_1.5  --npts 40  --nits 10000 --n_res 8  --device kraken --n_procs 128



python rhs_random.py --case random_3.10_1.50 --npts 500 --nits 10000 --n_res 10 --device kraken --n_procs 128

python rhs_random.py --case random_3.10_1.40 --npts 400 --nits 10000 --n_res 10 --device kraken --n_procs 128

python rhs_random.py --case random_3.10_1.20 --npts 200 --nits 10000 --n_res 10 --device kraken --n_procs 128

python rhs_random.py --case random_3.10_1.10 --npts 100 --nits 10000 --n_res 10 --device kraken --n_procs 128

python rhs_random.py --case random_3.10_1.5  --npts 50  --nits 10000 --n_res 10 --device kraken --n_procs 128



python rhs_random.py --case random_3.12_1.50 --npts 600 --nits 10000 --n_res 12 --device kraken --n_procs 128

python rhs_random.py --case random_3.12_1.40 --npts 480 --nits 10000 --n_res 12 --device kraken --n_procs 128

python rhs_random.py --case random_3.12_1.20 --npts 240 --nits 10000 --n_res 12 --device kraken --n_procs 128

python rhs_random.py --case random_3.12_1.10 --npts 120 --nits 10000 --n_res 12 --device kraken --n_procs 128

python rhs_random.py --case random_3.12_1.5  --npts 60  --nits 10000 --n_res 12 --device kraken --n_procs 128

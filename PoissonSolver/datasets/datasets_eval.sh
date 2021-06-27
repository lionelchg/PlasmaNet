#!/usr/bin/bash

#SBATCH --job-name=Pdata
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=60
#SBATCH --time=4:00:00
#SBATCH --partition=rome

python rhs_random.py -c eval.yml -nr 4 -nn $1
python rhs_random.py -c eval.yml -nr 8 -nn $1
python rhs_random.py -c eval.yml -nr 12 -nn $1
python rhs_random.py -c eval.yml -nr 16 -nn $1

python rhs_fourier.py -c eval.yml -nmin 1 -nmax 1 -dp 0 -nn $1
python rhs_fourier.py -c eval.yml -nmin 1 -nmax 3 -dp 0 -nn $1
python rhs_fourier.py -c eval.yml -nmin 1 -nmax 6 -dp 0 -nn $1

python rhs_fourier.py -c eval.yml -nmin 1 -nmax 3 -dp 2 -nn $1
python rhs_fourier.py -c eval.yml -nmin 1 -nmax 6 -dp 2 -nn $1


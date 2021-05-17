#!/usr/bin/bash
python rhs_random.py -c eval.yml -nr 4 -nn 101
python rhs_random.py -c eval.yml -nr 8 -nn 101
python rhs_random.py -c eval.yml -nr 12 -nn 101
python rhs_random.py -c eval.yml -nr 16 -nn 101

python rhs_fourier.py -c eval.yml -nm 1 -dp 0 -nn 101
python rhs_fourier.py -c eval.yml -nm 3 -dp 0 -nn 101
python rhs_fourier.py -c eval.yml -nm 6 -dp 0 -nn 101

python rhs_fourier.py -c eval.yml -nm 3 -dp 2 -nn 101
python rhs_fourier.py -c eval.yml -nm 6 -dp 2 -nn 101

python rhs_hills.py -c eval.yml -nn 101
#!/usr/bin/bash
# Script to download the dataset from Kraken's scratch
# The ssh hostname `kraken` must be resolvable

if [ ! -d datasets ] ; then
    mkdir datasets
fi
rsync -rzvhp kraken:/scratch/cfd/bogopolsky/DL/poisson_datasets ./datasets

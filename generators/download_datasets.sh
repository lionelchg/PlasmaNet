#!/usr/bin/env bash
# Script to download the dataset from Kraken's scratch
# The ssh hostname `kraken` must be resolvable

# Datasets local directory
DATA_DIR="../datasets"

# Distant subdirs to download
DISTANT_DIRS="50x50 64x64"

# Create DATA_DIR if needed
if [ ! -d $DATA_DIR ] ; then
    mkdir $DATA_DIR
fi

# Download DISTANT_DIRS using rsync
for folder in $DISTANT_DIRS; do
  rsync -rzvhp kraken:/scratch/cfd/bogopolsky/DL/poisson_datasets/$folder $DATA_DIR
done

#!/scratch/cfd/bogopolsky/DL/dl_env/bin/python
"""
Parse and plot performance analysis of PlasmaNet benchmarks

G. Bogopolsky, CERFACS, 28.06.2021
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import yaml
import pdb
from glob import glob
from itertools import product, cycle
import matplotlib as mpl
from matplotlib.lines import Line2D
from numpy.polynomial import Polynomial
from scipy.stats import linregress
from scipy.optimize import curve_fit

from PlasmaNet.common.utils import create_dir

def read_perfs_petsc(base_fn: str, nnxs: list):
    """ Read PETSc performance output file """
    nnodes_list = list()
    best_times = list()
    av_times = list()
    stddev_times = list()

    # Read the elapsed times
    for nnx in nnxs:
        fp = open(f'{base_fn}_{nnx:d}.log', 'r')
        for line in fp:
            if '*------' in line:
                nnodes_list.append(int(fp.readline().strip('\n').split('=')[1]))
                fp.readline()
                best_times.append(float(fp.readline().strip('\n').split('=')[1]))
                av_times.append(float(fp.readline().strip('\n').split('=')[1]))
                stddev_times.append(float(fp.readline().strip('\n').split('=')[1]))
                break
        fp.close()

    nnodes_list = np.array(nnodes_list)
    best_times = np.array(best_times)
    av_times = np.array(av_times)
    stddev_times = np.array(stddev_times)

    return nnodes_list, best_times, av_times, stddev_times

def parse_output(filepath, lookup_table):
    print(f"Reading {filepath}")
    with open(filepath, "r") as fid:
        lines = fid.readlines()

    data = {k: [] for k in lookup_table}
    for line in lines:
        for value_name, (value_string, value_type) in lookup_table.items():
            if value_string in line:
                value = value_type(line.split("=")[-1].strip())
                data[value_name].append(value)

    n_rows = min([len(data[k]) for k in data])
    print(f"Found {n_rows} data rows")
    # If a set has more data than another, cut it
    # All sets must have the same length for pandas
    for key in data:
        if len(data[key]) > n_rows:
            cur_len = len(data[key])
            data[key] = data[key][:n_rows - cur_len]
            print(f"Cutting singleton data for {key} set")

    return pd.DataFrame.from_dict(data)


def linear_model(x, a, b):
    """ Linear model for fits """
    return a + b*x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cases_root", type=str, default=None,
                        help="Cases root directory")
    parser.add_argument("-l", "--linsystem", type=str, default="linsystem.out",
                        help="Linsystem benchmark file")
    parser.add_argument("-o", "--output_name", type=str, default=None,
                        help="Output figure name")
    args = parser.parse_args()

    with open('bench_config.yml') as yaml_stream:
        bench_cfg = yaml.safe_load(yaml_stream)

    with open("network_base_config.yml", 'r') as yaml_stream:
        config = yaml.safe_load(yaml_stream)

    # Parse output files
    perf = pd.DataFrame()
    n_cases = 0

    # Parse network output
    networks = list(bench_cfg["networks"].keys())
    base_casename = config["network"]["casename"]
    if args.cases_root is not None:
        base_casename = base_casename.replace("cases", args.cases_root)
        print(f"Cases root changed to {base_casename}")
    for net, nn in product(bench_cfg["networks"], bench_cfg["sizes"]):
        filepath = os.path.join(base_casename, str(net), str(nn), "info.log")
        lookup_table = {
            f"{net}_model": ["model_timer", float],
            f"{net}_comm" : ["comm_timer", float],
        }
        case_df = parse_output(filepath, lookup_table)
        case = f"case_{n_cases}"
        print(f"{case}: size={nn}  network={net}")
        n_cases += 1
        case_df["case"] = case
        case_df["size"] = nn
        case_df[net] = case_df[f"{net}_model"] + case_df[f"{net}_comm"]
        perf = perf.append(case_df)

    # Print averages and standard deviation
    for case in perf["case"].unique():
        work = perf[perf["case"] == case]
        print("---------------------------------------------")
        print(f"{case} - nnx = {work['size'].unique().item()}")
        n_exp = work.shape[0]
        print(f"Averaging on {n_exp} experiences")
        for net in networks:
            tmp = work[net]
            print(f"Perf: {tmp.mean() * 1000:.4f} ± {tmp.std() * 1000:.4f} ms")
    print("---------------------------------------------")

    # Delete aberrant first data (because std dev is ugly)
    # Drop the first two measurements of each case
    to_del = perf.groupby("case").head(2)
    perf = pd.concat([perf, to_del]).drop_duplicates(keep=False)

    # Execute stats
    perf = perf.groupby("size").agg(["mean", "std"])

    # Sort dataset
    # Dataset can be ordered as [case_1, case_10, case_11, case_2...], so we must reorder it
    # We reindex it with a new list of index built by sorting it on the numerical value
    # perf = perf.reindex(index=perf.index.to_series().str.rsplit("_").str[-1].astype(int).sort_values().index)
    # No need to sort with Int64Index

    print(perf)

    # PETSc performance
    nnxs = [101, 201, 401, 801, 2001, 4001, 5001]
    # nnodes_list, best_times, av_times, stddev_times = read_perfs_petsc('petsc/log/A100/solver_cg_gamg_128_procs', nnxs)
    nnodes_list, best_times, av_times, stddev_times = read_perfs_petsc('petsc/log/V100/solver_cg_gamg_36_procs', nnxs)

    ###########################################
    #   Plots
    ###########################################

    fig, ax = plt.subplots(figsize=(7, 4))

    # Linear solver
    idx = perf.index**2
    ax.plot(nnodes_list, av_times, "-x", label="Linear solver")
    ax.fill_between(nnodes_list, av_times + stddev_times, av_times - stddev_times, alpha=.2)
    # Networks
    for net in networks:
        tot, model, comm = perf[net], perf[net + "_model"], perf[net + "_comm"]
        ax.plot(idx, tot["mean"], "-o", label=net)
        ax.fill_between(idx, tot["mean"] + tot["std"], tot["mean"] - tot["std"],
                        alpha=.2)
        ax.plot(idx, comm["mean"], "-+", label=net + " comm")
        ax.fill_between(idx, comm["mean"] + comm["std"], comm["mean"] - comm["std"], alpha=.2)
        ax.plot(idx, model["mean"], "-^", label=net + " model")
        ax.fill_between(idx, model["mean"] + model["std"], model["mean"] - model["std"], alpha=.2)

    ax.loglog()
    ax.legend()
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel(f"Mean execution time with standard deviation [s]", wrap=True)

    plt.tight_layout()


    # Save fig in figures directory, with an incremented number if a previous figure already exists
    if args.output_name is None:
        create_dir("figures/")
        fig_list = glob("figures/perf_plot_*.png")
        if len(fig_list) > 0:
            i_fig_list = [int(i.split("_")[-1].split(".")[0]) for i in fig_list]
            i_fig = max(i_fig_list)
        else:
            i_fig = 0
        output_name = "figures/perf_plot_{:03d}.png".format(i_fig + 1)
    else:
        output_name = args.output_name
    fig.savefig(output_name, dpi=250)
#!/scratch/cfd/bogopolsky/DL/dl_env/bin/python
"""
Parse and plot performance analysis of PlasmaNet benchmarks

G. Bogopolsky, CERFACS, 28.06.2021
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import yaml
from itertools import product


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


if __name__ == "__main__":

    with open('bench_config.yml') as yaml_stream:
        bench_cfg = yaml.safe_load(yaml_stream)

    with open("network_base_config.yml", 'r') as yaml_stream:
        config = yaml.safe_load(yaml_stream)

    # Parse output files
    perf = pd.DataFrame()
    n_cases = 0
    # Parse linsystem benchmark output
    for nn, solver_type, useUmfpack, assumeSortedIndices in product(
            bench_cfg["sizes"],
            bench_cfg["solver_types"],
            bench_cfg["useUmfpack"],
            bench_cfg["assumeSortedIndices"]
    ):

        filepath = "linsystem.out"
        case_desc = '_'.join([str(tmp) for tmp in [nn, solver_type, useUmfpack, assumeSortedIndices]])
        lookup_table = {"umf": [case_desc + "_solve_time", float]}

        case_df = parse_output(filepath, lookup_table)
        case = f"case_{n_cases}"
        print(f"{case}: size={nn}  solver_type={solver_type}  useUmfpack={useUmfpack}  "
              f"assumeSortedIndices={assumeSortedIndices}")
        n_cases += 1
        case_df["case"] = case
        case_df["size"] = nn
        # case_df["solver_type"] = solver_type
        # case_df["useUmfpack"] = useUmfpack
        # case_df["assumeSortedIndices"] = assumeSortedIndices
        perf = perf.append(case_df)

    # Parse network output
    networks = bench_cfg["networks"].keys()
    base_casename = config["network"]["casename"]
    for net, nn in product(bench_cfg["networks"], bench_cfg["sizes"]):
        filepath = os.path.join(base_casename, str(net), str(nn), "info.log")
        lookup_table = {
            f"{net}_model": ["model_timer", float],
            f"{net}_comm": ["comm_timer", float],
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
        tmp = work["umf"]
        print(f"Perf: {tmp.mean() * 1000:.4f} ± {tmp.std() * 1000:.4f} ms")
        for net in networks:
            tmp = work[net]
            print(f"Perf: {tmp.mean() * 1000:.4f} ± {tmp.std() * 1000:.4f} ms")
    print("---------------------------------------------")

    # Delete aberrant first data (because std dev is ugly)
    # Drop the first two measurements of each case
    to_del = perf.groupby("case").head(2)
    perf = pd.concat([perf, to_del]).drop_duplicates(keep=False)

    # Execute stats
    # perf = perf.groupby("case").agg(["mean", "std"])
    perf = perf.groupby("size").agg(["mean", "std"])

    # Sort dataset
    # Dataset can be ordered as [case_1, case_10, case_11, case_2...], so we must reorder it
    # We reindex it with a new list of index built by sorting it on the numerical value
    # perf = perf.reindex(index=perf.index.to_series().str.rsplit("_").str[-1].astype(int).sort_values().index)
    # No need to sort with Int64Index

    print(perf)

    ###########################################
    #   Plots
    ###########################################

    fig, ax = plt.subplots(figsize=(10, 4), ncols=2)
    ax, ax2 = ax.ravel()

    # Linear solver
    umf = perf["umf"]
    ax.plot(perf.index, umf["mean"], "-x", label="Linear solver")
    ax.fill_between(perf.index, umf["mean"] + umf["std"], umf["mean"] - umf["std"], alpha=.2)
    # Networks
    for net in networks:
        tot, model, comm = perf[net], perf[net + "_model"], perf[net + "_comm"]
        # ax.plot(perf.index, tot["mean"], "-o", label="PlasmaNet")
        # ax.fill_between(perf.index, tot["mean"] + tot["std"], tot["mean"] - tot["std"],
        #                 alpha=.2)
        # ax.plot(perf.index, comm["mean"], "-+", label="CPU <-> GPU comms")
        # ax.fill_between(perf.index, comm["mean"] + comm["std"], comm["mean"] - comm["std"], alpha=.2)
        # ax.plot(perf.index, model["mean"], "-^", label="GPU inference")
        # ax.fill_between(perf.index, model["mean"] + model["std"], model["mean"] - model["std"], alpha=.2)
        ax.plot(perf.index, tot["mean"], "-o", label=net)
        ax.fill_between(perf.index, tot["mean"] + tot["std"], tot["mean"] - tot["std"],
                        alpha=.2)
        ax.plot(perf.index, comm["mean"], "-+", label=net + " comm")
        ax.fill_between(perf.index, comm["mean"] + comm["std"], comm["mean"] - comm["std"], alpha=.2)
        ax.plot(perf.index, model["mean"], "-^", label=net + " model")
        ax.fill_between(perf.index, model["mean"] + model["std"], model["mean"] - model["std"], alpha=.2)

    ax.loglog()
    ax.legend()
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel(f"Mean execution time with standard deviation [s]", wrap=True)

    net = list(networks)[-1]
    umf, tot, model, comm = perf["umf"], perf[net], perf[net + "_model"], perf[net + "_comm"]
    speedup = umf["mean"] / tot["mean"]
    speedup_std = speedup * np.sqrt((umf["std"] / umf["mean"])**2 + (tot["std"] / tot["mean"]**2))
    print(speedup)
    print(speedup_std)
    ax2.plot(perf.index, speedup, "-o")
    ax2.fill_between(perf.index, speedup - speedup_std, speedup + speedup_std)

    ax2.semilogx()
    ax2.set_xlabel("Number of nodes")
    ax2.set_ylabel(f"Speedup of {net} network vs. linear solver")

    plt.tight_layout()
    fig.savefig("perf_plot.png", dpi=250)

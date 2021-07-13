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
from glob import glob
from itertools import product, cycle
import matplotlib as mpl
from matplotlib.lines import Line2D
from numpy.polynomial import Polynomial
from scipy.stats import linregress
from scipy.optimize import curve_fit

from PlasmaNet.common.utils import create_dir


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
    parser.add_argument("--paper", action="store_true", help="Plots for paper")
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
    # Parse linsystem benchmark output
    for nn, solver_type, useUmfpack, assumeSortedIndices in product(
            bench_cfg["sizes"],
            bench_cfg["solver_types"],
            bench_cfg["useUmfpack"],
            bench_cfg["assumeSortedIndices"]
    ):
        filepath = args.linsystem
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

    if not args.paper:
        fig, ax = plt.subplots(figsize=(10, 4), ncols=2)
        ax, ax2 = ax.ravel()

        # Linear solver
        umf = perf["umf"]
        idx = perf.index**2
        ax.plot(idx, umf["mean"], "-x", label="Linear solver")
        ax.fill_between(idx, umf["mean"] + umf["std"], umf["mean"] - umf["std"], alpha=.2)
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

        # net = networks[-1]
        for net in networks:
            umf, tot, model, comm = perf["umf"], perf[net], perf[net + "_model"], perf[net + "_comm"]
            speedup = umf["mean"] / tot["mean"]
            speedup_std = speedup * np.sqrt((umf["std"] / umf["mean"])**2 + (tot["std"] / tot["mean"])**2)
            print(pd.DataFrame({"mean": speedup, "std": speedup_std}))
            ax2.plot(idx, speedup, "-o", label=net)
            ax2.fill_between(idx, speedup - speedup_std, speedup + speedup_std, alpha=.2)

        ax2.semilogx()
        ax2.set_xlabel("Number of nodes")
        ax2.set_ylabel(f"Network speedup vs. linear solver")
        ax2.legend()

        plt.tight_layout()

    else:
        mpl.rcParams['text.usetex'] = True
        mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath} \usepackage{physics}"
        mpl.rcParams['font.size'] = 7
        fig, ax = plt.subplots(figsize=(6.3, 2.95), ncols=2)
        ax, ax2 = ax.ravel()

        umf = perf["umf"]
        idx = perf.index**2
        legend = []
        markers = ["d", "o", "P", "*", "p"]
        ax.plot(idx, umf["mean"], "-+", color="black", label="Linear solver")
        # ax.fill_between(idx, umf["mean"] + umf["std"], umf["mean"] - umf["std"], alpha=.2, color="black")
        # Fit
        fit0, pcov_fit0 = curve_fit(linear_model, np.log(idx), np.log(umf["mean"]),
                                    sigma=np.log(umf["std"]))
        # ax.plot(idx, np.exp(linear_model(np.log(idx), *fit0)), "s-", color="blue")

        legend.append(Line2D([0], [0], marker="+", color="black", label="Linear solver"))
        legend.extend([
            # Line2D([0], [0], lw=2.5, color="C0", label="PlasmaNet solver"),
            # Line2D([0], [0], lw=2.5, color="C2", label="GPU inference"),
            # Line2D([0], [0], lw=2.5, color="C1", label=r"CPU $\leftrightarrow$ GPU comms"),
            Line2D([0], [0], lw=1.5, color="darkblue", label="PlasmaNet solver"),
            Line2D([0], [0], lw=1.5, color="darkblue", linestyle="--", label="GPU inference"),
            Line2D([0], [0], lw=1.5, color="darkblue", linestyle=":", label=r"CPU $\leftrightarrow$ GPU comms"),
        ])
        # Networks
        work_networks = networks[0:1]  # Select single element by slicing to prevent iteration on the string itself
        for i, net in enumerate(work_networks):
            tot, model, comm = perf[net], perf[net + "_model"], perf[net + "_comm"]
            color = "darkblue"
            ax.plot(idx, tot["mean"], "d-", markersize=4, color=color)
            # ax.fill_between(idx, tot["mean"] + tot["std"], tot["mean"] - tot["std"],
            #                 alpha=.2, lw=0)
            ax.plot(idx, model["mean"], "d--", markersize=4, color=color)
            # ax.fill_between(idx, model["mean"] + model["std"], model["mean"] - model["std"],
            #                 alpha=.2, lw=0)
            ax.plot(idx, comm["mean"], "d:", markersize=4, color=color)
            # ax.fill_between(idx, comm["mean"] + comm["std"], comm["mean"] - comm["std"],
            #                 alpha=.2, lw=0)

            # Fit for the two regimes
            fit1, pcov_fit1 = curve_fit(linear_model, np.log(idx[:5]), np.log(tot["mean"][:5]), sigma=np.log(tot["std"][:5]))
            fit2, pcov_fit2 = curve_fit(linear_model, np.log(idx[6:]), np.log(tot["mean"][6:]), sigma=np.log(tot["std"][6:]))
            # ax.plot(idx[:5], np.exp(linear_model(np.log(idx[:5]), *fit1)), color="red", marker="s")
            # ax.plot(idx[6:], np.exp(linear_model(np.log(idx[6:]), *fit2)), color="green", marker="s")
            slope_cpu = fit0[1]
            slope_reg1, slope_reg2 = fit1[1], fit2[1]
            # slope_reg1, slope_reg2 = 0.0, 1.168
            # Display slopes
            ax.text(0.6, 0.6, "$\\text{{slope}} = {:.1f}$".format(slope_cpu), transform=ax.transAxes,
                    color="k", rotation=35)
            ax.text(0.2, 0.3, "$\\text{{slope}} = {:.2f}$".format(slope_reg1), transform=ax.transAxes, color="k")
            ax.text(0.6, 0.38, "$\\text{{slope}} = {:.2f}$".format(slope_reg2), transform=ax.transAxes,
                    color="k", rotation=26)

            net_label = net.replace("_", r"\_")
            legend.append(Line2D([0], [0], marker="d", color="w", markerfacecolor="darkblue", markersize=7, label=net_label))
            ax.set_prop_cycle(None)

        ax.loglog()
        ax.legend(handles=legend, frameon=False)
        ax.set_xlabel("Number of nodes")
        ax.set_ylabel(f"Execution time [s]", wrap=True)
        ax.tick_params(which="both", direction="in", top=True, right=True)

        # net = networks[-1]
        colors = cycle(["darkblue", "royalblue"])
        work_networks = networks[:2]
        for i, net in enumerate(work_networks):
            umf, tot, model, comm = perf["umf"], perf[net], perf[net + "_model"], perf[net + "_comm"]
            speedup = umf["mean"] / tot["mean"]
            speedup_std = speedup * np.sqrt((umf["std"] / umf["mean"])**2 + (tot["std"] / tot["mean"])**2)
            print(pd.DataFrame({"mean": speedup, "std": speedup_std}))
            net_label = net.replace("_", r"\_")
            ax2.plot(idx, speedup, marker=markers[i], markersize=4, label=net_label, color=next(colors))
            # ax2.fill_between(idx, speedup - speedup_std, speedup + speedup_std, alpha=.2, lw=0)
            # ax2.plot(idx, speedup, label=net_label)
            # ax2.fill_between(idx, speedup - speedup_std, speedup + speedup_std, alpha=.2, lw=0)

        ax2.semilogx()
        ax2.set_xlabel("Number of nodes")
        ax2.set_ylabel(f"Speedup")
        ax2.legend(frameon=False)
        ax2.tick_params(which="both", direction="in", top=True, right=True)

        index_title = cycle(('(a)', '(b)', '(c)', '(d)', '(e)', '(f)'))
        ax.text(0.9, 0.06, f"$\\bf{next(index_title)}$", transform=ax.transAxes)
        ax2.text(0.9, 0.06, f"$\\bf{next(index_title)}$", transform=ax2.transAxes)

        plt.tight_layout()

        # Write timings to csv for fit in notebook
        perf["UNet5-100k"].to_csv("gpu_timings.csv")

        # Print fits at the end
        print("Linear system fit: slope = {:.2f} with relative error {:.2e}".format(
            fit0[1], (np.sqrt(np.diag(pcov_fit0))/fit0)[1]))
        print("Network regime 1 fit: slope = {:.2f} with relative error {:.2e}".format(fit1[1], (np.sqrt(np.diag(pcov_fit1))/fit1)[1]))
        print("Network regime 2 fit: slope = {:.2f} with relative error {:.2e}".format(fit2[1], (np.sqrt(np.diag(pcov_fit2))/fit2)[1]))

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

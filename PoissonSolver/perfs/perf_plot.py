#!/scratch/cfd/bogopolsky/DL/dl_env/bin/python
"""
Parse and plot performance analysis of PlasmaNet runs with prints
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--cases_root", type=str, help="Cases root directory")
args = parser.parse_args()

params = {
    "case_1": 51,
    "case_2": 101,
    "case_3": 121,
    "case_4": 201,
    "case_5": 301,
    "case_6": 401,
    "case_7": 601,
    "case_8": 801,
    "case_9": 1001,
    "case_10": 1501,
    "case_11": 2001,
}

lookup_table = {
    "dl": ["Poisson DL perf", float],
    "ana": ["Poisson ana perf", float],
    "comm": ["CPU/GPU comm", float],
    "umf": ["Poisson umf perf:", float],
}

# cases_root = "./runs/IRSPR/1_periods/FlexiNet/lt_F_in_T/random_3.10_1.10"
# cases_root = "./runs/1_period/random_3.10_1.10"
# cases_root = "./runs/perf_conda"
cases_root = args.cases_root

def parse_output(filepath):
    print(f"Reading {filepath}")
    with open(filepath, "r") as fid:
        lines = fid.readlines()

    data = { k: [] for k in lookup_table }
    for line in lines:
        for value_name, (value_string, value_type) in lookup_table.items():
            if value_string in line:
                value = value_type(line.split(":")[-1].strip())
                data[value_name].append(value)

    n_rows = min([ len(data[k]) for k in data ])
    print(f"Found {n_rows} data rows")
    # If a set has more data than another, cut it
    # All sets must have the same length for pandas
    for key in data:
        if len(data[key]) > n_rows:
            cur_len = len(data[key])
            data[key] = data[key][:n_rows - cur_len]
            print(f"Cutting singleton data for {key} set")

    return pd.DataFrame.from_dict(data)


# Parse output files
perf = pd.DataFrame()
for case in params.keys():
    filepath = f"{cases_root}/{case}/info.log"

    case_df = parse_output(filepath)
    case_df["case"] = case
    perf = perf.append(case_df)

# Print averages and standard deviation
for case in params.keys():
    print("---------------------------------------------")
    print(f"{case} - nnx = {params[case]}")
    n_exp = perf[perf['case'] == case].shape[0]
    print(f"Averaging on {n_exp} experiences")
    for val in lookup_table:
        tmp = perf[perf["case"] == case][val]
        print(f"{val} perf: {tmp.mean() * 1000:.4f} ± {tmp.std() * 1000:.4f} ms")
    speedup = perf[perf["case"] == case]["umf"].mean() / perf[perf["case"] == case]["dl"].mean()
    print(f"umf/dl speedup: {speedup:.2f}")
print("---------------------------------------------")
print(perf[perf["case"] == "case_1"])

# Delete aberrant first data (because std dev is ugly)
perf = perf.drop([0])

# Execute stats
perf = perf.groupby("case").agg(["mean", "std"])
# Sort dataset
# Dataset can be ordered as [case_1, case_10, case_11, case_2...], so we must reorder it
# We reindex it with a new list of index built by sorting it on the numerical value
perf = perf.reindex(index=perf.index.to_series().str.rsplit("_").str[-1].astype(int).sort_values().index)

print(perf)

# Plots
# names = [f"{size}²" for size in params.values()]
names = np.array(list(params.values())) ** 2
print(names)

fig, ax = plt.subplots(figsize=(10, 4), ncols=2)
ax, ax2 = ax.ravel()

ax.plot(names, perf["dl"]["mean"], "-o", label="PlasmaNet")
ax.fill_between(names, perf["dl"]["mean"] + perf["dl"]["std"], perf["dl"]["mean"] - perf["dl"]["std"], alpha=.2)
# ax.plot(names, perf["ana"]["mean"], "-x", label="SuperLU GSSV")
# ax.fill_between(names, perf["ana"]["mean"] + perf["ana"]["std"], perf["ana"]["mean"] - perf["ana"]["std"], alpha=.2)
ax.plot(names, perf["umf"]["mean"], "-x", label="Linear solver")
ax.fill_between(names, perf["umf"]["mean"] + perf["umf"]["std"], perf["umf"]["mean"] - perf["umf"]["std"], alpha=.2)
ax.plot(names, perf["comm"]["mean"], "-+", label="CPU <-> GPU comms")
ax.fill_between(names, perf["comm"]["mean"] + perf["comm"]["std"], perf["comm"]["mean"] - perf["comm"]["std"], alpha=.2)
ax.plot(names, perf["dl"]["mean"] - perf["comm"]["mean"], "-^", label="GPU inference")

# ax.errorbar(names, perf["dl"]["mean"], perf["dl"]["std"], fmt="-o", label="PlasmaNet")
# ax.errorbar(names, perf["ana"]["mean"], perf["ana"]["std"], fmt="-x", label="Linear solver - SuperLU")
# ax.errorbar(names, perf["comm"]["mean"], perf["comm"]["std"], fmt="-+", label="CPU/GPU comm")

# ax.set_yscale("log")
ax.loglog()
ax.legend()
ax.set_xlabel("Number of nodes")
ax.set_ylabel(f"Mean execution time with standard deviation [s]", wrap=True)

speedup = perf["umf"]["mean"] / perf["dl"]["mean"]
speedup_std = speedup * np.sqrt((perf["umf"]["std"] / perf["umf"]["mean"])**2 + (perf["dl"]["std"] / perf["dl"]["mean"]**2))
print(speedup)
print(speedup_std)
ax2.plot(names, perf["umf"]["mean"] / perf["dl"]["mean"], "-o")

#ax2.loglog()
ax2.semilogx()
ax2.set_xlabel("Number of nodes")
ax2.set_ylabel("Speedup")


plt.tight_layout()
fig.savefig("perf_plot.png", dpi=250)


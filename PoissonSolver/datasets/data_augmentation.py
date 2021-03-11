#!/scratch/cfd/bogopolsky/DL/dl_env/bin/python3
########################################################################################################################
#                                                                                                                      #
#                                            Data augmentation on a dataset                                            #
#                                                                                                                      #
#                                      Guillaume Bogopolsky, CERFACS, 30.09.2020                                       #
#                                                                                                                      #
########################################################################################################################
#
# Operate a set of data augmentations on a dataset and output it as a new one

import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="Operate a set of data augmentations on "
                                             "a dataset and output it as a new one")
parser.add_argument("target_name", type=str, help="Name of the target dataset")
parser.add_argument("source_path", type=Path, help="Path of the source dataset")
parser.add_argument("--operations", type=str, nargs="+", help="Data augmentation operations list")
parser.add_argument("--plot", action="store_true", help="Perform some plots")
parser.add_argument("--plot_period", type=int, default=50, help="Plot period (default: 50)")
args = parser.parse_args()

import numpy as np
from tqdm import tqdm

# Load source dataset
potential = np.load(args.source_path / "potential.npy")
physical_rhs = np.load(args.source_path / "physical_rhs.npy")
nx, ny = potential.shape[1:]
print(f"Initial shape: {potential.shape}")


# Define some operations
def reverse_y_axis(potential, physical_rhs):
    """ Reverse the y axis of the dataset. """
    return np.flip(potential, axis=1), np.flip(physical_rhs, axis=1)


operations_register = {
    "reverse_y_axis": reverse_y_axis
}

# Execute the requested operations
for operation in args.operations:
    new_pot, new_rhs = operations_register[operation](potential, physical_rhs)
    potential = np.append(potential, new_pot, axis=0)
    physical_rhs = np.append(physical_rhs, new_rhs, axis=0)
print(f"Final shape: {potential.shape}")

# Some plots
if args.plot:
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use("agg")
    x = np.arange(nx)
    y = np.arange(ny)
    x, y = np.meshgrid(x, y, indexing="xy")
    
    for i in tqdm(range(0, potential.shape[0], args.plot_period)):
        fig, (ax1, ax2) = plt.subplots(figsize=(10, 4), nrows=1, ncols=2)
        p1 = ax1.contourf(x, y, physical_rhs[i], levels=50)
        plt.colorbar(p1, ax=ax1)
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_title("physical_rhs")
        p2 = ax2.contourf(x, y, potential[i], levels=50)
        plt.colorbar(p2, ax=ax2)
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_title("potential")
        target = Path("./figures/{}/".format(args.target_name))
        target.mkdir(parents=True, exist_ok=True)
        plt.savefig(target / "data_augmentation_{:05d}.png".format(i), dpi=150)
        plt.close(fig)

# Save the new dataset
target_path = Path("./rhs/{}x{}/{}/".format(nx, ny, args.target_name))
target_path.mkdir(parents=True, exist_ok=True)
np.save(target_path / "potential.npy", potential, allow_pickle=False)
np.save(target_path / "physical_rhs.npy", physical_rhs, allow_pickle=False)

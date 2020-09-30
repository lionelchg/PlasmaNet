#!/scratch/cfd/bogopolsky/DL/dl_env/bin/python3
########################################################################################################################
#                                                                                                                      #
#                                    Import and interpolate H5 solutions from AVIP                                     #
#                                                                                                                      #
#                                      Guillaume Bogopolsky, CERFACS, 29.09.2020                                       #
#                                                                                                                      #
########################################################################################################################
# 
# Load an AVIP case and extract the rhs and potential from the solution, interpolates on a regular cartesian mesh 
# and write the dataset as .npy files
# cf. script example from Willca: 
# /scratch/cfd/bogopolsky/PIC/02_Boeuf/00_WV-Verification/RUN/Willca_recent_plots/moyenne_1D_vs_r.py

import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="Load an AVIP case and extract the rhs and potential from the solution," 
                                        "interpolates on a regular cartesian mesh and write the dataset as .npy files")
parser.add_argument("case_name", type=str, help="Name of the current case")
parser.add_argument("case_path", type=Path, help="Path to the root of the AVIP case")
parser.add_argument("nx", type=int, help="Target x dimension")
parser.add_argument("ny", type=int, help="Target y dimension")
parser.add_argument("--plot", action="store_true", help="Activate plot")
parser.add_argument("--plot_period", type=int, default=50, help="Plot period (default: 50)")
args = parser.parse_args()

import numpy as np
import antares as asc
from tqdm import tqdm
asc.set_progress_bar(True)


# Read mesh and solution file
r = asc.Reader("hdf_avbp")
r["filename"] = args.case_path.joinpath("MESH", "mesh.mesh.h5").as_posix()
r["shared"] = True
mesh, base_all = r.read(), r.read()

r = asc.Reader("hdf_avbp")
r["filename"] = args.case_path.joinpath("RUN", "SOLUT", "ic1.sol_0<instant>.h5").as_posix()
r["base"] = base_all
base_all = r.read()

# Extract needed variables
variables = ["x", "y", "potential", "rhs_poisson", "VD_volume"]
base = base_all[:, :, variables]
del base_all

# Compute physical_rhs
print("Compute physical_rhs... ", end="", flush=True)
base[0].set_formula("physical_rhs = rhs_poisson / VD_volume")
base[0].compute("physical_rhs")
print("Done")

# Plots before interpolation
if args.plot:
    def plot_rhs_pot_tri(base, instant, target):
        """ Plot base element on triangular unstructured mesh. """
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.tri import Triangulation
        matplotlib.use("agg")

        fig, (ax1, ax2) = plt.subplots(figsize=(10, 4), nrows=1, ncols=2)
        triangles = Triangulation(base[0].shared["x"], base[0].shared["y"])

        p1 = ax1.tricontourf(triangles, base[0][instant]["physical_rhs"], levels=50)
        plt.colorbar(p1, ax=ax1)
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_title("physical_rhs")

        p2 = ax2.tricontourf(triangles, base[0][instant]["potential"], levels=50)
        plt.colorbar(p2, ax=ax2)
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_title("potential")

        plt.savefig(target, dpi=150)
        plt.close(fig)
    
    nb_inst = len(base[0].keys())
    for i in tqdm(range(0, nb_inst, args.plot_period)):
        instant = base[0].keys()[i]
        target = Path("./figures/{}/".format(args.case_name))
        target.mkdir(parents=True, exist_ok=True)
        plot_rhs_pot_tri(base, instant, target / "pre_interp_{}.png".format(instant))


# Create interpolation base
s = mesh.stats
xmin, xmax = s["min"].shared["x"], s["max"].shared["x"]
ymin, ymax = s["min"].shared["y"], s["max"].shared["y"]
x = np.linspace(xmin, xmax, args.nx)
y = np.linspace(ymin, ymax, args.ny)
x, y = np.meshgrid(x, y, indexing="ij")  # creating a regular structured cartesian mesh
interp_base = asc.Base()
interp_base["0000"] = asc.Zone()
instant_names = base[0].keys()  # create the same instants as in the source base
for instant in instant_names:
    interp_base[0][instant] = asc.Instant()
interp_base[0].shared["x"] = x  # add new mesh info
interp_base[0].shared["y"] = y

# Interpolation
t = asc.Treatment("interpolation")
t["coordinates"] = ["x", "y"]
t["source"] = base
t["target"] = interp_base
interp_base = t.execute()

# Plots after interpolation
if args.plot:
    def plot_rhs_pot_cart(base, instant, target):
        """ Plot base element on cartesian structured mesh. """
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.use("agg")

        fig, (ax1, ax2) = plt.subplots(figsize=(10, 4), nrows=1, ncols=2)

        p1 = ax1.contourf(base[0].shared["x"], base[0].shared["y"], base[0][instant]["physical_rhs"], levels=50)
        plt.colorbar(p1, ax=ax1)
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_title("physical_rhs")

        p2 = ax2.contourf(base[0].shared["x"], base[0].shared["y"], base[0][instant]["potential"], levels=50)
        plt.colorbar(p2, ax=ax2)
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_title("potential")

        plt.savefig(target, dpi=150)
        plt.close(fig)
    
    nb_inst = len(interp_base[0].keys())
    for i in tqdm(range(0, nb_inst, args.plot_period)):
        instant = interp_base[0].keys()[i]
        target = Path("./figures/{}/".format(args.case_name))
        target.mkdir(parents=True, exist_ok=True)
        plot_rhs_pot_cart(interp_base, instant, target / "post_interp_{}.png".format(instant))

# Save base as .npy
nb_inst = len(interp_base[0].keys())
potential = np.zeros((nb_inst, *x.shape), dtype=np.float64)
rhs = np.zeros((nb_inst, *x.shape), dtype=np.float64)
for i in tqdm(range(nb_inst)):
    potential[i] = interp_base[0][i]["potential"]
    rhs[i] = interp_base[0][i]["physical_rhs"]

target_path = Path("./rhs/{}x{}/{}/".format(args.nx, args.ny, args.case_name))
target_path.mkdir(parents=True, exist_ok=True)
np.save(target_path / "potential.npy", potential, allow_pickle=False)
np.save(target_path / "physical_rhs.npy", rhs, allow_pickle=False)

########################################################################################################################
#                                                                                                                      #
#                                        Compute the receptive field empirically                                       #
#                                                                                                                      #
#                              Ekhi Ajuria, Lionel Cheng, CERFACS, 24.09.2021 (mod. Victor Xing 01.09.21)              #
#                                                                                                                      #
########################################################################################################################


import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

import PlasmaNet.nnet.model as module_arch
from PlasmaNet.nnet.parse_config import ConfigParser
from PlasmaNet.common.plot import plot_ax_scalar

def plot_test(output, model, in_res, folder: Path, network, location):
    """ Plot the propagated gradients or values to deduce receptive fields """

    # Create folder if does not exist
    folder.mkdir(parents=True, exist_ok=True)

    # Create grids for plotting
    xx = np.arange(0, in_res)
    yy = np.arange(0, in_res)
    Xb, Yb = np.meshgrid(xx, yy)

    # Define receptive field variables
    rf_global_x = model.rf_global_x
    rf_global_y = model.rf_global_y

    # Define bounding box of expected receptive field
    if "center" in location:
        # For a center point the receptive field from the kernel sizes goes both ways
        x_rf = [in_res // 2 - rf_global_x // 2, in_res // 2 - rf_global_x // 2,
            in_res // 2 + rf_global_x // 2, in_res // 2 + rf_global_x // 2,
            in_res // 2 - rf_global_x // 2]
        y_rf = [in_res // 2 - rf_global_y // 2, in_res // 2 + rf_global_y // 2,
            in_res // 2 + rf_global_y // 2, in_res // 2 - rf_global_y // 2,
            in_res // 2 - rf_global_y // 2]
    elif "boundary" in location:
        # For a boundary point it only goes one way so that the receptive
        # field is two times lower
        x_rf = [0, 0, rf_global_x // 2, rf_global_x // 2, 0]
        y_rf = [0, rf_global_y // 2, rf_global_y // 2, 0, 0]

    # Image initialization
    fig, ax = plt.subplots(figsize=(5, 5))

    # Plot input, output and expected box
    plot_ax_scalar(fig, ax, Xb, Yb, output[0, 0, :, :].numpy(), r"RF", contour=False)
    ax.plot(x_rf, y_rf, linestyle="dashed", color="k", lw=2)

    # Cut the domain so that only the middle "interesting" part remains
    if "center" in location:
        ax.set_ylim(in_res // 2 - 0.6 * rf_global_y, in_res // 2 + 0.6 * rf_global_y)
        ax.set_xlim(in_res // 2 - 0.6 * rf_global_x, in_res // 2 + 0.6 * rf_global_x)
    elif "boundary" in location:
        ax.set_ylim(-0.1 * rf_global_y, 0.6 * rf_global_y)
        ax.set_xlim(-0.1 * rf_global_x, 0.6 * rf_global_x)

    plt.tight_layout()
    plt.savefig(
        os.path.join(folder, "fig_{}_rfx_{}_rfy_{}_k_{}.png".format(
                network, rf_global_x, rf_global_y, model.kernel_sizes[0]
            )),
        dpi=150,
    )
    plt.close()


def inverse_method(model, cfg_dict, network, center, erf_thres=0.045):
    """ Inverse RF calculating method where a value at the input map is propagated down
    the network.

    - model: network containing the weights initialized to a constant value and biases to 0
    - cfg_dict: dictionary loaded from the cfg.yml file
    - network: string containing the name of the studied network
    - center: boolean to study the center or the BC case

    Effective receptive field threshold: 2 standard deviations from value at the signal
    location (cf Luo et al. NIPS 2016) """
    # Generate input tensor (all zeros except the middle point)
    in_res = cfg_dict["arch"]["args"]["input_res"]
    data = torch.zeros((1, 1, in_res, in_res))
    if center:
        data[:, :, in_res // 2, in_res // 2] = 1
        location = "center"
    else:
        data[:, :, 0, 0] = 1
        location = "boundary_0_0"

    # Evaluate the network and follow metrics
    output = model(data).detach()

    # Plot
    fig_dir = Path(cfg_dict["name"]) / 'inverse' / location
    plot_test(output, model, in_res, fig_dir, network, location)

    if center:
        thres = erf_thres * output[0, 0, in_res // 2, in_res // 2].numpy()
    else:
        thres = erf_thres * output[0, 0, 0, 0].numpy()

    erf_output = torch.where(output > thres, 1.0, 0.0)
    rf_output = torch.where(output > 0, 1.0, 0.0)

    return int(torch.sum(rf_output) ** 0.5), int(torch.sum(erf_output) ** 0.5)

def direct_method(model, cfg_dict, network, center, erf_thres=0.045):
    """ Direct RF calculating method issued from:
        https://github.com/rogertrullo/Receptive-Field-in-Pytorch/blob/master/Receptive_Field.ipynb

    - model: network containing the weights initialized to a constant value and biases to 0
    - cfg_dict: dictionary loaded from the cfg.yml file
    - network: string containing the name of the studied network
    - center: boolean to study the center or the BC case

    Effective receptive field threshold: 2 standard deviations from value at the signal
    location (cf Luo et al. NIPS 2016) """

    # Generate Input data
    in_res = cfg_dict["arch"]["args"]["input_res"]
    img_ = torch.ones((1, 1, in_res, in_res)).clone().detach().requires_grad_(True)

    # Evaluatre network
    out_cnn = model(img_.clone())

    # Generate gradient tensor which will be 0 everywhere except in one middle or BC point
    grad = torch.zeros(out_cnn.size())
    if center:
        grad[:, :, in_res // 2, in_res // 2] = 0.1
        location = "center"
    else:
        grad[:, :, 0, 0] = 0.1
        location = "boundary_0_0"

    # Get the gradient only of the middle point!
    # Compute Receptive field
    out_cnn.backward(gradient=grad)
    grad_torch = img_.grad.detach()

    # Plot
    fig_dir = Path(cfg_dict["name"]) / 'direct' / location
    plot_test(grad_torch, model, in_res, fig_dir, network, location)

    if center:
        thres = erf_thres * grad_torch[0, 0, in_res // 2, in_res // 2].numpy()
    else:
        thres = erf_thres * grad_torch[0, 0, 0, 0].numpy()

    erf_grad_torch = torch.where(grad_torch > thres, 1.0, 0.0)
    rf_grad_torch = torch.where(grad_torch > 0, 1.0, 0.0)

    return int(torch.sum(rf_grad_torch) ** 0.5), int(torch.sum(erf_grad_torch) ** 0.5)

def weights_init_constant(m):
    """ Weight initialization enforced for receptive field study. """
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.constant_(m.weight, 0.1)
        # torch.nn.init.ones_(m.weight)
        torch.nn.init.zeros_(m.bias)

def compute_RF_2D():
    """ Compute receptive fields for given networks and input resolutions
    from inverse and direct methods for center and boundary points. """

    # Work with float64 to avoid overflow
    torch.set_default_dtype(torch.float64)

    # Parse cli argument
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    args = parser.parse_args()

    # Open configuration file
    with open(args.config, "r") as yaml_stream:
        cfg_dict = yaml.safe_load(yaml_stream)

    # Loop on options
    for file in cfg_dict["files"]:
        print("")
        print("-----------------------------------------------------------")
        print("")
        print("Entering file: {}".format(file))
        print("")
        for network in cfg_dict["networks"]:

            print("")
            print("Load Network: {}".format(network))
            print("")

            # Model initialization
            cfg_dict["arch"]["db_file"] = file
            cfg_dict["arch"]["name"] = network

            # Architecture parsing in database
            if "db_file" in cfg_dict["arch"]:
                with open(
                    Path(os.getenv("ARCHS_DIR")) / cfg_dict["arch"]["db_file"]) as yaml_stream:
                    archs = yaml.safe_load(yaml_stream)

                if network in archs:
                    tmp_cfg_arch = archs[cfg_dict["arch"]["name"]]
                else:
                    print("Network {} does not exist on file {} ===> Skipping".format(network, file))
                    break

                if "args" in cfg_dict["arch"]:
                    tmp_cfg_arch["args"] = {
                        **cfg_dict["arch"]["args"],
                        **tmp_cfg_arch["args"],
                    }
                cfg_dict["arch"] = tmp_cfg_arch

            # Creation of config object
            config = ConfigParser(cfg_dict, False)

            # Build model architecture and initialize its weights with;
            # 0 in biases and 1 in weights
            model = config.init_obj("arch", module_arch)

            # Apply the model with the defined weights
            model.apply(weights_init_constant)

            # Compute RF with direct and inverse methods for the center points
            dir_RF, dir_ERF = direct_method(model, cfg_dict, network, True)
            inv_RF, inv_ERF = inverse_method(model, cfg_dict, network, True)

            print("===================================")
            print("Originally Calculated RF: {}".format(model.rf_global_x))
            print("Direct Method RF: {}".format(dir_RF))
            print("Direct Method effective RF: {}".format(dir_ERF))
            print("Inverse Procedure RF: {}".format(inv_RF))
            print("Inverse Procedure effective RF: {}".format(inv_ERF))
            print("===================================")

            # Compute RF with direct and inverse methods for the boundary conditions points
            dir_RF, dir_ERF = direct_method(model, cfg_dict, network, False)
            inv_RF, inv_ERF = inverse_method(model, cfg_dict, network, False)

            print("Originally Calculated RF: {}".format(model.rf_global_x // 2))
            print("Direct Method RF: {}".format(dir_RF))
            print("Direct Method effective RF: {}".format(dir_ERF))
            print("Inverse Procedure RF: {}".format(inv_RF))
            print("Inverse Procedure effective RF: {}".format(inv_ERF))
            print("===================================")

if __name__ == "__main__":
    compute_RF_2D()
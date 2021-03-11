########################################################################################################################
#                                                                                                                      #
#                                       Interpolate a random_rhs dataset                                               #
#                                                                                                                      #
#                                       Ekhi Ajuria, CERFACS, 26.05.2020                                               #
#                                                                                                                      #
########################################################################################################################

"""
Takes an existing dataset and interpolates the fields into the sizes (1/2, 1/4, 1/8, 1/16 and 1/32).
This new tensor is useful for trainings where the output of an scale is used during training.

Input:
Using the parser --dataset option (str): Input the path of the dataset to be interpolated

OUTPUTS:
3 Tensors (creates a new folder called: args.dataset.name + '_Small_interpolated_scales'):
- potential_interp.npy [n_case, 6, nny, nnx] (where the 2 dimension concatenates pot, pot_1/2, pot_1/4,
                pot_1/8, pot_1/16, pot_1/32))
- potential.npy [n_case, nny, nnx] Original potential field
- physical_rhs.npy [n_case, nny, nnx] Original rhs field

7 Figures (creates a new folder called: args.dataset.name + '_Small_figures'):
-  'fig_all_example_{}.png'       All fields 1 1/2 1/4 1/8 1/16 1/32
-  'fig_all_orig_example_{}.png'  All fields 1 1/2 1/4 1/8 1/16 1/32 (but now all the images now have the size nnx, nny)
-  'fig_example_{}.png'           Fields 1 1/2 and 1/4 as well as the difference interp-1/2 and interp-1/4
-  'fig_Diff_FFT_log_{}.png'      FFT of the difference between interp-orig in log scale
-  'fig_FFT_log_{}.png'           FFT of the 1 1/2 1/4 1/8 1/16 1/32 in log scale
-  'fig_Diff_FFT_{}.png'          FFT of the difference between interp-orig
-  'fig_FFT_{}.png'               FFT of the 1 1/2 1/4 1/8 1/16 1/32

"""

import argparse
from pathlib import Path
from tqdm import tqdm

import numpy as np
import matplotlib
import os

from scipy import interpolate
from interpolated_plot import plot_fields, plot_fields_all, plot_a_f_fields

matplotlib.use('Agg')


# Hardcoded parameters
domain_length = 0.01
plot_period = 1000


def dataset_clean_filter(dataset, mesh_size, domain_length):
    """
    Similar to dataset_filter, it mirrors the input
    on the four quadrants before the FFT.
    """
    # Mirror the input
    mirror = np.zeros((2 * mesh_size - 1, 2 * mesh_size - 1))
    mirror[mesh_size - 1 :, mesh_size - 1 :] = dataset
    mirror[: mesh_size - 1, : mesh_size - 1] = dataset[ -1:0:-1, -1:0:-1]
    mirror[mesh_size - 1 :, : mesh_size - 1] = - dataset[:, -1:0:-1]
    mirror[: mesh_size - 1, mesh_size - 1 :] = - dataset[-1:0:-1, :]

    # Do the Fourier transform
    transf = np.fft.fft2(mirror)
    freq = np.fft.fftfreq(mesh_size * 2 - 1, domain_length / mesh_size)
    Freq = np.fft.fftshift(np.fft.fftfreq(mesh_size * 2 - 1, domain_length / mesh_size))

    return np.abs(np.fft.fftshift(transf)), Freq


# Main loop to parallelize
def big_loop(potential, pot_quarter_orig, pot_half_orig, pot_8_orig, pot_16_orig, pot_32_orig, fig_path):
    for i in tqdm(range(len(potential[:,0,0]))):

        # Define original axes
        xx_o = np.arange(len(potential[i,:,0]))/(len(potential[i,:,0])-1)
        yy_o = np.arange(len(potential[i,0,:]))/(len(potential[i,:,0])-1)

        # Interpolation fucntion of the original field
        f = interpolate.interp2d(xx_o, yy_o, potential[i], kind='cubic')

        # Useful to create new axes
        pot_32 = potential[i,::32,::32]
        pot_16 = potential[i,::16,::16]
        pot_8 = potential[i,::8,::8]
        pot_q = potential[i,::4,::4]
        pot_h = potential[i,::2,::2]

        # New axes on which the original field needs to be interpolated
        xx_q = np.arange(len(pot_q[:,0]))/(len(pot_q[:,0])-1)
        yy_q = np.arange(len(pot_q[0,:]))/(len(pot_q[0,:])-1)
        xx_h = np.arange(len(pot_h[:,0]))/(len(pot_h[:,0])-1)
        yy_h = np.arange(len(pot_h[0,:]))/(len(pot_h[0,:])-1)
        xx_8 = np.arange(len(pot_8[:,0]))/(len(pot_8[:,0])-1)
        yy_8 = np.arange(len(pot_8[0,:]))/(len(pot_8[0,:])-1)
        xx_16 = np.arange(len(pot_16[:,0]))/(len(pot_16[:,0])-1)
        yy_16 = np.arange(len(pot_16[0,:]))/(len(pot_16[0,:])-1)
        xx_32 = np.arange(len(pot_32[:,0]))/(len(pot_32[:,0])-1)
        yy_32 = np.arange(len(pot_32[0,:]))/(len(pot_32[0,:])-1)

        # Interpolate the original field into the different reduced sizes.
        pot_quarter = f(xx_q, yy_q)
        pot_half  = f(xx_h, yy_h)
        pot_8 = f(xx_8, yy_8)
        pot_16 = f(xx_16, yy_16)
        pot_32 = f(xx_32, yy_32)

        # The interpolated fields have a reduced size, so now they need to be
        # interpolated back to the original size (nny, nnx)
        # Therefore, 5 interpolating functions are needed.
        interpo = 'cubic'
        f_h = interpolate.interp2d(xx_h, yy_h, pot_half, kind=interpo)
        f_q = interpolate.interp2d(xx_q, yy_q, pot_quarter, kind=interpo)
        f_8 = interpolate.interp2d(xx_8, yy_8, pot_8, kind=interpo)
        f_16 = interpolate.interp2d(xx_16, yy_16, pot_16, kind=interpo)
        f_32 = interpolate.interp2d(xx_32, yy_32, pot_32, kind=interpo)

        pot_quarter_orig[i] = f_q(xx_o, yy_o)
        pot_half_orig[i] = f_h(xx_o, yy_o)
        pot_8_orig[i] = f_8(xx_o, yy_o)
        pot_16_orig[i] = f_16(xx_o, yy_o)
        pot_32_orig[i] = f_32(xx_o, yy_o)

        # Perform all the plots
        if i % 100 == 0:
            plot_fields_all(potential[i], pot_half, pot_quarter, pot_8, 
                            pot_16, pot_32, i, fig_path, True)
            plot_fields_all(potential[i], pot_half_orig[i], pot_quarter_orig[i], pot_8_orig[i], 
                            pot_16_orig[i], pot_32_orig[i], i, fig_path, False)
            plot_fields(potential[i], pot_quarter, pot_quarter_orig[i]-potential[i], 
                            pot_half, pot_half_orig[i]-potential[i], i, fig_path)

            # Get 2D FFT freqs and values of actual fields               
            n_A, n_f = dataset_clean_filter(potential[i], mesh_size, domain_length)
            n2_A, n2_f = dataset_clean_filter(pot_half_orig[i], mesh_size, domain_length)
            n4_A, n4_f = dataset_clean_filter(pot_quarter_orig[i], mesh_size, domain_length)
            n8_A, n8_f = dataset_clean_filter(pot_8_orig[i], mesh_size, domain_length)
            n16_A, n16_f = dataset_clean_filter(pot_16_orig[i], mesh_size, domain_length)
            n32_A, n32_f = dataset_clean_filter(pot_32_orig[i], mesh_size, domain_length)

            # Get 2D FFT freqs and values of the difference between the original and the interpolated field
            n2_diff_A, n2_diff_f = dataset_clean_filter(potential[i] - pot_half_orig[i], mesh_size, domain_length)
            n4_diff_A, n4_diff_f = dataset_clean_filter(potential[i] - pot_quarter_orig[i], mesh_size, domain_length)
            n8_diff_A, n8_diff_f = dataset_clean_filter(potential[i] - pot_8_orig[i], mesh_size, domain_length)
            n16_diff_A, n16_diff_f = dataset_clean_filter(potential[i] - pot_16_orig[i], mesh_size, domain_length)
            n32_diff_A, n32_diff_f = dataset_clean_filter(potential[i] - pot_32_orig[i], mesh_size, domain_length)

            plot_a_f_fields(n_A, n_f, n2_A, n2_f, n4_A, n4_f,  n8_A, n8_f, 
                            n16_A, n16_f, n32_A, n32_f, i, True, False)
            plot_a_f_fields(n_A, n_f, n2_A, n2_f, n4_A, n4_f,  n8_A, n8_f, 
                            n16_A, n16_f, n32_A, n32_f, i, False, False)
            plot_a_f_fields(n_A, n_f, n2_diff_A, n2_diff_f, n4_diff_A, n4_diff_f,  n8_diff_A, n8_diff_f, 
                            n16_diff_A, n16_diff_f, n32_diff_A, n32_diff_f, i, True, True)
            plot_a_f_fields(n_A, n_f, n2_diff_A, n2_diff_f, n4_diff_A, n4_diff_f,  n8_diff_A, n8_diff_f, 
                            n16_diff_A, n16_diff_f, n32_diff_A, n32_diff_f, i, False, True)

    return pot_half_orig, pot_quarter_orig, pot_8_orig, pot_16_orig, pot_32_orig


if __name__ == '__main__':
    # CLI argument parser
    parser = argparse.ArgumentParser(description="Filter a rhs_random dataset into a new dataset")
    parser.add_argument('dataset', type=Path, help='Input dataset path')

    args = parser.parse_args()

    # Load the input dataset
    potential = np.load(args.dataset / 'potential.npy')[:200]
    rhs = np.load(args.dataset / 'physical_rhs.npy')[:200]

    mesh_size = rhs.shape[1]

    print(f'Mesh size : {mesh_size}')
    print(f'Dataset size : {rhs.shape[0]}')
    print(f'Shape of potential field : {potential.shape}')

    # Determine new dataset name as well as the folder on which the figures will be saved.
    new_name = args.dataset.name
    new_name += '_Small_interpolated_scales'
    fig_fol = args.dataset.name + '_Small_figures'

    new_path = args.dataset.with_name(new_name)  # Return new Path object with changed name
    fig_path = args.dataset.with_name(fig_fol)  # Return new Path object with changed name
    if not os.path.exists(new_path):
        new_path.mkdir()
    if not os.path.exists(fig_path):
        fig_path.mkdir()

    # Initialize tensors to be filled!
    pot_quarter_orig = np.zeros_like(potential)
    pot_half_orig = np.zeros_like(potential)
    pot_8_orig = np.zeros_like(potential)
    pot_16_orig = np.zeros_like(potential)
    pot_32_orig = np.zeros_like(potential)

    # Perform the main loop
    pot_half_orig, pot_quarter_orig, pot_8_orig, pot_16_orig, pot_32_orig = big_loop(
        potential, pot_quarter_orig, pot_half_orig, pot_8_orig, pot_16_orig, pot_32_orig, fig_path
    )

    # Expand on new dimensions and concatenate on the new tensor.
    potential_expand = np.expand_dims(potential, axis=1)
    pot_half_orig = np.expand_dims(pot_half_orig, axis=1)
    pot_quarter_orig = np.expand_dims(pot_quarter_orig, axis=1)
    pot_8_orig = np.expand_dims(pot_8_orig, axis=1)
    pot_16_orig = np.expand_dims(pot_16_orig, axis=1)
    pot_32_orig = np.expand_dims(pot_32_orig, axis=1)

    new_pot = np.concatenate((potential_expand, pot_half_orig, pot_quarter_orig, 
                                pot_8_orig, pot_16_orig, pot_32_orig), axis = 1)

    # Save the new dataset
    np.save(new_path / 'potential_interp.npy', new_pot)
    np.save(new_path / 'potential.npy', potential)
    np.save(new_path / 'physical_rhs.npy', rhs)

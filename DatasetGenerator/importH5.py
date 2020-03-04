########################################################################################################################
#                                                                                                                      #
#                                            Import H5 solutions from AVIP                                             #
#                                                                                                                      #
#                                      Guillaume Bogopolsky, CERFACS, 26.02.2020                                       #
#                                                                                                                      #
########################################################################################################################

import os
import numpy as np
import antares as asc
from tqdm import tqdm

asc.set_progress_bar(True)


def load_avbp_run(solut_path, mesh_path=None, base=None, verbose=False):
    """ Load an AVBP case with Antares. """
    if base is None and mesh_path is None:
        raise ValueError('Specify either a mesh_path to create a base or an existing base')
    elif base is None and mesh_path is not None:
        # Load grid data from mesh file
        reader = asc.Reader('hdf_avbp')
        reader['filename'] = mesh_path
        reader['shared'] = True
        base = reader.read()

    # Add solution
    reader = asc.Reader('hdf_avbp')
    reader['filename'] = solut_path
    reader['base'] = base
    base = reader.read()

    if verbose:
        print(base[0])
        print(base[0][0])

    return base


def ant_to_numpy_quad(instant, variables, out_shape):
    """ Extract a list of variables on a quad mesh from an Antares base in a 2D Numpy array. """
    if isinstance(variables, list):
        res = list()
        for key in variables:
            res.append(instant[key].reshape(out_shape))
    else:
        raise TypeError('Variables parameter should be a list')
    return res


def save_variables_from_instant(zone, instant_number, save_path, variables, out_shape, verbose=False):
    """ Extract the variables from a instant and save them as a .npy file. """
    res = ant_to_numpy_quad(zone[instant_number], variables, out_shape)
    for i, name in enumerate(variables):
        if not os.path.exists(f'{save_path}/{name}'):
            os.makedirs(f'{save_path}/{name}')
        if verbose:
            print(f'Writing {save_path}/{name}/{name}_{instant_number}.npy')
        np.save(f'{save_path}/{name}/{name}_{instant_number}.npy', res[i])


def save_multiple_file(base, save_path, variables, out_shape):
    """ Save as a single .npy file per instant. """
    for n_instant in tqdm(base[0].keys()):
        save_variables_from_instant(base[0], n_instant, save_path, variables, out_shape)


def save_single_file(base, save_path, variables, out_shape):
    """ Save as a single .npy file for all instants. """
    for name in variables:
        print('Saving {}...'.format(name))
        nb_instants = len(base[0].keys())
        dtype = base[0][0][name].dtype
        out = np.zeros((nb_instants, *out_shape), dtype=dtype)
        for n_instant, name_instant in tqdm(enumerate(base[0].keys()), total=nb_instants):
            out[n_instant, :, :] = base[0][n_instant][name].reshape(out_shape)
        np.save(f'{save_path}/{name}.npy', out)


if __name__ == '__main__':
    verbose = False

    # Load base
    solut_path = '/Users/bogopolsky/THESE/POISSON/PlasmaNet/datasets/64x64/sliding_gaussian_zerofield/src/RUN/SOLUT_zerofield/poisson_000<instant>.h5'
    mesh_path = '/Users/bogopolsky/THESE/POISSON/PlasmaNet/datasets/64x64/sliding_gaussian_zerofield/src/MESH/mesh_quad_64.mesh.h5'
    base = load_avbp_run(solut_path, mesh_path, verbose=verbose)

    # Save base
    print('Converting and saving...')
    save_path = '/Users/bogopolsky/THESE/POISSON/PlasmaNet/datasets/64x64/sliding_gaussian_zerofield'
    variables = ['E_field_x', 'E_field_y', 'potential', 'rhs', 'physical_rhs']
    out_shape = (64, 64)

    # Save as multiple .npy files for each instant
    # save_multiple_file(base, save_path, variables, out_shape)

    # Save as a single .npy file for all instants
    save_single_file(base, save_path, variables, out_shape)



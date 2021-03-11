import torch
import numpy as np
from ..common.operators_numpy import lapl


def lapl_diff(potential, physical_rhs, dx, dy, nx, ny):
    interior_diff = abs(lapl(potential, dx, dy, nx, ny) + physical_rhs)
    interior_diff[0, :] = 0
    interior_diff[-1, :] = 0
    interior_diff[:, 0] = 0
    interior_diff[:, -1] = 0
    return interior_diff


def func_energy(potential, electric_field, physical_rhs, voln):
    field_energy = 1 / 2 * (electric_field[0]**2 + electric_field[1]**2)
    potential_energy = physical_rhs * potential
    energy = np.sum((field_energy - potential_energy) * voln)
    return energy


def func_energy_torch(potential, electric_field, physical_rhs, voln):
    field_energy = 1 / 2 * (electric_field[0]**2 + electric_field[1]**2)
    potential_energy = physical_rhs * potential
    energy = torch.sum((field_energy - potential_energy) * voln)
    return energy

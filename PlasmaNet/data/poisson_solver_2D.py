########################################################################################################################
#                                                                                                                      #
#                                            2D Poisson solver using numpy                                             #
#                                                                                                                      #
#                               Lionel Cheng, Guillaume Bogopolsky, CERFACS, 10.03.2020                                #
#                                                                                                                      #
########################################################################################################################

import os
import sys
import time
from multiprocessing import Pool

# Disable multithreading for OpenBlas, which might kill vectorisation in some cases.
# Moreover, the matrix size is not too big, so the overhead will no be negligible.
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as co
from scipy import sparse
from scipy.sparse.linalg import spsolve
from tqdm import tqdm


def laplace_square_matrix(n_points):
	diags = np.zeros((5, n_points * n_points))

	# Filling the diagonal
	diags[0, :] = - 4 * np.ones(n_points**2)
	diags[1, :] = np.ones(n_points**2)
	diags[2, :] = np.ones(n_points**2)
	diags[3, :] = np.ones(n_points**2)
	diags[4, :] = np.ones(n_points**2)

	# Correcting the diagonal to take into account dirichlet boundary conditions
	for i in range(n_points**2):
		if i < n_points or i >= (n_points - 1) * n_points or i % n_points == 0 or i % n_points == n_points - 1:
			diags[0, i] = 1
			diags[1, min(i + 1, n_points**2 - 1)] = 0
			diags[2, max(i - 1, 0)] = 0
			diags[3, min(i + n_points, n_points**2 - 1)] = 0
			diags[4, max(i - n_points, 0)] = 0

	# Creating the matrix
	return sparse.csc_matrix(sparse.dia_matrix((diags, [0, 1, -1, n_points, -n_points]), shape=(n_points**2, n_points**2)))


def dirichlet_bc(rhs, n_points, up, down, left, right):
	rhs[:n_points] = up * np.zeros(n_points)
	rhs[n_points*(n_points - 1):] = down * np.zeros(n_points)
	rhs[:n_points*(n_points - 1) + 1:n_points] = left * np.zeros(n_points)
	rhs[n_points-1::n_points] = right * np.zeros(n_points)


def plot_fig(X, Y, potential, physical_rhs, save=False, name='potential_2D', nit=0):
	# Plotting the potential
	matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
	fig, [ax1, ax2] = plt.subplots(ncols=2, figsize=(14, 7))
	CS1 = ax1.contourf(X, Y, physical_rhs, 100, cmap='jet')
	cbar1 = fig.colorbar(CS1, pad=0.05, fraction=0.08, ax=ax1, aspect=5)
	cbar1.ax.set_ylabel(r'$\rho/\epsilon_0$ [V.m$^{-2}$]')
	ax1.set_aspect("equal")
	CS2 = ax2.contourf(X, Y, potential, 100, cmap='jet')
	cbar2 = fig.colorbar(CS2, pad=0.05, fraction=0.08, ax=ax2, aspect=5)
	cbar2.ax.set_ylabel('Potential [V]')
	ax2.set_aspect("equal")

	if save:
		plt.savefig(name + str(nit), bbox_inches='tight')
	else:
		plt.show()


def gaussian(x, y, amplitude, x0, y0, sigma_x, sigma_y):
	return amplitude * np.exp(-((x - x0)/sigma_x)**2 - ((y - y0)/sigma_y)**2)


def params(nits, n_points):
	""" Generator for the parameters. """
	xmin, xmax = 0, 0.01
	ymin, ymax = 0, 0.01
	dx = (xmax - xmin) / (n_points - 1)
	dy = (ymax - ymin) / (n_points - 1)
	x = np.linspace(xmin, xmax, n_points)
	y = np.linspace(ymin, ymax, n_points)

	X, Y = np.meshgrid(x, y)

	A = laplace_square_matrix(n_points)

	# Creating the rhs
	ni0 = 1e16
	sigma_x, sigma_y = 1e-3, 1e-3
	for nit in range(nits):
		x0, y0 = 0.3e-2 * (1 + nit / nits), 0.3e-2 * (1 + nit / nits)
		yield (ni0, sigma_x, sigma_y, x0, y0, X, Y, dx, A, n_points)


def compute(args):
	""" Compute the solution for the given parameters. """
	ni0, sigma_x, sigma_y, x0, y0, X, Y, dx, A, n_points = args

	# Interior rhs
	physical_rhs = gaussian(X.reshape(-1), Y.reshape(-1), ni0, x0, y0, sigma_x, sigma_y) * co.e / co.epsilon_0
	rhs = - physical_rhs * dx ** 2

	# Imposing Dirichlet boundary conditions
	dirichlet_bc(rhs, n_points, 0, 0, 0, 0)

	# Solving the sparse linear system
	tmp_pot = spsolve(A, rhs).reshape(n_points, n_points)
	tmp_rhs = physical_rhs.reshape(n_points, n_points)

	return (tmp_pot, tmp_rhs)


if __name__ == '__main__':

	plot = False

	nits = 1000
	n_points = 128

	if len(sys.argv) == 2:
		processes = int(sys.argv[1])
	else:
		processes = 2

	time_start = time.time()
	with Pool(processes=processes) as p:
        # Performance is very sensitive to chunksize, which depends on the machine.
        # On kraken, use chunksize=5 and processes=36 or 72
        # On MacBookAir, chunksize=50 and processes=2
		results = list(tqdm(p.imap(compute, params(nits, n_points), chunksize=5), total=nits))

	# Extract results to numpy arrays for saving
	potential = np.zeros((nits, n_points, n_points))
	physical_rhs_list = np.zeros((nits, n_points, n_points))
	for i, (pot, rhs) in enumerate(results):
		potential[i, :, :] = pot
		physical_rhs_list[i, :, :] = rhs

		if i % 20 == 0 and plot:
			_, _, _, _, _, X, Y, _, _, _ = list(params(1, n_points))[0]
			plot_fig(X, Y, potential[i, :, :], physical_rhs_list[i, :, :], save=True, name='potential_2D_gauss', nit=i)

	time_stop = time.time()
	np.save('potential.npy', potential)
	np.save('physical_rhs.npy', physical_rhs_list)
	print('Elapsed time (s) : %.2e' % (time_stop - time_start))

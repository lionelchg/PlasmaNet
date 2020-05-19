import os
import numpy as np
import matplotlib.pyplot as plt

fig_dir = 'figures/ft/'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

def ft(f, N):
    T = np.zeros((N, N)) + np.zeros((N, N)) * 1j
    W_N = np.exp(-1j * 2 * np.pi / N)
    for k in range(N):
        for j in range(N):
            T[k, j] = np.exp(-1j * 2 * np.pi / N * (k - N / 2) * j) / N

    return np.real(np.dot(T, f))

def ift(f, N):
    C = np.zeros((N, N))
    W_N = np.exp(-1j)

if __name__ == '__main__':
    xmin, xmax = 0, 2 * np.pi
    Lx = xmax - xmin
    nnx = 501
    ncx = nnx - 1
    x = np.linspace(xmin, xmax, nnx)
    k_range = range(ncx)
    dx = Lx / (nnx - 1)

    test_function = 10 * np.sin(x) + 5 * np.sin(3 * x) + 3 * np.sin(2 * x) + np.zeros_like(x) * 1j
    # df_test = ft(test_function[:-1], ncx)
    df_test = np.zeros(ncx)
    for k in range(ncx):
        

    transf = np.fft.fft(test_function)
    freq = np.fft.fftfreq(nnx, Lx / nnx)

    fig, axes = plt.subplots(nrows=3, figsize=(8, 8))

    axes[0].plot(x, test_function)
    axes[0].grid(True)
    axes[0].set_xlabel('$x$')

    axes[1].plot(k_range, np.real(df_test))
    axes[1].grid(True)
    axes[1].set_xlabel('$k$')

    axes[2].plot(freq, np.real(transf))
    axes[2].grid(True)
    axes[2].set_xlabel('$k$')

    plt.tight_layout()
    plt.savefig(fig_dir + 'test_function')
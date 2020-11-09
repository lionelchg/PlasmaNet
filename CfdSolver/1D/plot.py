import matplotlib.pyplot as plt
from test_funcs import gaussian, step, packet_wave


def plot_fd(x_th, x, u_gauss, u_step, u_2pw, u_4pw, schemes, figtitle, figname):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))

    axes = axes.reshape(-1)

    axes[0].plot(x_th, gaussian(x_th, 1, 0.3))
    for i_scheme, scheme in enumerate(schemes):
        axes[0].plot(x, u_gauss[i_scheme, :], '.', ms=4, label=scheme)
    ax_prop(axes[0])

    axes[1].plot(x_th, step(x_th, 1))
    for i_scheme, scheme in enumerate(schemes):
        axes[1].plot(x, u_step[i_scheme, :], '.', ms=4, label=scheme)
    ax_prop(axes[1])

    axes[2].plot(x_th, packet_wave(x_th, 1, 0.5))
    for i_scheme, scheme in enumerate(schemes):
        axes[2].plot(x, u_2pw[i_scheme, :], '.', ms=4, label=scheme)
    ax_prop(axes[2])

    axes[3].plot(x_th, packet_wave(x_th, 1, 0.25))
    for i_scheme, scheme in enumerate(schemes):
        axes[3].plot(x, u_4pw[i_scheme, :], '--', ms=4, label=scheme)
    ax_prop(axes[3])

    fig.suptitle(figtitle)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(figname, bbox_inches='tight')
    plt.close(fig)


def ax_prop(ax):
    ax.legend()
    ax.grid(True)

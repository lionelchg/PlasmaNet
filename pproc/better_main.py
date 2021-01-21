import os
import re
import argparse

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
from cfdsolver.utils import create_dir

def ax_prop(ax, title):
    ax.legend()
    ax.grid(True)
    ax.set_title(title)
    ax.set_yscale('log')
    ax.legend()

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='NetworksPostprocessing')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='Config file path (default: None)')
    args = args.parse_args()

    fig_dir = 'figures/3-networks/test/'
    create_dir(fig_dir)
    
    data_dir = "../outputs/3-networks/log/"
    train_names = {"UNet5 - random_16":"config_1/random_16/1228_140218/events.out.tfevents.1609160549.krakengpu1.cluster.74639.0",
                "UNet5 - random_32":"config_1/random_32/1228_140150/events.out.tfevents.1609160519.krakengpu1.cluster.74548.0",
                "UNet6 - random_16":"config_2/random_16/1228_140113/events.out.tfevents.1609160481.krakengpu2.cluster.26220.0",
                "UNet6 - random_32":"config_2/random_32/1228_140131/events.out.tfevents.1609160503.krakengpu2.cluster.26294.0"}
    
    # Wanted metrics and losses on the plot
    plot_var = 'ComposedLosses'
    variables = ["DirichletBoundaryLoss", "LaplacianLoss"]
    data_types = ["train", "valid"]

    # Number of variables, data_types
    nvariables, ndtypes = len(variables), len(data_types)

    # Figure for variables and metrics
    fig, axes = plt.subplots(nrows=nvariables, ncols=ndtypes, figsize=(5 * nvariables, 5 * ndtypes))

    for train_name, end_folder in train_names.items():
        case_folder = data_dir + end_folder
        event_acc = EventAccumulator(case_folder)
        event_acc.Reload()
        
        for i, var in enumerate(variables):
            for j, data_type in enumerate(data_types):
                # E. g. get wall clock, number of steps and value for a scalar 'Accuracy'
                _, epochs, vals = zip(*event_acc.Scalars(f'{plot_var}/{var}/{data_type}'))
                axes[i][j].plot(epochs, vals, label=train_name)
        
    for i in range(nvariables):
        for j in range(ndtypes):
            ax_prop(axes[i][j], f'{variables[i]}/{data_types[j]}')

    fig.savefig(fig_dir + plot_var, bbox_inches='tight')
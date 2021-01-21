import os
import re
import argparse

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
from cfdsolver.utils import create_dir

def ax_prop(ax, x, y, label, title):
    ax.plot(x, y, label=label)
    ax.grid(True)
    ax.set_title(title)
    ax.set_yscale('log')

if __name__ == '__main__':
    fig_dir = 'figures/3-networks/nscales/'
    create_dir(fig_dir)
    
    data_dir = "../outputs/3-networks/log/"
    train_names = {"UNet5 - random_16":"config_1/random_16/1228_140218/events.out.tfevents.1609160549.krakengpu1.cluster.74639.0",
                "UNet5 - random_32":"config_1/random_32/1228_140150/events.out.tfevents.1609160519.krakengpu1.cluster.74548.0",
                "UNet6 - random_16":"config_2/random_16/1228_140113/events.out.tfevents.1609160481.krakengpu2.cluster.26220.0",
                "UNet6 - random_32":"config_2/random_32/1228_140131/events.out.tfevents.1609160503.krakengpu2.cluster.26294.0"}
    
    # Wanted metrics and losses on the plot
    losses = ["DirichletBoundaryLoss", "LaplacianLoss"]
    metrics = ["residual", "Eresidual"]
    data_types = ["train", "valid"]

    # Number of losses, metrics, data_types
    nlosses, nmetrics, ndtypes = len(losses), len(metrics), len(data_types)

    # Figure for losses and metrics
    fig1, axes1 = plt.subplots(nrows=nlosses, ncols=ndtypes, figsize=(5 * nlosses, 5 * ndtypes))
    fig2, axes2 = plt.subplots(nrows=nmetrics, ncols=ndtypes, figsize=(5 * nmetrics, 5 * ndtypes))

    for train_name, end_folder in train_names.items():
        case_folder = data_dir + end_folder
        event_acc = EventAccumulator(case_folder)
        event_acc.Reload()
        
        for i, loss in enumerate(losses):
            for j, data_type in enumerate(data_types):
                # E. g. get wall clock, number of steps and value for a scalar 'Accuracy'
                _, epochs, vals = zip(*event_acc.Scalars(f'ComposedLosses/{loss}/{data_type}'))
                ax_prop(axes1[i][j], epochs, vals, train_name, f'{loss}/{data_type}')
        
        for i in range(nlosses):
            for j in range(ndtypes):
                axes1[i][j].legend()
        
        for i, metric in enumerate(metrics):
            for j, data_type in enumerate(data_types):
                # E. g. get wall clock, number of steps and value for a scalar 'Accuracy'
                _, epochs, vals = zip(*event_acc.Scalars(f'Metrics/{metric}/{data_type}'))
                ax_prop(axes2[i][j], epochs, vals, train_name, f'{metric}/{data_type}')
        
        for i in range(nmetrics):
            for j in range(ndtypes):
                axes2[i][j].legend()

    fig1.savefig(fig_dir + "losses", bbox_inches='tight')
    fig2.savefig(fig_dir + "metrics", bbox_inches='tight')
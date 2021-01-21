import os
import glob
import re
import argparse
import yaml
import matplotlib.pyplot as plt
import glob

# To import the trainings
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from cfdsolver.utils import create_dir

def ax_prop(ax, title):
    ax.legend()
    ax.grid(True)
    ax.set_title(title)
    ax.set_yscale('log')
    ax.legend()

def plot_variables(train_names, var_dir, variables, data_types, figname):
    """ Plot the variables specified in `variables` and `data_types` for
    all the training specified in train_names dicitonnary """
    # Number of variables, data_types
    nvariables, ndtypes = len(variables), len(data_types)

    # Figure for variables and metrics
    fig, axes = plt.subplots(nrows=nvariables, ncols=ndtypes, figsize=(5 * ndtypes, 5 * nvariables))
    axes = axes.reshape(-1)

    for train_name, event_file in train_names.items():
        event_acc = EventAccumulator(event_file)
        event_acc.Reload()
        
        index = 0
        for i, var in enumerate(variables):
            for j, data_type in enumerate(data_types):
                # E. g. get wall clock, number of steps and value for a scalar 'Accuracy'
                _, epochs, vals = zip(*event_acc.Scalars(f'{var_dir}/{var}/{data_type}'))
                axes[index].plot(epochs, vals, label=train_name)
                index += 1

    index = 0
    for i in range(nvariables):
        for j in range(ndtypes):
            ax_prop(axes[index], f'{variables[i]}/{data_types[j]}')
            index += 1

    fig.savefig(figname, bbox_inches='tight')

def autocomplete(data_dir, train_names):
    """ Autocomplete the path of the trainings events file.
    Always take the last training in terms of dates """
    for train_name, end_folder in train_names.items():
        for sub in os.scandir(data_dir + end_folder):
            if sub.is_dir():
                event_file = glob.glob(sub.path + '/events*')[0]
                train_names[train_name] = event_file

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='NetworksPostprocessing')
    args.add_argument('-c', '--config', required=True, type=str,
                      help='Config file path (default: None)')
    args = args.parse_args()

    with open(args.config, 'r') as yaml_stream:
        cfg = yaml.safe_load(yaml_stream)

    # Create directory for the figures
    fig_dir = 'figures/' + cfg['casename']
    create_dir(fig_dir)
    
    # Directories to find the training data
    data_dir = cfg['data']['main_dir']
    train_names = cfg['data']['trainings']
    autocomplete(data_dir, train_names)

    params_figs = cfg['output']

    for figname in params_figs:
        # Wanted metrics and losses on the plot
        var_dir = params_figs[figname]['var_dir']
        variables = params_figs[figname]['variables']
        data_types = params_figs[figname]['data_types']

        # Plot the wanted data
        plot_variables(train_names, var_dir, variables, data_types, fig_dir + figname)
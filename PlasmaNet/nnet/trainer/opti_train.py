########################################################################################################################
#                                                                                                                      #
#                                           PlasmaNet.nnet: train model                                                #
#                                                                                                                      #
#                         Guillaume Bogopolsky, Lionel Cheng, Ekhi Ajuria, CERFACS, 10.03.2020                         #
#                                                                                                                      #
########################################################################################################################

import os
import argparse
import collections

import numpy as np
import torch
import yaml
import pdb
import matplotlib.pyplot as plt

import optuna
from pathlib import Path

import PlasmaNet.nnet.data.data_loaders as module_data
import PlasmaNet.nnet.model.loss as module_loss
import PlasmaNet.nnet.model.metric as module_metric
import PlasmaNet.nnet.model as module_arch
from PlasmaNet.nnet.parse_config import ConfigParser
from PlasmaNet.nnet.trainer import Trainer
from PlasmaNet.poissonsolver.network import PoissonNetworkOpti

# Fix random seeds for reproducibility
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

def model_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

def run_train_opti(config, config_eval):

    logger = config.get_logger('train')

    # Setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    # Build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    #num_params = model_params(model)
    num_params = model.nparams
    logger.info(model)

    # Initialize model weights
    if config['initializer'] != 'off':
        def init_weights(m):
            if type(m) == torch.nn.Conv2d:
                getattr(torch.nn.init, config['initializer']['type'])(m.weight, **config['initializer']['args'])
        model.apply(init_weights)

    # Get function handles of loss and metrics
    criterion = config.init_obj('loss', module_loss)
    metrics = [getattr(module_metric, metric) for metric in config['metrics']]

    # Build optimizer, learning rate scheduler
    # Disable scheduler by commenting all lines with lr_scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    if config['lr_scheduler'] != 'off':
        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    else:
        lr_scheduler = None

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()

    # When training has finished, evaluate network
    poisson_nn = PoissonNetworkOpti(config_eval['network'], trainer.model)
    score = poisson_nn.evaluateopti()
    return score, num_params 



def main():
    args = argparse.ArgumentParser(description='PlasmaNet.nnet')
    args.add_argument('-ct', '--configtrain', default=None, type=str,
                      help='training config file path (default: None)')
    args.add_argument('-ce', '--configeval', default=None, type=str,
                      help='evaluation config file path (default: None)')
    args = args.parse_args()

    with open(args.configtrain, 'r') as yaml_stream:
        cfg_dict_train = yaml.safe_load(yaml_stream)

    with open(args.configeval, 'r') as yaml_stream:
        cfg_dict_eval = yaml.safe_load(yaml_stream)    

    # Creation of config object
    config = ConfigParser(cfg_dict_train)

    # 1. Define an objective function to be maximized.
    def objective(trial):

        # 2. Define Variables to optimize
        config['arch']['args']['n_filters'] = trial.suggest_int('n_filters', 2, 10, 2)
        config['arch']['args']['filter_size'] = trial.suggest_int('filter_size', 10, 100, 10)
        config['arch']['args']['scales'] = trial.suggest_int('scales', 1, 6, 1) 
        score, num_params = run_train_opti(config, cfg_dict_eval)
        return score, num_params


    # 3. Create a study object and optimize the objective function.
    study = optuna.create_study(study_name='parameter_opti',
                    storage='sqlite:////scratch/cfd/PlasmaDL/optim/study/study_long.db',
                    load_if_exists=True, directions=["minimize", "minimize"])

    # 4. Plot final results 
    pruned_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE]
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))


    print("Best trials:")
    trials = sorted(study.best_trials, key=lambda t: t.values)

    for trial in trials:
        print("  Trial#{}".format(trial.number))
        print("    Values: score ={}, num_params={}".format(trial.values[0], trial.values[1]))
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

    results = np.zeros((5, len(complete_trials)))
    colors = []
    for j, trial in enumerate(complete_trials):
        print("  Trial#{}".format(trial.number))
        print("    Values: score ={}, num_params={}".format(trial.values[0], trial.values[1]))
        results[0, j] = trial.values[0]
        results[1, j] = trial.values[1]
        print("  Params: ")
        ww = 0
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
            if key == 'scales' and value == 1:
                colors.append('blue')
            elif key == 'scales' and value == 2:
                colors.append('orange')
            elif key == 'scales' and value == 3:
                colors.append('green')
            elif key == 'scales' and value == 4:
                colors.append('red')
            elif key == 'scales' and value == 5:
                colors.append('brown')
            elif key == 'scales' :
                colors.append('white')

            results[2+ww, j] = value
            ww +=1



    plt.scatter(results[1], results[0], c= colors, s= results[2]*10, label = 'scales')
    plt.ylabel('Accuracy')
    plt.xlabel('N Parameters')
    plt.xscale('log')
    plt.legend()
    plt.savefig('/scratch/cfd/PlasmaDL/optim/results.png')

    pdb.set_trace()

    study.optimize(objective, n_trials= 200) #, n_jobs= 1) #100)

    # 4. Plot final results 
    pruned_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))


    print("Best trials:")
    trials = sorted(study.best_trials, key=lambda t: t.values)

    for trial in trials:
        print("  Trial#{}".format(trial.number))
        print("    Values: score ={}, num_params={}".format(trial.values[0], trial.values[1]))
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))


if __name__ == '__main__':
    main()

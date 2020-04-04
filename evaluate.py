########################################################################################################################
#                                                                                                                      #
#                                              PlasmaNet: evaluate model                                               #
#                                                                                                                      #
#                         Guillaume Bogopolsky, Lionel Cheng, Ekhi Ajuria, CERFACS, 10.03.2020                         #
#                                                                                                                      #
########################################################################################################################

import argparse
import collections

import torch
from tqdm import tqdm

import PlasmaNet.data.data_loaders as module_data
import PlasmaNet.model.loss as module_loss
import PlasmaNet.model.metric as module_metric
import PlasmaNet.model.multiscalenet as module_arch
from PlasmaNet.parse_config import ConfigParser
from PlasmaNet.trainer.trainer import plot_batch


def main(config):
    logger = config.get_logger('test')

    # Setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)

    # Build model architecture
    model = config.init_obj('arch', module_arch)

    # Get function handles of loss and metrics
    loss_fn = config.init_obj('loss', module_loss)
    metric_fns = [getattr(module_metric, metric) for metric in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # Prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    # Output configuration
    # out_dir = Path('./outputs/eval')
    # out_dir.mkdir(parents=True, exist_ok=True)
    out_dir = config.fig_dir

    with torch.no_grad():
        for i, (data, target, data_norm, target_norm) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            data_norm, target_norm = data_norm.to(device), target_norm.to(device)
            output = model(data)

            #
            # save sample images, or do something with output here
            #

            fig = plot_batch(output, target, data, 0, i)
            fig.savefig(out_dir / 'batch_{:05d}.png'.format(i), dpi=150, bbox_inches='tight')

            # Computing loss, metrics on test set
            if loss_fn.require_input_data():
                loss = loss_fn(output, target, data=data, target_norm=target_norm, data_norm=data_norm)
            else:
                loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for j, metric in enumerate(metric_fns):
                total_metrics[j] += metric(output, target, config) * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        metric.__name__: total_metrics[i].item() / n_samples for i, metric in enumerate(metric_fns)
    })
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PlasmaNet')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to checkpoint to resume (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # Custom CLI options to modify configuration from default values given in yaml file
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['-ds', '--dataset'], type=str, target='data_loader;args;data_dir'),
        CustomArgs(['-n', '--name'], type=str, target='name')

    ]
    config = ConfigParser.from_args(args, options)
    main(config)

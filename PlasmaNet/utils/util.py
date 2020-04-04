########################################################################################################################
#                                                                                                                      #
#                                                Some utility functions                                                #
#                                                                                                                      #
#                                     Guillaume Bogopolsky, CERFACS, 03.03.2020                                        #
#                                                                                                                      #
########################################################################################################################

from itertools import repeat
from pathlib import Path

import pandas as pd
import yaml


def read_yaml(fname):
    """ Read yaml configuration file. """
    fname = Path(fname)
    with fname.open('rt') as handle:
        return yaml.safe_load(handle)


def write_yaml(content, fname):
    """ Write yaml configuration file. """
    fname = Path(fname)
    with fname.open('wt') as handle:
        yaml.safe_dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    """ Wrapper function for an endless data loader. """
    for loader in repeat(data_loader):
        yield from loader


class MetricTracker:
    """
    Track metrics from the network by storing them in a pandas.DataFrame and sending them to TensorBoard if a writer
    is specified (deactivated functionality)
    """
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        """ Reset all keys to zero. """
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        """ Update key in DataFrame with the given value and a count if specified. """
        # if self.writer is not None:
        #     self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        """ Return the average for the given key. """
        return self._data.average[key]

    def result(self):
        """ Return averages as a dictionary. """
        return dict(self._data.average)

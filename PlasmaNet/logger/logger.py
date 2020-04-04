########################################################################################################################
#                                                                                                                      #
#                                                Setup logging manager                                                 #
#                                                                                                                      #
#                                     Guillaume Bogopolsky, CERFACS, 03.03.2020                                        #
#                                                                                                                      #
########################################################################################################################

import logging
import logging.config
import os
from pathlib import Path

from ..utils import read_yaml


def setup_logging(save_dir, log_config=os.path.abspath(__file__), default_level=logging.INFO):
    """ Setup logging configuration. """
    log_config = Path(log_config)
    log_config = log_config.parent / 'logger_config.yml'
    if log_config.is_file():
        config = read_yaml(log_config)
        # modify logging paths based on run config
        for _, handler in config['handlers'].items():
            if 'filename' in handler:
                handler['filename'] = str(save_dir / handler['filename'])

        logging.config.dictConfig(config)
    else:
        print('Warning: logging configuration file is not found in {}'.format(log_config))
        logging.basicConfig(level=default_level)

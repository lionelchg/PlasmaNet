# PlasmaNet

Solving the electrostatic Poisson equation for plasma simulations using a deep learning approach.

## Usage

### Installation

#### Python environment

First set up the correct Python environment. If you are working on Kraken, you may use the already prepared
environment `/scratch/cfd/bogopolsky/DL/dl_env` by using:

`source /scratch/cfd/bogopolsky/DL/dl_env/bin/activate`

To use your own environment, follow the above instructions to activate a Python 3.8 environment, create a new 
Python 3.8 venv and activate it:

`python -m venv path/to/your/env`  
`source path/to/your/env/bin/activate`

Install the required packages:

`pip install -r requirements.txt`

#### Install PlasmaNet

Clone and install the repository:

`git clone https://nitrox.cerfacs.fr/cfd-apps/plasmanet`  
`cd plasmanet`  
`pip install -e .`

You are now ready!

### Train a model

To train a model, create a working directory. You simply need:
- to copy the `train.py` and `config.yml` file from the PlasmaNet repository
- edit `config.yml` 
- launch the training using the `batch_KRAKEN` template also available in the repository

### Evaluate a model

Similar to the training, using the `evaluate.py` and `eval.yml` files.
You will want to set the `-r` option in the batch file to specify the trained model you want to use.

### Configuration file

The configuration file is organised in sections for each element of the network, with the following keys:
- `type`, the name of the object that will be used (e.g. `Adam` as optimizer)
- `args`, the list of the arguments that will be passed to the constructor of the specified `type` (e.g. `lr: 4e-4`)
- `pipe_config`, an optional boolean (default False) if the constructor requires the `config` object containing the 
configuration options (e.g. the `globals` field for the normalization of the dataset)

This configuration is used in `train.py` by the `init_obj` method of the `ConfigParser` class to initialize 
the corresponding object by looking for it in the specified module.

```python
import PlasmaNet.model.loss as module_loss

criterion = config.init_obj('loss', module_loss)
```

More information about the configuration file and its use can be found in the README of the template project
[here](https://github.com/victoresque/pytorch-template).


## Credits

- Ekhi Ajuria-Illarramendi, for the original work during the summer 2019 workshop
- @victoresque on GitHub for his [pytorch-template](https://github.com/victoresque/pytorch-template)

########################################################################################################################
#                                                                                                                      #
#                     PlasmaNet: Solving the electrostatic Poisson equation for plasma simulations                     #
#                                                                                                                      #
#                         Guillaume Bogopolsky, Lionel Cheng, Ekhi Ajuria, CERFACS, 26.02.2020                         #
#                                                                                                                      #
########################################################################################################################

# Main program

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import yaml
from torch.utils.data import Dataset

from PlasmaNet import MultiSimpleNet
from PlasmaNet.trainer import train
from PlasmaNet.validator import validate

matplotlib.use('Agg')

# Load the Config file
with open('config.yml', 'r') as handle:
    default_params = yaml.safe_load(handle)

option_params = default_params['option_params']
data_params = default_params['data_params']
optimization_params = default_params['optimization_params']

data_channels = 1

# Baseline_folder
folder_base = data_params['output_dir']
folder_name = data_params['output_name']
folder = os.path.join(folder_base, folder_name)
load_folder = data_params['data_dir']
folder_state = option_params['state_dir']

if not os.path.exists(folder):
    os.makedirs(folder)

saving_val = folder + '/Val_Images'
saving_train = folder + '/Train_Images'
saving_div = folder + '/Div_Images'
saving_model = folder + '/model_saves'

if not os.path.exists(saving_val):
    os.makedirs(saving_val)
if not os.path.exists(saving_train):
    os.makedirs(saving_train)
if not os.path.exists(saving_div):
    os.makedirs(saving_div)
if not os.path.exists(saving_model):
    os.makedirs(saving_model)

# Data loader (primitive version, dixit Ekhi)

file_Ex = load_folder + '/E_field_x.npy'
file_Ey = load_folder + '/E_field_y.npy'
file_rhs = load_folder + '/physical_rhs.npy'
file_potential = load_folder + '/potential.npy'

rhs = np.load(file_rhs)
potential = np.load(file_potential)

# Normalize
rhs_max = rhs.max()
potential_max = potential.max()

rhs /= rhs_max
potential /= potential_max

# Plot Histograms

# Fixing random state for reproducibility

# the histogram of the data
n, bins, patches = plt.hist(rhs.flatten(), 100, density=True, facecolor='g')

plt.xlabel('Rhs')
plt.ylabel('Probability')
plt.title('Rhs Distribution')
plt.show()
plt.savefig(folder + '/Div_Images' + '/RHS_Distribution.png')
plt.close()

# the histogram of the data

n1, bins1, patches1 = plt.hist(potential.flatten(), 100, density=True, facecolor='g')

plt.xlabel('Potential')
plt.ylabel('Probability')
plt.title('Potential Distribution')
plt.show()
plt.savefig(folder + '/Div_Images' + '/potential_Distribution.png')
plt.close()

# N of Testing values and total size
tt = len(potential[:, 0, 0])
t_n = 1  # Separation of test dataset (no test for now)

h = np.int(len(potential[0, :, 0]))
w = np.int(len(potential[0, 0, :]))

# Declaration of the X_train and X_test Tensors
X_Train = np.zeros((tt - t_n, 1, h, w))
X_Test = np.zeros((t_n, 1, h, w))
Y_Train = np.zeros((tt - t_n, 1, h, w))
Y_Test = np.zeros((t_n, 1, h, w))

X_Train = np.expand_dims(rhs[t_n:, :, :], axis=1)
X_Test = np.expand_dims(rhs[:t_n, :, :], axis=1)

Y_Train[:, 0] = potential[t_n:]
Y_Test[:, 0] = potential[:t_n]

# Print the shape of the Loaded Velocity vector
print('The Loaded Dataset  has the following shape: ', 'X Train: ', X_Train.shape, 'X Test: ', X_Test.shape,
      'Y Train: ', Y_Train.shape, 'Y Test', Y_Test.shape)

if torch.cuda.is_available():
    print('Using CUDA')
    model = MultiSimpleNet(data_channels).cuda().type(torch.float32)
    # model = Net().cuda()
else:
    model = MultiSimpleNet(data_channels)
    # model1 = MultiSimpleNet(data_channels)


# summary(model,(64,1,64,64))

# Initialize network weights with Kaiming normal method (a.k.a MSRA)
def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.kaiming_uniform_(m.weight)


model.apply(init_weights)

init_lr = optimization_params['initial_lr']

# Optimizer and initial learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)

# This is a learning rate scheduler, it will automatically reduce the learning rate if a plateau is reached
from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=optimization_params['scheduler_factor'],
                              patience=optimization_params['scheduler_patience'],
                              verbose=True, threshold=optimization_params['scheduler_threshold'], threshold_mode='rel')

for param_group in optimizer.param_groups:
    print('LR', float(param_group['lr']))

# Set up loss function weightings
inside_weight = option_params['inside_weight']
bound_weight = option_params['bound_weight']
elec_weight = option_params['elec_weight']
lapl_weight = option_params['lapl_weight']

# Create State, name and path
state_name = option_params['state_name']

state_dir = folder_state + '/model_saves'
state_path = os.path.join(state_dir, state_name)

state = {}
time_vec = np.zeros(8)

# create the logs
state['train_loss_plot'] = np.array([])
state['val_loss_plot'] = np.array([])
state['train_MSE_plot'] = np.array([])
state['val_MSE_plot'] = np.array([])
state['train_elec_plot'] = np.array([])
state['val_elec_plot'] = np.array([])
state['train_lapl_plot'] = np.array([])
state['val_lapl_plot'] = np.array([])

# initialise arrays for recording results
train_loss_plot = np.array([])
val_loss_plot = np.array([])
train_MSE_plot = np.array([])
val_MSE_plot = np.array([])
train_elec_plot = np.array([])
val_elec_plot = np.array([])
train_lapl_plot = np.array([])
val_lapl_plot = np.array([])

epochs = option_params['n_epochs']

# Use alpha = 0.5, 1.0, 1.5 for training, keep alpha = 0 for validation
val_no = data_params['val_no']  # number in the dataset from alpha = 0


class RandomRotation(object):
    def __init__(self):
        pass

    def __call__(self, img, d4dice):

        # d4dice = 4*np.random.rand()
        # print("d4dice ", d4dice)
        # print("img ", img.shape)
        # ndims = self.size
        if d4dice >= 0 and d4dice < 1:
            img = torch.flip(img, [1])
        elif d4dice >= 1 and d4dice < 2:
            img = torch.flip(img, [2])
        elif d4dice >= 2 and d4dice < 3:
            img = torch.flip(img, [1])
            img = torch.flip(img, [2])
        else:
            img = img

        return img


class NewTensorDataset(Dataset):
    """Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, data, target, data_augmentation=None):
        self.data = data
        self.target = target
        self.data_augmentation = data_augmentation

    def __getitem__(self, index):
        data1 = self.data[index]
        target1 = self.target[index]
        if self.data_augmentation is not None:
            d4dice = 4 * np.random.rand()

            data = self.data_augmentation(data1, d4dice)
            target = self.data_augmentation(target1, d4dice)

        return data, target

    def __len__(self):
        return self.data.size(0)


# Shuffle training data before separation
shuffler = np.random.permutation(X_Train.shape[0])
X_Train = X_Train[shuffler]
Y_Train = Y_Train[shuffler]

# create tensors, clean and noisy, and split into train and val
train_targets = torch.from_numpy(np.float32(Y_Train[val_no:]))
val_targets = torch.from_numpy(np.float32(Y_Train[:val_no]))

train_inputs = torch.from_numpy(np.float32(X_Train[val_no:]))
val_inputs = torch.from_numpy(np.float32(X_Train[:val_no]))

augmen = data_params['data_augmentation']

if augmen:
    train_dataset = NewTensorDataset(train_inputs, train_targets, data_augmentation=RandomRotation())
else:
    train_dataset = torch.utils.data.TensorDataset(train_inputs, train_targets)

val_dataset = torch.utils.data.TensorDataset(val_inputs, val_targets)
batch_sz = option_params['batch_size']

# Into a dataloader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_sz, num_workers=4,
                                           shuffle=data_params['shuffle_train_set'])
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_sz, num_workers=4,
                                         shuffle=data_params['shuffle_val_set'])

# Define loss function
criterion = torch.nn.MSELoss()

for epoch in range(1, epochs + 1):
    # ---------------------------------------------------------------------
    # Train on train set, then also test on val set and test set
    # train_loss, train_MSE, train_lapl = train(epoch)
    # val_loss, val_MSE, val_lapl = val()
    train_loss, train_inside, train_bound, train_lapl, train_elec = train(epoch, model, criterion, train_loader, optimizer, scheduler,
                                                                          inside_weight, bound_weight, lapl_weight, elec_weight, potential_max, rhs_max, folder)
    val_loss, val_inside, val_bound, val_lapl, val_elec = validate(epoch, model, criterion, val_loader, inside_weight, bound_weight, lapl_weight,
                                                                   elec_weight, potential_max, rhs_max, folder)

    # Step scheduler, will reduce LR if loss has plateaued
    # scheduler.step(train_loss)

    # Store training loss function, and also raw MSE and lapl
    train_loss_plot = np.append(train_loss_plot, train_loss)
    val_loss_plot = np.append(val_loss_plot, val_loss)
    train_MSE_plot = np.append(train_MSE_plot, train_inside)
    val_MSE_plot = np.append(val_MSE_plot, val_inside)
    train_lapl_plot = np.append(train_lapl_plot, train_lapl)
    val_lapl_plot = np.append(val_lapl_plot, val_lapl)
    train_elec_plot = np.append(train_elec_plot, train_elec)
    val_elec_plot = np.append(val_elec_plot, val_elec)

    state['train_loss_plot'] = np.append(state['train_loss_plot'], train_loss)
    state['val_loss_plot'] = np.append(state['val_loss_plot'], val_loss)
    state['train_MSE_plot'] = np.append(state['train_MSE_plot'], train_inside)
    state['val_MSE_plot'] = np.append(state['val_MSE_plot'], val_inside)
    state['train_lapl_plot'] = np.append(state['train_lapl_plot'], train_lapl)
    state['val_lapl_plot'] = np.append(state['val_lapl_plot'], val_lapl)
    state['train_elec_plot'] = np.append(state['train_elec_plot'], train_elec)
    state['val_elec_plot'] = np.append(state['val_elec_plot'], val_elec)

    state['model_state_dict'] = model.state_dict()
    state['optimizer_state_dict'] = optimizer.state_dict()
    torch.save(state, state_path)

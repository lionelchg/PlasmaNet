########################################################################################################################
#                                                                                                                      #
#                                                  Plots for trainer                                                   #
#                                                                                                                      #
#                                     Guillaume Bogopolsky, CERFACS, 02.04.2020                                        #
#                                                                                                                      #
########################################################################################################################

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')


def plot_batch(output, target, data, epoch, batch_idx):
    """ Matplotlib plots. """
    # Detach tensors and send them to cpu as numpy
    data_np = data.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    output_np = output.detach().cpu().numpy()

    # Lots of plots
    fig, axes = plt.subplots(figsize=(20, 16), nrows=4, ncols=4)
    fig.suptitle(f'Epoch {epoch} batch {batch_idx}', fontsize=16, y=0.95)

    for k in range(4):  # First 4 items of the batch
        tt = axes[k, 0].imshow(data_np[batch_idx + k, 0], origin='lower')
        axes[k, 0].set_title('rhs')
        axes[k, 0].axis('off')
        fig.colorbar(tt, ax=axes[k, 0])

        tt = axes[k, 1].imshow(output_np[batch_idx + k, 0], origin='lower')
        axes[k, 1].set_title('predicted potential')
        axes[k, 1].axis('off')
        fig.colorbar(tt, ax=axes[k, 1])

        tt = axes[k, 2].imshow(target_np[batch_idx + k, 0], origin='lower')
        axes[k, 2].set_title('target potential')
        axes[k, 2].axis('off')
        fig.colorbar(tt, ax=axes[k, 2])

        tt = axes[k, 3].imshow(np.abs(target_np[batch_idx + k, 0] - output_np[batch_idx + k, 0]), origin='lower')
        axes[k, 3].set_title('residual')
        axes[k, 3].axis('off')
        fig.colorbar(tt, ax=axes[k, 3])
    return fig


def plot_distrib(output, target, epoch, batch_idx):
    """ Plot distribution (ie. flattened target vs. output) """
    output_np = output[batch_idx, 0].detach().flatten().cpu().numpy()
    target_np = target[batch_idx, 0].detach().flatten().cpu().numpy()

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_title(f'Output distribution at epoch {epoch} for batch {batch_idx}')

    ax.plot(target_np, output_np, '.', markersize=1.5)
    ax.set_xlabel('target')
    ax.set_ylabel('output')

    return fig

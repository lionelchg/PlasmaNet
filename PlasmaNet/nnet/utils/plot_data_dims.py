########################################################################################################################
#                                                                                                                      #
#                                                Disastrous plot function                                              #
#        It can be useful to visualize the data loader inputs for the model tackling all the problems at once          #
#                                                                                                                      #
#                                            Ekhi Ajuria, CERFACS, 03.04.2020                                          #
#                                                                                                                      #
########################################################################################################################

import matplotlib.pyplot as plt


def plot_dataloader_complete(d_1, d_2, d_3, d_4, potential, physical_rhs, x_tensor, y_tensor, folder):
    """
    Takes as inputs:
    - 4 boundary channels, which follow:
                     4
                 - - - - -
                |         |
              1 |         | 3
                |         |
                 - - - - -
                     2
    - Potential field
    - rhs
    - x and y indexes

    """
    # Lots of plots
    fig, axes = plt.subplots(figsize=(20, 14), nrows=3, ncols=4)
    fig.suptitle(' BC and data loading ')

    tt = axes[0, 0].imshow(d_1[0, 0].data.cpu(), origin='lower')
    axes[0, 0].set_title('d1')
    axes[0, 0].axis('off')
    fig.colorbar(tt, ax=axes[0, 0])

    tt = axes[0, 1].imshow(d_2[0, 0].data.cpu(), origin='lower')
    axes[0, 1].set_title('d2')
    axes[0, 1].axis('off')
    fig.colorbar(tt, ax=axes[0, 1])

    tt = axes[0, 2].imshow(d_3[0, 0].data.cpu(), origin='lower')
    axes[0, 2].set_title('d3')
    axes[0, 2].axis('off')
    fig.colorbar(tt, ax=axes[0, 2])

    tt = axes[0, 3].imshow(d_4[0, 0].data.cpu(), origin='lower')
    axes[0, 3].set_title('d4')
    axes[0, 3].axis('off')
    fig.colorbar(tt, ax=axes[0, 3])

    tt = axes[1, 0].plot(potential[0, 0, :, 0].data.cpu())
    axes[1, 0].set_title('d1')

    tt = axes[1, 1].plot(potential[0, 0, 0, :].data.cpu())
    axes[1, 1].set_title('d2')

    tt = axes[1, 2].plot(potential[0, 0, :, -1].data.cpu())
    axes[1, 2].set_title('d3')

    tt = axes[1, 3].plot(potential[0, 0, -1, :].data.cpu())
    axes[1, 3].set_title('d4')

    tt = axes[2, 0].imshow(x_tensor[0, 0].data.cpu(), origin='lower')
    axes[2, 0].set_title('x tensor')
    axes[2, 0].axis('off')
    fig.colorbar(tt, ax=axes[2, 0])

    tt = axes[2, 1].imshow(y_tensor[0, 0].data.cpu(), origin='lower')
    axes[2, 1].set_title('y tensor')
    axes[2, 1].axis('off')
    fig.colorbar(tt, ax=axes[2, 1])

    tt = axes[2, 2].imshow(physical_rhs[0, 0].data.cpu(), origin='lower')
    axes[2, 2].set_title('rhs')
    axes[2, 2].axis('off')
    fig.colorbar(tt, ax=axes[2, 2])

    tt = axes[2, 3].imshow(potential[0, 0].data.cpu(), origin='lower')
    axes[2, 3].set_title('Potential')
    axes[2, 3].axis('off')
    fig.colorbar(tt, ax=axes[2, 3])

    # print("Saving this image in folder: ", folder)
    fig.savefig(folder / 'Data_loader_inputs.png', dpi=150, bbox_inches='tight')

########################################################################################################################
#                                                                                                                      #
#                                         Define divergence function and mask                                          #
#                                                                                                                      #
#                        Ekhi Ajuria, Guillaume Bogopolsky (transcription) CERFACS, 27.02.2020                         #
#                                                                                                                      #
########################################################################################################################


import torch


def divergence_mask(field, dx, dy):
    """
    Calculates the electric field divergence (with boundary condition modifications). This is essentially a replica
    of makeRhs in Manta and FluidNet.

    Parameters
    ----------
    field : torch.Tensor
        Input 2D field: tensor of size (batch, 2, H, W)
    dx, dy : float

    Returns
    -------
    Tensor
        Output divergence (scalar field)
    """

    # Check sizes
    divergence = torch.zeros_like(field[:, 0]).type(field.type()).unsqueeze(1)
    assert field.dim() == 4 and divergence.dim() == 4, 'Dimension mismatch'
    assert field.size(1) == 2, 'E is not 2D'
    batch_size = field.size(0)
    h, w = field.size(2), field.size(3)

    assert field.is_contiguous() and divergence.is_contiguous(), 'Input is not contiguous'

    # Define work vectors
    field_ij_p = torch.zeros_like(field)  # Field in ij
    field_ij_n = torch.zeros_like(field)
    field_ij = field.clone()

    # Create translated work tensors
    field_ij_p[:, 0, :, 1:] = field_ij[:, 0, :, :-1]  # Copy field_x with padding 1 along y
    field_ij_p[:, 0, :, 0] = field_ij[:, 0, :, 0]  # Copy 1st value (TODO duplicated on 1st and 2nd y positions?)
    field_ij_p[:, 1, 1:, :] = field_ij[:, 1, :-1, :]  # Copy field_y with padding 1 along x
    field_ij_p[:, 1, 0, :] = field_ij[:, 1, 0, :]  # Copy 1st value (TODO duplicated on 1st and 2nd x positions?)

    field_ij_n[:, 0, :, :-1] = field_ij[:, 0, :, 1:]  # Copy field_x with -1 padding along y
    field_ij_n[:, 0, :, -1] = field_ij[:, 0, :, -1]  # Copy last value (TODO duplicated on nth and n-1 positions?)
    field_ij_n[:, 1, :-1, :] = field_ij[:, 1, 1:, :]
    field_ij_n[:, 1, -1, :] = field_ij[:, 1, -1, :]

    # Finite element derivative
    div_center = (field_ij_n.select(1, 0) - field_ij_p.select(1, 0)) / (2 * dx) + \
                 (field_ij_n.select(1, 1) - field_ij_p.select(1, 1)) / (2 * dy)
    div_fwd = (field_ij_n.select(1, 0) - field_ij.select(1, 0)) / dx + \
              (field_ij_n.select(1, 1) - field_ij.select(1, 1)) / dy
    div_back = (field_ij.select(1, 0) - field_ij_p.select(1, 0)) / dx + \
               (field_ij.select(1, 1) - field_ij_p.select(1, 1)) / dy

    # Indexes and masks
    indexes_x = ((torch.arange(w)).view(1, w)).expand(h, w).unsqueeze(0)
    indexes_y = ((torch.arange(h)).view(h, 1)).expand(h, w).unsqueeze(0)

    indexes = torch.zeros((2, h, w))
    indexes[0], indexes[1] = indexes_x, indexes_y

    indexes.unsqueeze(0).expand(batch_size, 2, h, w)

    zero = torch.zeros_like(indexes_x).type(torch.int16)
    ones = torch.ones_like(indexes_x).type(torch.int16)

    # center_flag[0] = torch.where(indexes_x[0] > 0 and indexes_y[0] > 0 and indexes_x[0] < (w - 1)
    # and indexes_y[0] < ones * (h - 1), ones[0], zero[0])
    center_flag = torch.where(indexes_x[0] < (w - 1), ones[0], zero[0])
    center_flag = torch.where(indexes_x[0] > 0, center_flag, zero[0])
    center_flag = torch.where(indexes_y[0] > 0, center_flag, zero[0])
    center_flag = torch.where(indexes_y[0] < (h - 1), center_flag, zero[0])

    fwd_flag = torch.where(indexes_x[0] < 1, ones[0], zero[0])
    fwd_flag = torch.where(indexes_y[0] < 1, ones[0], fwd_flag)

    bwd_flag = torch.where(indexes_x[0] >= (w - 1), ones[0], zero[0])
    bwd_flag = torch.where(indexes_y[0] >= (h - 1), ones[0], bwd_flag)

    # Send flags tensor to GPU
    center_flag, fwd_flag, bwd_flag = center_flag.cuda(), fwd_flag.cuda(), bwd_flag.cuda()

    # Compute gradient
    div = div_center.masked_fill(center_flag.ne(1), 0) + div_fwd.masked_fill(fwd_flag.ne(1), 0) + \
          div_back.masked_fill(bwd_flag.ne(1), 0)

    div[:, -1, 0] = (field_ij_n[:, 0, -1, 0] - field_ij[:, 0, -1, 0]) / dx + \
                    (field_ij[:, 1, -1, 0] - field_ij_p[:, 1, -1, 0]) / dy
    div[:, 0, -1] = (field_ij[:, 0, 0, -1] - field_ij_p[:, 0, 0, -1]) / dx + \
                    (field_ij_n[:, 1, 0, -1] - field_ij[:, 1, 0, -1]) / dy

    assert field_ij_p.dim() == 4, 'Dimension mismatch in field_ij_p'
    assert field_ij_n.dim() == 4, 'Dimension mismatch in field_ij_n'
    assert field_ij.dim() == 4, 'Dimension mismatch in field_ij'

    divergence[:, :, 0:h, 0:w] = div.view(batch_size, 1, h, w)

    divergence[:, 0, :, -1] = torch.zeros((batch_size, h))
    divergence[:, 0, :, 0] = torch.zeros((batch_size, h))
    divergence[:, :, :, w - 2] = divergence[:, :, :, w - 3]

    return divergence

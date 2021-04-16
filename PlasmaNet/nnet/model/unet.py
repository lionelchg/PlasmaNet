import torch
import torch.nn as nn

from ..base import BaseModel

# Create the model

class _ConvBlock(nn.Module):
    """
    General convolution block for UNet. Depending on the location of the block
    in the architecture, the block can begin with a MaxPool2d (for bottom)
    or end with an UpSample or deconvolution layer (for up)
    """
    def __init__(self, fmaps, block_type, up_type='upsample', up_arg=None):
        super(_ConvBlock, self).__init__()
        layers = list()
        # Apply pooling on down and bottom blocks
        if block_type == 'down' or block_type == 'bottom':
            layers.append(nn.MaxPool2d(2))

        # Append all the specified layers
        for i in range(len(fmaps) - 1):
            layers.append(nn.Conv2d(fmaps[i], fmaps[i + 1], kernel_size=3, padding=1))
            # No ReLu at the very last layer
            if i != len(fmaps) - 2 or block_type != 'out':
                layers.append(nn.ReLU())

        # Apply either Upsample or deconvolution
        if block_type == 'up' or block_type == 'bottom':
            if up_type == 'upsample':
                output_size = up_arg
                layers.append(nn.Upsample(output_size))
            elif up_type == 'deconvolution':
                total_padding = up_arg
                padding = up_arg // 2
                output_padding = up_arg % 2
                # ConvTranspose2d has a specific way of treating the output shape
                # see pytorch documentation
                layers.append(nn.ConvTranspose2d(fmaps[-1], fmaps[-1], 2, stride=2,
                        dilation=1+padding, output_padding=output_padding))
        
        # Build the sequence of layers
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)

class UNet(BaseModel):
    """
    General UNet. All the layers are specified in the config file. Three different options are possible
    when going up the U: upsample, deconvolution or interpolation. Only interpolation
    allows the network to work on different resolutions
    """
    def __init__(self, in_fmaps, down_blocks, bottom_fmaps, up_blocks, out_fmaps, 
                    input_res=None, up_type='upsample'):
        super(UNet, self).__init__()
        n_scales = 2 + len(down_blocks)
        self.up_type = up_type

        if self.up_type == 'upsample':
            # For upsample the list of resolution is needed when 
            # the number of points is not a power of 2
            list_res = [int(input_res / 2**i) for i in range(n_scales - 1)]
            list_args = list_res
        elif self.up_type == 'deconvolution':
            # For deconvolution the difference between the real size and the one after upconv without padding
            list_res = [int(input_res / 2**i) for i in range(n_scales)]
            list_args = [list_res[i] - 2 * list_res[i + 1] for i in range(n_scales - 1)]

        # Entry layer
        self.ConvsDown = nn.ModuleList()
        self.ConvsDown.append(_ConvBlock(in_fmaps, ''))

        # Intermediate down layers (with MaxPool at the beginning)
        for down_fmaps in down_blocks:
            self.ConvsDown.append(_ConvBlock(down_fmaps, 'down'))

        # Bottom layer (MaxPool at the beginning and Upsample/Deconv at the end)
        self.ConvBottom = _ConvBlock(bottom_fmaps, 'bottom', self.up_type, list_args.pop())

        # Intemediate layers up (UpSample/Deconv at the end)
        self.ConvsUp = nn.ModuleList()
        for up_fmaps in up_blocks:
            self.ConvsUp.append(_ConvBlock(up_fmaps, 'up', self.up_type, list_args.pop()))
        
        # Out layer
        self.ConvsUp.append(_ConvBlock(out_fmaps, 'out'))

    def forward(self, x):
        # List of the temporary x that are used for linking with the up branch
        inputs_down = list()

        # Apply the down loop
        for ConvDown in self.ConvsDown:
            x = ConvDown(x)
            inputs_down.append(x)
        
        # Bottom part of the U
        x = self.ConvBottom(x)
        
        # Apply the up loop
        for ConvUp in self.ConvsUp:
            input_tmp = inputs_down.pop()
            x = ConvUp(torch.cat((x, input_tmp), dim=1))
                
        return x
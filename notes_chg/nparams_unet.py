class _ConvBlockIn:
    """
    Two Conv2d layers, all with kernel_size 3 and padding of 1 (padding ensures output size is same as input size)
    ReLU after first two Conv2d layers
    """
    def __init__(self, in_channels, mid_channels, out_channels):
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
    
    def nparams(self):
        return 9 * (self.in_channels * self.mid_channels 
                        + self.mid_channels * self.out_channels)

class _ConvBlockDown:
    """
    Maxpooling to reduce the size
    Two Conv2d layers, all with kernel_size 3 and padding of 1 (padding ensures output size is same as input size)
    ReLU after first two Conv2d layers
    """
    def __init__(self, in_channels, mid_channels, out_channels):
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels

    def nparams(self):
        return 9 * (self.in_channels * self.mid_channels 
                        + self.mid_channels * self.out_channels)

class _ConvBlockUp:
    """
    Maxpooling to reduce the size
    Two Conv2d layers, all with kernel_size 3 and padding of 1 (padding ensures output size is same as input size)
    ReLU after first two Conv2d layers
    """
    def __init__(self, in_channels, mid_channels, out_channels):
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels

    def nparams(self):
        return 9 * (self.in_channels * self.mid_channels 
                        + self.mid_channels * self.out_channels)

class _ConvBlockOut:
    """
    Maxpooling to reduce the size
    Two Conv2d layers, all with kernel_size 3 and padding of 1 (padding ensures output size is same as input size)
    ReLU after first two Conv2d layers
    """
    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels

    def nparams(self):
        return self.in_channels * self.out_channels

class UNet3:
    def __init__(self):
        self.conv = []
        self.conv.append(_ConvBlockIn(1, 64, 64))
        self.conv.append(_ConvBlockDown(64, 64, 64))
        self.conv.append(_ConvBlockDown(64, 76, 64))
        self.conv.append(_ConvBlockUp(128, 64, 64))
        self.conv.append(_ConvBlockUp(128, 64, 64))
        self.conv.append(_ConvBlockOut(64, 1))
    
    def compute_params(self):
        npar = 0
        for block in self.conv:
            npar += block.nparams()
        return npar

class UNet4:
    def __init__(self):
        self.conv = []
        self.conv.append(_ConvBlockIn(1, 48, 48))
        self.conv.append(_ConvBlockDown(48, 48, 48))
        self.conv.append(_ConvBlockDown(48, 48, 48))
        self.conv.append(_ConvBlockDown(48, 64, 64))
        self.conv.append(_ConvBlockUp(112, 64, 64))
        self.conv.append(_ConvBlockUp(112, 60, 48))
        self.conv.append(_ConvBlockUp(96, 48, 48))
        self.conv.append(_ConvBlockOut(48, 1))
    
    def compute_params(self):
        npar = 0
        for block in self.conv:
            npar += block.nparams()
        return npar

class UNet5:
    def __init__(self):
        self.conv = []
        self.conv.append(_ConvBlockIn(1, 32, 32))
        self.conv.append(_ConvBlockDown(32, 32, 32))
        self.conv.append(_ConvBlockDown(32, 32, 32))
        self.conv.append(_ConvBlockDown(32, 48, 62))
        self.conv.append(_ConvBlockDown(62, 60, 62))
        self.conv.append(_ConvBlockUp(124, 64, 64))
        self.conv.append(_ConvBlockUp(96, 64, 64))
        self.conv.append(_ConvBlockUp(96, 32, 32))
        self.conv.append(_ConvBlockUp(64, 32, 32))
        self.conv.append(_ConvBlockOut(32, 1))
    
    def compute_params(self):
        npar = 0
        for block in self.conv:
            npar += block.nparams()
        return npar

class UNet6:
    def __init__(self):
        self.conv = []
        self.conv.append(_ConvBlockIn(1, 32, 32))
        self.conv.append(_ConvBlockDown(32, 32, 32))
        self.conv.append(_ConvBlockDown(32, 32, 32))
        self.conv.append(_ConvBlockDown(32, 48, 48))
        self.conv.append(_ConvBlockDown(48, 48, 48))
        self.conv.append(_ConvBlockDown(48, 48, 48))
        self.conv.append(_ConvBlockUp(96, 60, 48))
        self.conv.append(_ConvBlockUp(96, 48, 48))
        self.conv.append(_ConvBlockUp(96, 48, 48))
        self.conv.append(_ConvBlockUp(80, 32, 32))
        self.conv.append(_ConvBlockUp(64, 32, 32))
        self.conv.append(_ConvBlockOut(32, 1))
    
    def compute_params(self):
        npar = 0
        for block in self.conv:
            npar += block.nparams()
        return npar

if __name__ == '__main__':
    net3 = UNet3()
    print(f'Unet3 nparams = {net3.compute_params():d}')
    net4 = UNet4()
    print(f'Unet4 nparams = {net4.compute_params():d}')
    net5 = UNet5()
    print(f'Unet5 nparams = {net5.compute_params():d}')
    net6 = UNet6()
    print(f'Unet6 nparams = {net6.compute_params():d}')
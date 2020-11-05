import numpy as np

class Grid:
    def __init__(self, config):
        # Mesh properties
        self.ndim = 2
        self.nnx, self.nny = config['mesh']['nnx'], config['mesh']['nny']
        self.ncx, self.ncy = self.nnx - 1, self.nny - 1  # Number of cells
        self.xmin, self.xmax = config['mesh']['xmin'], config['mesh']['xmax']
        self.ymin, self.ymax = config['mesh']['ymin'], config['mesh']['ymax']
        self.Lx, self.Ly = self.xmax - self.xmin, self.ymax - self.ymin
        self.dx = (self.xmax - self.xmin) / self.ncx
        self.dy = (self.ymax - self.ymin) / self.ncy
        self.x = np.linspace(self.xmin, self.xmax, self.nnx)
        self.y = np.linspace(self.ymin, self.ymax, self.nny)
        # Grid construction
        self.X, self.Y = np.meshgrid(self.x, self.y)
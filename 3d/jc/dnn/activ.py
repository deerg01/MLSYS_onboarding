import numpy as np
from .nn import nnModule


class Sigmoid(nnModule):
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    def __init__(self, name = ''):
        super().__init__('Sigmoid ' + name)
        self.y = None
    
    def forward(self, x):
        self.y = self.sigmoid(x)
        return self.y
    
    def backward(self, d_y): 
        return d_y * (1.0 - self.y) * self.y  # y = sigmoid(x), y' = (1-y)y


class ReLU(nnModule):
    def __init__(self, name=''):
        super().__init__('ReLU ' + name)
        self.mask = None
    
    def forward(self, x):
        self.mask = (x<0)
        ret = x.copy()
        ret[self.mask] = 0
        return ret
    
    def backward(self, d_y):
        d_y[self.mask] = 0
        d_x = d_y 
        return d_x
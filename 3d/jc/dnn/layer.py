import numpy as np
from .optim import Updatable
from .nn import nnModule

class Linear(Updatable):
    def __init__(self, input_size, output_size, name = ''):
        stddev = np.sqrt(2/input_size)
        super().__init__(f'Linear ({str(input_size)} -> {str(output_size)}) {name}', 
                         input_size, 
                         output_size,
                         weight = stddev * np.random.randn(input_size, output_size), 
                         bias = stddev * np.zeros(output_size))   
    
    def forward(self, x):
        self.x = x
        self.y = np.dot(x, self.weight) + self.bias
        return self.y
    
    def backward(self, d_y):
        self.d_W = np.dot(self.x.T, d_y)
        self.d_b = np.sum(d_y, axis=0)
        d_x = np.dot(d_y, self.weight.T)        
        
        return d_x


class LayerNorm():
    pass
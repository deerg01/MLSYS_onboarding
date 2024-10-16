from .nn import nnModule
import numpy as np


class Updatable(nnModule):
    def __init__(self, name: str, input_size, output_size, weight = None, bias = None):
        super().__init__(name, input_size, output_size)
        self.weight = weight
        self.bias = bias
        self.x = None
        self.y = None
        self.d_W = None
        self.d_b = None


class Optimizer:
    def __init__(self, parameters, learning_rate=1e-2) -> None:
        self.parameters = parameters
        self.learning_rate = learning_rate
        
    def step(self):
        pass
    
    def zero_grad(self):
        for module in self.parameters:
            if isinstance(module, Updatable):
                module.d_W = None
                module.d_b = None

    def step(self):
        raise NotImplementedError(self.name + ' Optimizer does not have step method')
    
    

class SGD(Optimizer):
    def __init__(self, parameters, learning_rate=0.01):
        super().__init__(parameters, learning_rate)
    
    def step(self):
        for module in self.parameters:
            if isinstance(module, Updatable):
                module.weight -= self.learning_rate * module.d_W
                module.bias -= self.learning_rate * module.d_b
                
                
class MomentumSGD(Optimizer):
    def __init__(self, parameters, learning_rate=0.01, momentum = 0.5):
        if momentum < 0 or momentum >= 1:
            raise ValueError('momentum should be in [0, 1)')
        super().__init__(parameters, learning_rate)
        self.momentum = momentum
        for module in self.parameters:
            if isinstance(module, Updatable):
                module.old_d_W = 0
                module.old_d_b = 0

    def step(self):
        for module in self.parameters:
            if isinstance(module, Updatable):
                module.old_d_W = self.momentum * module.old_d_W - self.learning_rate * module.d_W
                module.old_d_b = self.momentum * module.old_d_b - self.learning_rate * module.d_b
                
                module.weight += module.old_d_W
                module.bias += module.old_d_b
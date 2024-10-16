import numpy as np


class nnModule:
    def __init__(self, name: str, input_size = None, output_size = None):
        self.name = name
        self.input_size = input_size
        self.output_size = output_size
    def forward(self):
        raise NotImplementedError(self.name + ' Module does not have forward function')
    
    def backward(self):
        raise NotImplementedError(self.name + ' Module does not have backward function')
    
    def __str__(self) -> str:
        return self.name
        

class nn(nnModule):
    def __init__(self, name):
        super().__init__(name, None, None)
        self.moduleList = []
        self.last_size = None
        
    def addModule(self, module: nnModule):
        module.name
        module.output_size
        
        assert not ((module.input_size is None) ^ (module.output_size is None)) 
        if self.input_size is None:
            self.input_size = module.input_size
            self.output_size = module.output_size
        elif module.output_size is not None:
            if module.input_size is not None and self.output_size != module.input_size:
                raise ValueError('Layer input/output size does not match')
            self.output_size = module.output_size
        self.moduleList.append(module)        
        
    def forward(self, x: np.ndarray):
        for (index, module) in enumerate(self.moduleList):
            x = module.forward(x)
        return x
    
    def backward(self, d_y):
        for module in reversed(self.moduleList):
            d_y = module.backward(d_y)
    
    def __str__(self) -> str:
        return '\n'.join([self.name] + list(map(str,self.moduleList)))

    def parameters(self):
        return self.moduleList
    
            
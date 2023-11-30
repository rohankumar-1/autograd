
from autograd.engine import Tensor
import numpy as np

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Layer(Module):

    # each Layer is comprised of an activation function, and the weight matrix / bias vector that comprise it
    def __init__(self, n_inputs, n_outputs, func=None):
        self.W = Tensor(np.random.randn(n_outputs, n_inputs), label='W')
        self.b = Tensor(np.zeros((n_outputs, 1)), label='B')
        self._act = func

    def __repr__(self):
        lin = self._act if self._act!=None else 'linear'
        return f"Layer: {lin}, {self.W.data.shape[0]} neurons"

    def __call__(self, X):
        prod = self.W @ X + self.b # usual y = wx + b
        
        # apply activation and return
        if self._act == 'tanh':
            return prod.tanh()
        elif self._act == 'relu':
            return prod.relu()
        elif self._act == 'leaky':
            return prod.leaky_relu()
        elif self._act == 'crelu':
            return prod.crelu()
        else:
            return prod

    def parameters(self):
        return [self.W, self.b]

    
class MLP(Module):

    def __init__(self, nin, nouts, acts=[]):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], func=acts[i]) for i in range(len(nouts))]

    def forward(self, x):
        x = Tensor(x) if not isinstance(x, Tensor) else x
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        layers_str = '\n - '.join([l.__repr__() for l in self.layers])
        return f"MLP of {len(self.layers)}: \n - In: {layers_str}"
        
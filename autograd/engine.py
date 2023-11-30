import numpy as np

# class Tensor will be a class that holds a matrix of values, and will be the primitive of this neural net. Note that a Tensor that 
# is given a scalar value for X will essentially be the same as the Value class of Karpathy's micrograd

class Tensor:
    def __init__(self, X, _children='' ,_op='', label='', requires_grad=True):
        # add data, this is pretty straight forward, data refers to any intermediate of a neural network (weight, input, func output)
        if not isinstance(X, np.ndarray):
            self.data = np.array(X)
        else:
            self.data = X
        self._op = _op
        self.label = label
        self._prev = set(_children)
        self.requires_grad = requires_grad
        self.grad = np.zeros(self.data.shape)   # initially, the gradient is all 0s, and is a matrix same size of data
        self._backward = lambda: None  # this is where our local gradient formula will be kept, as a function

    def __repr__(self):
        # for ease of use, a simple way to print tensors
        return f"{self.data}"

    def __add__(self, other):
        # element-wise addition of matrices/vectors/scalar
        
        if not isinstance(other, Tensor):
            other = Tensor(other) 

        out = Tensor(np.add(np.atleast_2d(self.data.T).T, np.atleast_2d(other.data.T).T), _children=(self, other), _op='+') # this part does forward pass

        # create local gradient formula for addition of two matrices, output is Jacobian 
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(np.multiply(self.data, other.data), _children=(self, other), _op='*')

        def _backward():
            self.grad += other.data @ out.grad
            other.grad += self.data @ out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
            
        out = Tensor(np.matmul(self.data, other.data), _children=(self, other), _op='@')

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward
        
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-1) * other

    def __pow__(self, pow):
        assert isinstance(pow, (int, float))
        out = Tensor(np.power(self.data, pow), (self,), f'**{pow}')

        def _backward():
            self.grad += (pow * (np.power(self.data ,(pow - 1)))) * out.grad
        out._backward = _backward

        return out

    def __truediv__(self, other):
        return self * (other**-1)

    def sum(self):
        out = Tensor(np.sum(self.data), (self,), 'sum')

        def _backward():
            self.grad += np.tensordot(np.ones((self.data.shape[0], self.data.shape[1], 1, 1)), out.grad) 
        out._backward = _backward

        return out

    def relu(self):
        out = Tensor(np.maximum(self.data, 0), _children=(self, ), _op='relu')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        
        return out

    def crelu(self):
        val = np.maximum(self.data,0)
        val = np.minimum(val, 3)
        out = Tensor(val, _children=(self, ), _op='relu')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def leaky_relu(self):
        c = 0.0001
        out = Tensor(np.maximum(self.data, self.data*c), _children=(self, ), _op='leaky relu')

        def _backward():
            g = 1 if out.data > 0 else c
            self.grad += g * out.grad
        out._backward = _backward
        
        return out
    
    def tanh(self):
        out = Tensor(np.tanh(self.data), _children=(self, ), _op='tanh')
       
        def _backward():
            self.grad += (1.-np.tanh(x)**2) @ out.grad
        out._backward = _backward

        return out

    def CrossEntropyLoss(self, y_gt):
        y_gt = other if isinstance(y_gt, Tensor) else Tensor(y_gt)
        
        # used on y_pred, pass in y_gt and then calculate cross entropy loss
        ypred_transpose = self.data.T
        ygts_transpose = y_gt.data.T
        
        epsilon=1e-12
        predictions = np.clip(ypred_transpose, epsilon, 1. - epsilon)
        N = predictions.shape[0]
        ce = -np.sum(ygts_transpose*np.log(predictions+1e-9))/N

        #loss = log_loss(ypred_transpose, ygts_transpose, normalize=True)
        out = Tensor(ce, _children=(self, y_gt), _op='log loss')

        def _backward():
            self.grad += (self.data - y_gt.data) * out.grad
        out._backward = _backward

        return out
    
    def backward(self):
        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.data = np.atleast_2d(self.data)
        self.grad = np.ones(self.data.shape)
        for v in reversed(topo):
            v._backward()

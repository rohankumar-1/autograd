# A Tiny Implementation of Vectorized Automatic Differentiation

This was inspired by Andrej Karpathy's micrograd and PyTorch's autograd. The basic idea is that we can efficiently calculate the chained gradients used for backpropagation using a nice implementation of a class to hold our data. This data class will store a gradient in it, initially 0. We track all operations done on these datapoints, and then using the order of operations we can link up all gradients of these functions, to quickly calculate the gradient of every single datapoint (parameters, input, output) used in the entire model.

What I Added:
- vectorized the layer weights, in order to use the much more efficient dot product
- implemented batch vectorization, inputs can be fed in batch at a time (not just individual)
  - this included interesting matrix math with tensor-matrix multiplication that allows us to avoid large amounts of computation
- added more options for activation (ReLU, clipped ReLU, leaky ReLU, tanh)
- created loss functions (Cross-entropy, MSE)

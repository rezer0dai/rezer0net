import numpy as np

class Layer:
    def __init__(self, in_dim, out_dim, optim):
        self.w = np.random.randn(in_dim, out_dim) * np.sqrt(1. / out_dim)
        self.optim = optim

    def forward(self, data):
        self.x = data
        return self.x.dot(self.w)

    def backward(self, error):
        self.grad = self.x.T.dot(error)
        return error.dot(self.w.T)

    def step(self, batch_size):
        self.w = self.optim.step(self.w, self.grad, batch_size)

class ReLU:
    def forward(self, x):
        self.out = x >= 0.
        return x * self.out

    def backward(self, error):
        return error * self.out

    def step(self, a):
        pass

class Sigmoid:
    def forward(self, z):
        self.out = 1. / (1. + np.exp(-z))
        return self.out

    def backward(self, error):
        return error * self.out * (1. - self.out)

    def step(self, a):
        pass

class SoftMax:
    def forward(self, x):
        return np.exp(x) / np.exp(x).sum(1, keepdims=True)

    def backward(self, error):
        return error * 1.

    def step(self, a):
        pass

from ff_layers import *

class Network:
    def __init__(self, layers, loss):
        self.loss = loss
        self.layers = layers

    def predict(self, x):
        for l in self.layers:
            x = l.forward(x)
        return x

    def fit(self, x, y):
        y_pred = self.predict(x)

        e = self.loss(y, y_pred)
        for l in reversed(self.layers):
            e = l.backward(e)

        for l in self.layers:
            l.step(x.shape[0])

        return y_pred

def loss_func(y, y_pred):
    return (y_pred - y)

from ff_net import *
from ff_layers import *
from optim import *

lr = 1e-3
gamma = 1 - 1e-4
ff = Network([
    Layer(1, 1, SGD(.1, 1.)),#NAG(lr, gamma, .9)),#SGDwMomentum(.1, 1., .9)),#
    ], loss=loss_func)

from utils import *


ff.layers[0].w = ff.layers[0].w * 0. + 5.
print(ff.layers[0].w)
for i in range(40):
    ff.fit(np.ones([1, 1]) * 5, np.ones([1, 1]) * 20)
    print(ff.predict(np.ones([1, 1]) * 5), ff.layers[0].w)



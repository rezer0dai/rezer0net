import random

def problem():
    x = random.sample(range(10),10)
    y = x[3]
    return np.asarray(x).reshape(1, len(x)) * 1., np.array(y, dtype=np.int).reshape(1, 1)

def batch_problem(batch_size):
    b_x, b_y = [], []
    for _ in range(batch_size):
        x, y = problem()
        b_x.append(x)
        b_y.append(y)
    return np.concatenate(b_x), np.concatenate(b_y)

def loss_func(y, y_pred):
    return y_pred - y

from ff_net import *
from ff_layers import *
from optim import *

lr = 1e-2
gamma = 1 - 1e-4
ff = Network([
    Layer(10, 64, NAG(lr, gamma, .9)),
    ReLU(),
    Layer(64, 10, NAG(lr, gamma, .9)),
    SoftMax(),
    ], loss=loss_func)

from utils import *

for i in range(10000):
    x, y = batch_problem(64)
    y = one_hot(y, 10)

    y_pred = ff.fit(x, y)
    if i % 100:
        continue
    print(i, multiclass_loss(y, y_pred))

for _ in range(20):
    x, y = problem()
    print(np.argmax(ff.predict(x)), y.item())

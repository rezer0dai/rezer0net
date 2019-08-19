import numpy as np

def one_hot(data, n_cols):
    data_onehot = np.zeros((data.shape[0], n_cols))
    data_onehot[range(data.shape[0]), data.flatten()] = 1.
    return data_onehot

def multiclass_loss(Y, Y_hat):
    L_sum = np.sum(np.multiply(Y, np.log(Y_hat)))
    m = Y.shape[0]
    L = -(1/m) * L_sum
    return L


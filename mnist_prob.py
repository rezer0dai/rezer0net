import torch
import torchvision.transforms as transforms
from torchvision import datasets

batch_size = 64

dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "./mnist",
        train=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
        download=True,
    ),
    batch_size=batch_size,
    shuffle=True,
)

def loss_func(y, y_pred):
    return y_pred - y

from ff_net import *
from ff_layers import *
from optim import *

lr = 1e-2
gamma = 1 - 1e-4
ff = Network([
    Layer(28 * 28, 64, NAG(lr, gamma, .9)),
    Sigmoid(),
    Layer(64, 10, NAG(lr, gamma, .9)),
    SoftMax(),
    ], loss=loss_func)

from utils import *

for epoch in range(10):
    for i, (imgs, labels) in enumerate(dataloader):
        x = imgs.view(imgs.shape[0], -1).numpy()
        labels = labels.view(x.shape[0], -1).numpy()

        y = one_hot(labels, 10)

        y_pred = ff.fit(x, y)
        if i % 100:
            continue
        print("EPOCH:", epoch, "iteration:", i, "Loss:", multiclass_loss(y, y_pred))

for i in range(len(x)):
    print(ff.predict(x[i].reshape(1, -1)).argmax(), labels[i].reshape(1, -1).item())

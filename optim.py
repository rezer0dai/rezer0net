class SGD:
    def __init__(self, lr, schedule):
        self.lr = lr
        self.schedule = schedule

    def step(self, w, grad, batch_size):
        self.lr = self.lr * self.schedule
        return w - grad * self.lr / batch_size

class SGDwMomentum:
    def __init__(self, lr, schedule, momentum):
        self.schedule = schedule
        self.momentum = momentum

        self.vt = 0
        self.lr = lr

    def step(self, w, grad, batch_size):
        self.lr = self.lr * self.schedule
        self.vt = self.vt * self.momentum + self.lr * grad / batch_size
        return w - self.vt

class NAG:
    def __init__(self, lr, schedule, momentum):
        self.schedule = schedule
        self.momentum = momentum

        self.vt = 0
        self.lr = lr

    def step(self, w, grad, batch_size):
        self.lr = self.lr * self.schedule
        w = w + self.vt * self.lr * self.momentum
        self.vt = self.vt * self.momentum + grad / batch_size
        w = w - self.vt * self.lr
        return w - self.vt * self.lr * self.momentum

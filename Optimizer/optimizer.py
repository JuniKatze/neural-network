import numpy as np

class Parameter:
    def __init__(self, data):
        self.data = np.array(data, dtype=np.float64, copy=True)
        self.grad = None

class Optimizer:
    def __init__(self, params, lr):
        self.params = list(params)
        self.lr = float(lr)
        self.state = {}

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        raise NotImplementedError

# p = p - lr * g
class SGD(Optimizer):
    def __init__(self, params, lr=0.1):
        super().__init__(params, lr)

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            p.data -= self.lr * p.grad

# v = momentum * v + grad
# p = p - lr * v
class Momentum(Optimizer):
    def __init__(self, params, lr=0.1, momentum=0.9):
        super().__init__(params, lr)
        self.momentum = momentum

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue

            key = id(p)
            if key not in self.state:
                self.state[key] = {
                    "v": np.zeros_like(p.data)
                }

            v = self.state[key]["v"]
            v[...] = self.momentum * v + p.grad
            p.data -= self.lr * v

# v = momentum * v + g
# d = g + momentum * v
# p = p - lr * d
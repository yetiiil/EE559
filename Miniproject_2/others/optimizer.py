from torch import empty

class SGD():
    def __init__(self, params, lr, momentum):
        super().__init__()
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.prevs = [empty(x.shape).zero_() for x, _ in self.params]

    def step(self):
        for i, (weight, grad) in enumerate(self.params):
            weight.add_(-grad * self.lr + self.momentum * self.prevs[i])
            self.prevs[i] = -grad * self.lr + self.momentum * self.prevs[i]

    def zero_grad(self):
        for _, grad in self.params:
            grad.zero_()

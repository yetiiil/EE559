class SGD():
    def __init__(self, params, lr):
        super().__init__()
        self.params = params
        self.lr = lr

    def step(self):
        for weight, grad in self.params:
            weight.add_(-grad * self.lr)

    def zero_grad(self):
        for _, grad in self.params:
            grad.zero_()

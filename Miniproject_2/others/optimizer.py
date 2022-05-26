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


class Adam():
    def __init__(self, params, lr= 0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__()
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        #self.beta1prevs = [empty(x.shape).zero_() for x, _ in self.params]
        #self.beta2prevs = [empty(x.shape).zero_() for x, _ in self.params]
        self.epsilon = epsilon
        self.theta_0 = 0	
        self.m_t = [empty(x.shape).zero_() for x, _ in self.params]
        self.v_t = [empty(x.shape).zero_() for x, _ in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, (weight, grad) in enumerate(self.params):
            self.m_t[i] = self.beta1 * self.m_t[i] + (1-self.beta1) * grad
            self.v_t[i] = self.beta2 * self.v_t[i] + (1-self.beta2) * (grad * grad)
            #self.beta1prevs[i] = -grad * self.lr + self.momentum * self.prevs[i]
            #self.beta2prevs[i] = -grad * self.lr + self.momentum * self.prevs[i]
            m_hat = self.m_t[i] / (1 - (self.beta1**self.t))
            v_hat = self.v_t[i] / (1 - (self.beta1**self.t))
            weight.add_(-self.lr * m_hat/(v_hat.sqrt() + self.epsilon))

    def zero_grad(self):
        for _, grad in self.params:
            grad.zero_()
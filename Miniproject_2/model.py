from numpy import save
from .others.module import MSE, Sequential, Conv2d, ReLU, TransposeConv2d, Sigmoid
from .others.optimizer import SGD
import torch
from torch import nn


class Model():
    def __init__(self) -> None :
    ## instantiate model + optimizer + loss function + any other stuff you need
        self.net = Sequential()
        self.net.add_module(Conv2d(3, 48, 3, stride=2))
        self.net.add_module(ReLU())
        self.net.add_module(Conv2d(48, 48, 3, stride=2))
        self.net.add_module(ReLU())
        self.net.add_module(TransposeConv2d(48, 48, 3, stride=2))
        self.net.add_module(ReLU())
        self.net.add_module(TransposeConv2d(48, 3, 3, stride=2))
        self.net.add_module(Sigmoid())

        for m in self.net.modules:
            if isinstance(m, Conv2d) or isinstance(m, TransposeConv2d):
                m.weight.normal_()
                m.bias.zero_()

        self.optimizer = SGD(self.net.param(), lr=1e-3)
        self.criterion = MSE()
       # self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        torch.set_grad_enabled(False)

    def save_model(self, path) -> None :
        torch.save(self.net.state_dict(), path)

    def load_pretrained_model(self, path) -> None:
        self.net.load_state_dict(torch.load(path))


    def train(self, train_input, train_target, num_epochs) -> None:
    #:train_input: tensor of size (N, C, H, W) containing a noisy version of the images. same images, which only differs from the input by their noise.
    #:train_target: tensor of size (N, C, H, W) containing another noisy version of the
       # train_input, train_target = train_input.to(self.device).type(torch.float), train_target.to(self.device).type(torch.float)
        model = self.net
        criterion = self.criterion
        optimizer = self.optimizer
        mini_batch_size=100

        for e in range(num_epochs):
            acc_loss = 0
            for b in range(0, train_input.size(0), mini_batch_size):
                optimizer.zero_grad()

                output = model.forward(train_input.narrow(0, b, mini_batch_size))
                loss = criterion.forward(output/255, train_target.narrow(0, b, mini_batch_size)/255)
                acc_loss = acc_loss + loss.item()
                
                top_grad = criterion.backward()
                model.backward(top_grad)

                optimizer.step()
            print(e, acc_loss)

        pass

    def predict(self, test_input) -> torch.Tensor:
    #:test ̇input: tensor of size (N1, C, H, W) that has to be denoised by the trained or the loaded network.
    # #: returns a tensor of the size (N1, C, H, W)
        return self.net(test_input)

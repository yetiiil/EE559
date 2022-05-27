from .others.module import MSE, Sequential, Conv2d, ReLU, Upsampling, Sigmoid
from .others.optimizer import SGD, Adam
from pathlib import Path
import pickle
import torch

class Model():
    def __init__(self) -> None :
    ## instantiate model + optimizer + loss function + any other stuff you need
        self.net = Sequential()
        self.net.add_module("conv1", Conv2d(3, 48, 2, stride=2))
        self.net.add_module("relu1", ReLU())
        self.net.add_module("conv2", Conv2d(48, 48, 2, stride=2))
        self.net.add_module("relu2", ReLU())
        self.net.add_module("trans1", Upsampling(48, 48, 2, stride=2))
        self.net.add_module("relu3", ReLU())
        self.net.add_module("trans2", Upsampling(48, 3, 2, stride=2))
        self.net.add_module("sig", Sigmoid())

        self.optimizer = Adam(self.net.param())
        self.criterion = MSE()
        #self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        torch.set_grad_enabled(False)

    def save_model(self, model_path = 'bestmodel.pth') -> None :
        model_path=Path(__file__).parent /"bestmodel.pth"
        with open(model_path, 'wb') as f:
            pickle.dump(self.net, f)

    def load_pretrained_model(self) -> None:
        model_path=Path(__file__).parent /"bestmodel.pth"
        with open(model_path, 'rb') as f:
            self.net = pickle.load(f)
            self.optimizer = Adam(self.net.param())


    def train(self, train_input, train_target, num_epochs) -> None:
    #:train_input: tensor of size (N, C, H, W) containing a noisy version of the images. same images, which only differs from the input by their noise.
    #:train_target: tensor of size (N, C, H, W) containing another noisy version of the
        train_input, train_target = train_input.type(torch.float)/255.0, train_target.type(torch.float)/255.0
        model = self.net
        criterion = self.criterion
        optimizer = self.optimizer
        mini_batch_size=100

        for epoch in range(num_epochs):
            acc_loss = 0
            for b in range(0, train_input.size(0), mini_batch_size):
                optimizer.zero_grad()

                output = model.forward(train_input.narrow(0, b, mini_batch_size))
                loss = criterion.forward(output, train_target.narrow(0, b, mini_batch_size))
                acc_loss = acc_loss + loss.item()
                
                top_grad = criterion.backward()
                model.backward(top_grad)

                optimizer.step()
            
    def predict(self, test_input) -> torch.Tensor:
    #:test ̇input: tensor of size (N1, C, H, W) that has to be denoised by the trained or the loaded network.
    # #: returns a tensor of the size (N1, C, H, W)
        return (self.net.forward(test_input.type(torch.float)/255.0) * 255)

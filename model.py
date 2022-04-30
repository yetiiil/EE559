import torch

from torch import optim
from torch import Tensor
from torch import nn
from torch.nn import functional as F

class Model():
    def  __init__(self):
    ## instantiate model + optimizer + loss function + any other stuff you need
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(256, 200)
        self.fc2 = nn.Linear(200, 10)

    def load_pretrained_model(self) -> None:
        
        state_dict = torch.load('bestmodel.pth')
        self.model.load_state_dict(state_dict)

    def train(self, train_input, train_target) -> None:
    #:train_input: tensor of size (N, C, H, W) containing a noisy version of the images. same images, which only differs from the input by their noise.
    #:train_target: tensor of size (N, C, H, W) containing another noisy version of the
        noisy_imgs_1 , noisy_imgs_2 = torch.load('./data/train_data.pkl')
        

    def predict(self, test_input) -> torch.Tensor:
    #:test ̇input: tensor of size (N1, C, H, W) that has to be denoised by the trained or the loaded network.
    # #: returns a tensor of the size (N1, C, H, W)
        pass

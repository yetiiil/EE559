import torch
import math
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torch import optim
import torchvision.transforms as T
from torch import Tensor
from torch import nn
import ipywidgets as widgets
from torch.nn.modules import padding
from pathlib import Path
   

def psnr ( denoised , ground_truth ) :
    # Peak Signal to Noise Ratio : denoised and ground ̇truth have range [0 , 1]
    mse = torch.mean(( denoised - ground_truth ) ** 2)
    return -10 * torch.log10( mse + 10**-8)


class mod(nn.Module):
    def __init__(self, 
                 skip_connections = True, batch_normalization = True):
        super().__init__()
        self.rel=nn.LeakyReLU(0.1)
        
        self.conv1 = nn.Conv2d(3, 3*4,
                               kernel_size = 8,
                               stride=2,
                               padding = 3)
        
        self.conv21 = nn.Conv2d(3*4, 3*6,
                               kernel_size = 6,
                               padding = 2)
        self.conv22 = nn.Conv2d(3*6, 3*12,
                               kernel_size = 6,
                               padding = 2)
        self.conv23 = nn.Conv2d(3*12, 3*12,
                               kernel_size = 4,
                               padding = 2)

        self.maxlong = nn.MaxPool2d(kernel_size=(3,1),padding=0)
        self.maxwide = nn.MaxPool2d(kernel_size=(1,3),padding=0)
        self.upwide = nn.Upsample(scale_factor=(1,3), mode='nearest')
        self.uplong = nn.Upsample(scale_factor=(3,1), mode='nearest')

        self.conv24 = nn.Conv2d(3*12, 3*4,
                               kernel_size = 2,
                               padding = 1)
        self.conv25 = nn.Conv2d(3*4, 3*4,
                               kernel_size = 5,
                               padding = 2)
        self.conv3 = nn.ConvTranspose2d(3*4, 3,
                                        kernel_size = 8,
                                        stride=2,
                                        padding = 3)
        self.m2=nn.Dropout(p=0.4)
        
    
    def load_model(self,path) -> None:
        self.load_state_dict(torch.load(path))
        pass
            
    
    def forward(self, x):
        y1 = self.conv1(x)
        y = self.m2(y1)
        y = self.conv21(y)
        y = self.rel(y)
        y = self.conv22(y)
        y = self.rel(y)
        y = self.conv23(y)
        ylong=self.maxlong(y)
        ywide=self.maxwide(y)
        y= self.upwide(ywide)+self.uplong(ylong)
        y= self.conv24(y)
        y=self.conv25(y)
        y=y+y1
        y=self.conv3(y)
        y = F.relu(y)
 
        return y
        
        
        
class Model():
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.model=mod().to(self.device)
    
    def train(self, train_input, train_target, num_epochs):
        criterion = nn.MSELoss()
        optim = torch.optim.Adam(self.model.parameters(), lr = 0.001)
        mini_batch_size=100
        train_input, train_target = train_input.to(self.device).type(torch.float), train_target.to(self.device).type(torch.float)
        for e in range(num_epochs):
            acc_loss = 0
            for b in range(0, train_input.size(0), mini_batch_size):
                optim.zero_grad()
                output = self.model(train_input.narrow(0, b, mini_batch_size))
                loss = criterion(output/255, train_target.narrow(0, b, mini_batch_size)/255)
                acc_loss = acc_loss + loss.item()
                loss.backward()
                torch.no_grad()
                optim.step()
            print(e, acc_loss)
        pass
        
    def load_pretrained_model(self):
        model_path=Path(__file__).parent /"model.pt"
        #self.model = torch.load(model_path)
        self.model.load_model(model_path)
        pass
        
    def predict(self,noisy_imgs_val):
        #self.model.eval()
        return self.model(noisy_imgs_val.to(self.device).type(torch.float))


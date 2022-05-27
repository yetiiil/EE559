import torch
from torch import nn
from pathlib import Path
from torch.nn import functional as F
   
torch.set_grad_enabled(True)

def psnr(denoised , ground_truth) :
    # Peak Signal to Noise Ratio : denoised and ground ̇truth have range [0 , 1]
    mse = torch.mean(( denoised - ground_truth ) ** 2)
    return -10 * torch.log10(mse + 10**-8)
                
class mod(nn.Module):
    def __init__(self, skip_connections = True, batch_normalization = True):
        super().__init__()
        torch.set_grad_enabled(True)

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
        class UNet (nn.Module):
            def __init__(self,in_channels = 3, out_channels = 3):
                '''
                initialize the unet 
                '''
                super(UNet, self).__init__()

                self.encode1 = nn.Sequential(
                    nn.Conv2d(in_channels,48,3,stride=1,padding='same',padding_mode ='reflect'),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(48,48,3,stride=1,padding='same',padding_mode ='reflect'),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2))
        
                self.encode2 = nn.Sequential(
                    nn.Conv2d(48,48,3,stride=1,padding='same',padding_mode ='reflect'),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2))
        
                self.encode3 = nn.Sequential(
                    nn.Conv2d(48,48,3,stride=1,padding='same',padding_mode ='reflect'),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(48,48,3,stride=1,padding='same',padding_mode ='reflect'),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode='nearest'))

                self.decode1 = nn.Sequential(
                    nn.Conv2d(96,96,3,stride=1,padding='same',padding_mode ='reflect'),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(96,96,3,stride=1,padding='same',padding_mode ='reflect'),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode='nearest'))
        
                self.decode2 = nn.Sequential(
                    nn.Conv2d(144,96,3,stride=1,padding='same',padding_mode ='reflect'),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(96,96,3,stride=1,padding='same',padding_mode ='reflect'),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode='nearest'))
        
                self.decode3 = nn.Sequential(
                    nn.Conv2d(96 + in_channels, 64,3,stride=1,padding='same',padding_mode ='reflect'),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 32, 3,stride=1,padding='same',padding_mode ='reflect'),
                    nn.ReLU(inplace=True))
        
                self.output_layer = nn.Sequential(
                    nn.Conv2d(32 ,out_channels,3 ,stride=1,padding='same',padding_mode ='reflect'),
                    nn.ReLU(inplace=True))
        
                self._init_weights()

            def forward(self,x):
                '''
                forward function
                '''
                pool1 = self.encode1(x)
                pool2 = self.encode2(pool1)
                pool3 = self.encode2(pool2)
                pool4 = self.encode2(pool3)
                upsample5 = self.encode3(pool4)
                concat5 = torch.cat((upsample5,pool3),dim=1)
                upsample4 = self.decode1(concat5)
                concat4 = torch.cat((upsample4,pool2),dim=1)
                upsample3 = self.decode2(concat4)
                concat3 = torch.cat((upsample3,pool1),dim=1)
                upsample2 = self.decode2(concat3)
                concat2 = torch.cat((upsample2,x),dim=1)
                upsample1 = self.decode3(concat2)
                output = self.output_layer(upsample1)
                return output
    
            def _init_weights(self):
                """Initializes weights using He et al. (2015)."""
                for m in self.modules():
                    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight.data)
                        nn.init.constant_(m.bias.data, 0)

        self.model = UNet()

        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 1e-3, betas=(0.9, 0.999), eps=1e-8)
        self.criterion = nn.MSELoss()

    def save_model(self) -> None :
        torch.save(self.model.state_dict(), 'bestmodel.pth')

    def load_pretrained_model(self) -> None:
        model_path=Path(__file__).parent /"bestmodel.pth"
        if self.device == 'cuda':
            self.model.load_state_dict(torch.load(model_path))
        else:
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 1e-3, betas=(0.9, 0.999), eps=1e-8)
        torch.set_grad_enabled(True)
    
    def train(self, train_input, train_target, num_epochs):

        train_input, train_target = train_input.to(self.device).type(torch.float), train_target.to(self.device).type(torch.float)

        model = self.model
        criterion = self.criterion
        device = self.device
        model = model.to(device)
        optimizer = self.optimizer

        mini_batch_size=100

        model.train()

        print("Starting Training Loop...")
        for epoch in range(num_epochs):
            running_loss = 0.0
            for b in range(0, train_input.size(0), mini_batch_size):
                optimizer.zero_grad()
                denoised_source = model(train_input.narrow(0, b, mini_batch_size))
                loss = criterion(denoised_source, train_target.narrow(0, b, mini_batch_size))
                loss.requires_grad_()
                loss.backward()
                optimizer.step() 

                running_loss += loss.item()

            epoch_loss = running_loss / len(train_input)
            #print('{} Loss: {:.4f}'.format('current '+ str(epoch), epoch_loss))
        
    def predict(self,test_input):
        return torch.clip(self.model(test_input.to(self.device).type(torch.float)), 0.0, 255.0)

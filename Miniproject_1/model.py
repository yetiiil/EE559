import torch
from torch import optim
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

class Model():
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = UNet().to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)

    def save_model(self) -> None :
        torch.save(self.model.state_dict(), 'bestmodel.pth')

    def load_pretrained_model(self) -> None:
        if self.use_cuda:
            self.model = torch.load_state_dict(torch.load('bestmodel.pth'))
        else:
            self.model = torch.load_state_dict(torch.load('bestmodel.pth', map_location='cpu'))

    def train(self, train_input, train_target, num_epochs) -> None:
        #:train_input: tensor of size (N, C, H, W) containing a noisy version of the images. same images, which only differs from the input by their noise.
        #:train_target: tensor of size (N, C, H, W) containing another noisy version of the

        train_input, train_target = train_input.to(self.device).type(torch.float), train_target.to(self.device).type(torch.float)

        model = self.model
        criterion = self.criterion
        optimizer = self.optimizer

        mini_batch_size = 100

        model.train()
        print("Starting Training")
        print("Starting Training Loop...")
        for e in tqdm(range(num_epochs)):
            running_loss = 0.0

            for b in range(0, train_input.size(0), mini_batch_size):
                optimizer.zero_grad()

                denoised_source = model(train_input.narrow(0, b, mini_batch_size))

                loss = criterion(denoised_source, train_target.narrow(0, b, mini_batch_size))

                loss.backward()
                optimizer.step() 

                running_loss +=loss.item() * train_target.narrow(0, b, mini_batch_size).size(0)
            epoch_loss = running_loss / len(train_input)
            print('{} Training Loss: {:.4f}'.format('current '+ str(e+1), epoch_loss))

    def predict(self, test_input) -> Tensor:
        #:test ̇input: tensor of size (N1, C, H, W) that has to be denoised by the trained or the loaded network.
        #:returns a tensor of the size (N1, C, H, W)
        device = self.device
        test_input = test_input.to(device).type(torch.float)

        model = self.model.to(device)

        return model(test_input)

class UNet(nn.Module):
    def __init__(self,in_channels = 3, out_channels = 3):
        '''
        initialize the unet
        '''
        super(UNet, self).__init__()

        self.encode1 = nn.Sequential(
            nn.Conv2d(in_channels,48,3,stride=1,padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(48,48,3,stride=1,padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        
        self.encode2 = nn.Sequential(
            nn.Conv2d(48,48,3,stride=1,padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        
        self.encode3 = nn.Sequential(
            nn.Conv2d(48,48,3,stride=1,padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        
        self.encode4 = nn.Sequential(
            nn.Conv2d(48,48,3,stride=1,padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        
        self.encode5 = nn.Sequential(
            nn.Conv2d(48,48,3,stride=1,padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        
        self.encode6 = nn.Sequential(
            nn.Conv2d(48,48,3,stride=1,padding='same'),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'))

        self.decode1 = nn.Sequential(
            nn.Conv2d(96,96,3,stride=1,padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(96,96,3,stride=1,padding='same'),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'))

        self.decode2 = nn.Sequential(
            nn.Conv2d(144,96,3,stride=1,padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(96,96,3,stride=1,padding='same'),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'))
        
        self.decode3 = nn.Sequential(
            nn.Conv2d(144,96,3,stride=1,padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(96,96,3,stride=1,padding='same'),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'))
        
        self.decode4 = nn.Sequential(
            nn.Conv2d(144,96,3,stride=1,padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(96,96,3,stride=1,padding='same'),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'))
        
        self.decode5 = nn.Sequential(
            nn.Conv2d(96 + in_channels,64,3,stride=1,padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3,stride=1,padding='same'),
            nn.ReLU(inplace=True))
        
        ## output layer
        self.output_layer = nn.Sequential(
            nn.Conv2d(32,out_channels,3,stride=1,padding='same'),
            nn.LeakyReLU(0.1, inplace=True))
        
        ## initialize weight
        self._init_weights()

    def forward(self,x):
        '''
        forward function
        '''
        pool1 = self.encode1(x)
        pool2 = self.encode2(pool1)
        pool3 = self.encode3(pool2)
        pool4 = self.encode4(pool3)
        pool5 = self.encode4(pool4)
        upsample6 = self.encode6(pool5)
        concat6 = torch.cat((upsample6,pool4),dim=1)
        upsample5 = self.decode1(concat6)
        concat5 = torch.cat((upsample5,pool3),dim=1)
        upsample4 = self.decode2(concat5)
        concat4 = torch.cat((upsample4,pool2),dim=1)
        upsample3 = self.decode3(concat4)
        concat3 = torch.cat((upsample3,pool1),dim=1)
        upsample2 = self.decode4(concat3)
        concat2 = torch.cat((upsample2,x),dim =1)
        umsample0 = self.decode5(concat2)
        output = self.output_layer(umsample0)
        return output
    
    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)
import torch
from torch import optim
from torch import Tensor
from torch import nn
from torch.nn import functional as F

class Model():
    def __init__(self) -> None :
    ## instantiate model + optimizer + loss function + any other stuff you need
        class UNet(nn.Module):
            def __init__(self, in_channels=3, out_channels=3):
                super(UNet, self).__init__()
                # Layers: enc_conv0, enc_conv1, pool1
                self._block1 = nn.Sequential(
                    nn.Conv2d(in_channels, 48, 3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(48, 48, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2))

                # Layers: enc_conv(i), pool(i); i=2..5
                self._block2 = nn.Sequential(
                    nn.Conv2d(48, 48, 3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2))

                # Layers: enc_conv6, upsample5
                self._block3 = nn.Sequential(
                    nn.Conv2d(48, 48, 3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(48, 48, 3, stride=2, padding=1, output_padding=1))
                    #nn.Upsample(scale_factor=2, mode='nearest'))

                # Layers: dec_conv5a, dec_conv5b, upsample4
                self._block4 = nn.Sequential(
                    nn.Conv2d(96, 96, 3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(96, 96, 3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
                    #nn.Upsample(scale_factor=2, mode='nearest'))

                # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
                self._block5 = nn.Sequential(
                    nn.Conv2d(144, 96, 3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(96, 96, 3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
                    #nn.Upsample(scale_factor=2, mode='nearest'))

                # Layers: dec_conv1a, dec_conv1b, dec_conv1c,
                self._block6 = nn.Sequential(
                    nn.Conv2d(96 + in_channels, 64, 3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 32, 3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, out_channels, 3, stride=1, padding=1),
                    nn.LeakyReLU(0.1))

            def forward(self, x):
                # Encoder
                pool1 = self._block1(x)
                pool2 = self._block2(pool1)
                pool3 = self._block2(pool2)
                pool4 = self._block2(pool3)
                pool5 = self._block2(pool4)

                # Decoder
                upsample5 = self._block3(pool5)
                concat5 = torch.cat((upsample5, pool4), dim=1)
                upsample4 = self._block4(concat5)
                concat4 = torch.cat((upsample4, pool3), dim=1)
                upsample3 = self._block5(concat4)
                concat3 = torch.cat((upsample3, pool2), dim=1)
                upsample2 = self._block5(concat3)
                concat2 = torch.cat((upsample2, pool1), dim=1)
                upsample1 = self._block5(concat2)
                concat1 = torch.cat((upsample1, x), dim=1)

                # Final activation
                return self._block6(concat1)

        self.model = UNet()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.optimizer = optim.SGD(self.model.parameters(), lr = 1e-3)
        self.loss_function = nn.MSELoss()

    def save_model(self) -> None :
        torch.save(self.model.state_dict(), 'bestmodel.pth')

    def load_pretrained_model(self) -> None:
        self.model.load_state_dict(torch.load('bestmodel.pth'))

        if self.use_cuda:
            self.model.load_state_dict(torch.load('bestmodel.pth'))
        else:
            self.model.load_state_dict(torch.load('bestmodel.pth', map_location='cpu'))

    def train(self, train_input, train_target) -> None:
    #:train_input: tensor of size (N, C, H, W) containing a noisy version of the images. same images, which only differs from the input by their noise.
    #:train_target: tensor of size (N, C, H, W) containing another noisy version of the
        nb_epochs = 400
        mini_batch_size = 100

        train_input, train_target = train_input.to(self.device), train_target.to(self.device)
        
    def predict(self, test_input) -> torch.Tensor:
    #:test ̇input: tensor of size (N1, C, H, W) that has to be denoised by the trained or the loaded network.
    # #: returns a tensor of the size (N1, C, H, W)
        pass

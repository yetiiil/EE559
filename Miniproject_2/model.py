from .others.module import MSE, Sequential, Conv2d, ReLU, TransposeConv2d, Sigmoid
from .others.optimizer import SGD
import torch
from pathlib import Path
import pickle


noisy_imgs, clean_imgs = torch.load('/Users/liyuxiao/Downloads/CS2022/DeepLearning/EE559/val_data.pkl')
noisy_imgs , clean_imgs = noisy_imgs.float()/255, clean_imgs.float()/255

def psnr(denoised ,ground_truth):
  # Peak Signal to Noise Ratio : denoised and ground˙truth have range [0 , 1]
  mse = torch.mean ((denoised - ground_truth) ** 2)
  return -10 * torch.log10(mse + 10**-8)

class Model():
    def __init__(self) -> None :
    ## instantiate model + optimizer + loss function + any other stuff you need
        self.net = Sequential()
        self.net.add_module("conv1", Conv2d(3, 48, 2, stride=2))
        self.net.add_module("relu1", ReLU())
        self.net.add_module("conv2", Conv2d(48, 48, 2, stride=2))
        self.net.add_module("relu2", ReLU())
        self.net.add_module("trans1", TransposeConv2d(48, 48, 2, stride=2))
        self.net.add_module("relu3", ReLU())
        self.net.add_module("trans2", TransposeConv2d(48, 3, 2, stride=2))
        self.net.add_module("sig", Sigmoid())

        for m in self.net.modules:
            if isinstance(m, Conv2d) or isinstance(m, TransposeConv2d):
                m.weight.normal_()
                m.bias.zero_()

        self.optimizer = SGD(self.net.param(), lr=1e-2, momentum=0.9)
        self.criterion = MSE()
       # self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        torch.set_grad_enabled(False)

    def save_model(self, model_path = 'bestmodel.pth') -> None :
        model_path=Path(__file__).parent /"bestmodel.pth"
        with open(model_path, 'wb') as f:
            pickle.dump(self.net, f)

    def load_pretrained_model(self) -> None:
        model_path=Path(__file__).parent /"bestmodel.pth"
        with open(model_path, 'rb') as f:
            self.net = pickle.load(f)


    def train(self, train_input, train_target, num_epochs) -> None:
    #:train_input: tensor of size (N, C, H, W) containing a noisy version of the images. same images, which only differs from the input by their noise.
    #:train_target: tensor of size (N, C, H, W) containing another noisy version of the
        train_input, train_target = train_input.type(torch.float), train_target.type(torch.float)
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
            print(psnr(clean_imgs, model.forward(noisy_imgs)))

        pass

    def predict(self, test_input) -> torch.Tensor:
    #:test ̇input: tensor of size (N1, C, H, W) that has to be denoised by the trained or the loaded network.
    # #: returns a tensor of the size (N1, C, H, W)
        return self.net.forward(test_input.type(torch.float))

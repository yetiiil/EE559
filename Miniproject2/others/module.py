from torch import empty , cat , arange, Tensor
from torch.nn.functional import fold, unfold
import torch

class Module(object):
    def __init__(self):
        super().__init__()

    def forward(self, *input):
        raise NotImplementedError
    
    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []


class Sigmoid(Module):
    def __init__(self):
        """
        Module representing a Sigmoid activation function
        """
        super(Sigmoid).__init__()
    
    def forward(self, input:Tensor):
        """
        Forward pass of the Sigmoid module. Keeps the positive input unchanged and sets the output 0 for negative input
        :param x: Tensor: input tensor
        :return: Tensor, output tensor
        """
        self.input = input
        return 1 / (1 + (-input).exp())

    def backward(self, gradwrtoutput:Tensor):
        """
        Backward pass of the Sigmoid module
        :param gradwrtoutput: Tensor: loss gradient with respect to the output
        :return: Tensor: loss gradient with respect to the input
        """
        return gradwrtoutput * (1 / (1 + (-input).exp())) * ((-input).exp() / (1 + (-input).exp()))

    def param(self):
        """
        returns 
        :return: 
        """
        return []

class Conv2d(Module):


class Upsampling(Module):

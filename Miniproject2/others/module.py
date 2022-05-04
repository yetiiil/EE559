from torch import empty , cat , arange
from torch.nn.functional import fold, unfold

class Module(object):
    def forward(self, *input):
        raise NotImplementedError
    
    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []
        
class Conv2d(Module):

class Sigmoid(Module):

class SGD(Module):

class Upsampling(Module):



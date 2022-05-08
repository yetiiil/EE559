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
      
class ReLU(Module):
    """ReLU
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input : Tensor):
        self.cache["value"] = input > 0
        return input * self.cache["value"]

    def backward(self, grad : Tensor):
        v = self.cache["value"]
        return grad * v

    
class MSE(Module):
    """Mean Squred Loss
    """
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, pred : Tensor, target : Tensor):
        self.e = pred - target
        return (self.e ** 2).sum() / self.e.shape[0]
    
    def backward(self, grad : Tensor):
        return 2 * self.e * grad / self.e.shape[0]

    def param(self):
        return super().param()

    
class TransposedConv2d(Module):
    """Transposed Convolutional 2D
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, output_padding=0) -> None:
        super().__init__()

        if (isinstance(kernel_size, int)):
            self.kernel_size = (kernel_size, kernel_size)
        elif (isinstance(kernel_size, tuple)):
            self.kernel_size = kernel_size

        if (isinstance(stride, int)):
            self.stride = (stride, stride)
        elif (isinstance(stride, tuple)):
            self.stride = stride

        self.padding = padding
        self.output_padding = output_padding
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = empty((in_channels, out_channels, self.kernel_size[0], self.kernel_size[1]))
        self.bias = empty(out_channels)

        self.weightGrads = empty((in_channels, out_channels, self.kernel_size[0], self.kernel_size[1])).zero_()
        self.biasGrads = empty(out_channels).zero_()
        self.cache = {}

    def _filterMat(self) -> Tensor:
        """
        For weight with the shape of (2, 3, s0, s1), 
        it will be flatten into a (2, 3*s0*s1) matrix:
             
               oc0     oc1     oc2

        ic0: |s0*s1| |s0*s1| |s0*s1|

        ic1: |s0*s1| |s0*s1| |s0*s1|

        """

        return self.weight.reshape(self.in_channels, -1) # (ic, oc, s0, s1) -> (ic, oc*s0*s1)
    
    def _inputMat(self, input : Tensor) -> Tensor:
        """
        For input with the shape of (2, 2, h, w), 
        it will be flatten and transposed into a (2, h*w, 2) matrix:
                
              ic0   ic1

        d0:  |h*w| |h*w|

        d1:  |h*w| |h*w|

        """

        (bs, c, _, _) = input.shape
        return input.reshape(bs, c, -1).transpose(1, 2) # (bs, ic, h, w) -> (bs, h*w, ic)
    
    def _gradMat(self, grad : Tensor) -> Tensor:
        (_, _, s0, s1) = self.weight.shape
        return unfold(grad, kernel_size=(s0, s1), stride=self.stride, padding=self.padding) # (bs, oc*s0*s1, h*w)

    def _inputXfilter(self, input : Tensor) -> Tensor:
        """
        For weight with shape of (2, 3, s0, s1) and input with the shape of (2, 2, h, w), 
        return a (2, h*w, 3, s0, s1) matrix:
                       
                       |s0*s1 + s0*s1 + s0*s1|
        |h*w| |h*w| x                   
                       |s0*s1 + s0*s1 + s0*s1|

        """

        filterMat = self._filterMat()  # (ic, oc*s0*s1)
        inputMat = self._inputMat(input) # (bs, h*w, ic)

        self.cache["input"] = input
        self.cache["inputMat"] = inputMat

        (bs, hw, ic) = inputMat.shape

        result = inputMat.reshape(-1, ic).mm(filterMat).reshape(bs, hw, -1)
        return result.transpose(1, 2)

    def forward(self, input : Tensor):
        (_, _, h, w) = input.shape
        (_, _, s0, s1) = self.weight.shape
        (sh, sw) = self.stride

        oh = (h - 1) * sh + s0
        ow = (w - 1) * sw + s1

        blocks = self._inputXfilter(input) # (bs, h*w, oc, s0, s1)
        print(blocks.shape)

        output = fold(blocks, (oh, ow), (s0, s1), stride=self.stride)

        output = output[:, :, self.padding : oh - self.padding, self.padding : ow - self.padding] # (bs, oc, oh, ow)
        output += self.bias.reshape((self.bias.shape[0], 1, 1))
            
        return output

    def backward(self, grad : Tensor):
        gradMat = self._gradMat(grad) # (bs, oc*s0*s1, h*w)
        inputMat = self.cache["inputMat"] # (bs, h*w, ic)
        filterMat = self._filterMat() # (ic, oc*s0*s1)
        input = self.cache["input"] # (bs, ic, h, w)

        self.weightGrads += gradMat.bmm(inputMat).sum(dim=0).T.reshape(self.weight.shape) # (ic, oc, s0, s1) grad for update filter
        self.biasGrads += grad.sum(dim=(0, 2, 3))
        
        (bs, ocs0s1, hw) = gradMat.shape
        return gradMat.transpose(1, 2).reshape(-1, ocs0s1).mm(filterMat.T).reshape(bs, hw, -1).transpose(1, 2).reshape(input.shape) # (bs, ic, h, w) grad for back propagation

    def param(self):
        return [(self.weight, self.weightGrads), (self.bias, self.biasGrads)]

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding

        if (isinstance(kernel_size, int)):
            self.kernel_size = (kernel_size, kernel_size)
        elif (isinstance(kernel_size, tuple)):
            self.kernel_size = kernel_size

        if (isinstance(stride, int)):
            self.stride = (stride, stride)
        elif (isinstance(stride, tuple)):
            self.stride = stride    

        self.weight = empty((in_channels, out_channels, self.kernel_size[0], self.kernel_size[1]))
        self.bias = empty(out_channels)

        self.weightGrads = empty((in_channels, out_channels, self.kernel_size[0], self.kernel_size[1])).zero_()
        self.biasGrads = empty(out_channels).zero_()
        self.cache = {}

    def _inputMat(self, input : Tensor) -> Tensor:
        """
        For input with the shape of (2, 2, h, w), 
        it will be flatten and transposed into a (2, h*w, 2) matrix:
                
              ic0   ic1

        d0:  |h*w| |h*w|

        d1:  |h*w| |h*w|

        """

        (bs, c, _, _) = input.shape
        return input.reshape(bs, c, -1).transpose(1, 2) # (bs, ic, h, w) -> (bs, h*w, ic)

    def _filterMat(self) -> Tensor:
        """
        For weight with the shape of (2, 3, s0, s1), 
        it will be flatten into a (2, 3*s0*s1) matrix:
             
               oc0     oc1     oc2

        ic0: |s0*s1| |s0*s1| |s0*s1|

        ic1: |s0*s1| |s0*s1| |s0*s1|

        """

        return self.weight.reshape(self.in_channels, -1) # (ic, oc, s0, s1) -> (ic, oc*s0*s1)

    def forward(self, input : Tensor):
        bs, ic, H, W = input.size()
        ic, oc, h, w = self.weight.size()
        unfold_input = torch.nn.functional.unfold(input, self.kernel_size, self.stride)
        weight = self.weight.view(oc, -1)
        output = ((weight @ unfold_input) + self.bias.view(-1,1)).view(bs, oc, ((H-h+1)/self.stride[0]).floor(), -1)

        self.cache["unfold_input"] = unfold_input
        self.cache["input"] = input

        return output

    def backward(self, gradwrtoutput: Tensor):
        '''
        gradwrtoutput shape (N, oc, Hout, Wout)
        - Hout = 1 + (H + 2 * pad - h) / stride
        - Wout = 1 + (W + 2 * pad - w) / stride
        '''
        input = self.cache["input"]
        gradient_reshape = gradwrtoutput.permute(1, 2, 3, 0).reshape(self.out_channels, -1)
        unfold_input = self.cache["unfold_input"].permute(2, 0, 1).reshape(gradient_reshape.shape[1], -1) 

        self.weightGrads += (gradient_reshape @ unfold_input).reshape(self.weight.shape)
        self.biasGrads += gradwrtoutput.sum(dim=(0, 2, 3))

        weight_reshaped = self.weight.reshape(self.out_channels, -1)
        dx = weight_reshaped.t() @ gradient_reshape
        dx = dx.reshape(self.cache["unfold_input"].permute(1, 2, 0).shape).permute(2, 0, 1)
        dx = dx.fold(dx, (input.shape[2],input.shape[3]), kernel_size=self.kernel_size, stride=self.stride)
        return dx

    def param(self):
        return [(self.weight, self.weightGrads), (self.bias, self.biasGrads)]


class Upsampling(Module):

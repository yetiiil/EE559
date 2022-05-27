from torch import empty , cat , arange, Tensor
from torch.nn.functional import fold, unfold
from .kaiming import kaiming_normal_

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
        e = 1 / (1 + (-self.input).exp())
        return gradwrtoutput * e * (1-e)

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
        self.cache = {}

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
    
    def backward(self, grad : Tensor=1.0):
        return 2 * self.e * grad / self.e.shape[0]

    def param(self):
        return super().param()

    
class Upsampling(Module):
    """Transposed Convolutional 2D
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0) -> None:
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

        self.weight = empty((in_channels, out_channels, self.kernel_size[0], self.kernel_size[1])).normal_()
        self.weight = kaiming_normal_(self.weight)
        self.bias = empty(out_channels).normal_()

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

        if (isinstance(kernel_size, int)):
            self.kernel_size = (kernel_size, kernel_size)
        elif (isinstance(kernel_size, tuple)):
            self.kernel_size = kernel_size

        if (isinstance(stride, int)):
            self.stride = (stride, stride)
        elif (isinstance(stride, tuple)):
            self.stride = stride

        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = empty((out_channels, in_channels, self.kernel_size[0], self.kernel_size[1])).normal_()
        self.weight = kaiming_normal_(self.weight)
        self.bias = empty(out_channels).normal_()

        self.weightGrads = empty((out_channels, in_channels, self.kernel_size[0], self.kernel_size[1])).zero_()
        self.biasGrads = empty(out_channels).zero_()
        self.cache = {}
    
    def forward(self, input : Tensor):
        # (bs, ic, h, w)
        (bs, ic, h, w) = input.shape
        (oc, ic, s0, s1) = self.weight.shape
        inputMat = unfold(input, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding) # (bs, ic*s0*s1, h*w)
        filterMat = self.weight.view(oc, -1) # (oc, ic*s0*s1)

        self.cache["inputMat"] = inputMat
        self.cache["input"] = input

        oh = (h - s0 + 2*self.padding) // self.stride[0] + 1
        ow = (w - s1 + 2*self.padding) // self.stride[1] + 1

        (bs, icsos1, hw) = inputMat.shape

        return (filterMat @ inputMat).view(bs, oc, oh, -1) + self.bias.view(self.bias.shape[0], 1, 1)

    def backward(self, grad : Tensor):
        inputMat = self.cache["inputMat"] # (bs, ic*s0*s1, h*w)
        (bs, oc, _, _) = grad.shape # (bs, oc, h*w)
        self.weightGrads += grad.view(bs, oc, -1).bmm(inputMat.transpose(1, 2)).sum(dim=0).reshape(self.weight.shape) # (oc, ic*s0*s1)
        self.biasGrads += grad.sum(dim=(0, 2, 3))

        input = self.cache["input"]
        (bs, ic, h, w) = input.shape
        (oc, ic, s0, s1) = self.weight.shape
        (bs, oc, oh, ow) = grad.shape
        blocks = (grad.view(bs, oc, oh*ow).transpose(1, 2).reshape(-1, oc).mm(self.weight.view(oc, -1))).reshape(bs, oh*ow, -1).transpose(1, 2) # (bs, ic*s0*s1, oh*ow)
        return fold(blocks, (h, w), (s0, s1), stride=self.stride, padding=self.padding).reshape(input.shape) # (bs, ic, h, w)
    
    def param(self):
        return [(self.weight, self.weightGrads), (self.bias, self.biasGrads)]


class Sequential(Module):
    def __init__(self, *mods) -> None:
        self.names = []
        self.modules = {}
        self.params = []
        for i in range(len(mods)):
            self.add_module(str(i), mods[i])

    def add_module(self, name : str, mod : Module):
        self.names.append(name)
        self.modules[name] = mod
        for param in mod.param():
            self.params.append(param)
    
    def forward(self, input : Tensor):
        y = input
        for name in self.names:
            y = self.modules[name].forward(y)
        return y
    
    def backward(self, grad : Tensor = 1.0):
        for name in reversed(self.names):
            grad = self.modules[name].backward(grad)
        return grad
    
    def param(self):
        return self.params
    
    def state_dict(self):
        states = {}
        for name in self.names:
            mod = self.modules[name]
            if isinstance(mod, Conv2d) or isinstance(mod, Upsampling):
                states[name + ".weight"] = mod.weight
                states[name + ".bias"] = mod.bias
        return states
    
    def load_state_dict(self, state_dict: 'dict[str, Tensor]'):
        for k, v in state_dict.items():
            name = k.split('.')[0]
            mod = self.modules[name]
            if isinstance(mod, Conv2d) or isinstance(mod, Upsampling):
                param = k.split('.')[1]
                if param == "weight":
                    mod.weight = v
                elif param == "bias":
                    mod.bias = v
                    
class upsample(Module):
    """Transposed Convolutional 2D
    """

    def __init__(self, input,ratio):
        self.input=input
        self.kernel_size = (ratio, ratio)
        self.in_channels=input.shape[1]
        self.out_channels=input.shape[1]
        self.ratio=ratio
        self.stride = (ratio, ratio)

        self.padding = 0
        self.output_padding = 0

        self.weight = empty((self.in_channels, self.out_channels, self.kernel_size[0], self.kernel_size[1])).normal_()
        self.weight=self.weight/self.weight
       
        self.bias = empty(self.out_channels).normal_()
        self.bias=self.bias-self.bias

        self.weightGrads = empty((self.in_channels, self.out_channels, self.kernel_size[0], self.kernel_size[1])).zero_()
        self.biasGrads = empty(self.out_channels).zero_()
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

        filterMat = self._filterMat()
         # (ic, oc*s0*s1)
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
        output = fold(blocks, (oh, ow), (s0, s1), stride=self.stride)

        output = output[:, :, self.padding : oh - self.padding, self.padding : ow - self.padding] # (bs, oc, oh, ow)
        output += self.bias.reshape((self.bias.shape[0], 1, 1))
            
        return output

    def backward(self, grad : Tensor):
        (bs,oc,h,_)=grad.size()
        unfold_input = unfold(grad,kernel_size=self.ratio,stride=self.ratio)
        unfold_weight=torch.ones(4)
        folding=fold(unfold_input.mean(1),output_size=(2,2),kernel_size=(1))
        return folding

    def param(self):
        return [(self.weight, self.weightGrads), (self.bias, self.biasGrads)]


class NearestUpsample():
    def __init__(self,ratio):
        self.ratio=ratio
        
    def forward(self,x):
        a,b,c,d=x.shape
        out=empty(a,1,c*self.ratio,d*self.ratio)
        for i in range (x.shape[1]):
            temp=x[:,i,:,:]
            temp=temp[None,:].permute(1,0,2,3)
            up=upsample(temp,2)
            temp=up.forward(temp)
            out=torch.cat((out,temp),0)
        out=out[1:,:,:,:]
        out=out.permute(1,0,2,3)
        return(out.float())

    def backward(self,grad):
        (bs,oc,h,_)=grad.size()  
        out=empty(bs,1,int(h/self.ratio),int(h/self.ratio))
        for i in range (grad.shape[1]):
            temp=grad[:,i,:,:]
            temp=temp[None,:].permute(1,0,2,3)
            up=upsample(temp,2)
            temp=up.backward(temp)[None,:]
            out=torch.cat((out,temp),0)
        out=out[1:,:,:,:]
        out=out.permute(1,0,2,3)
        return(out)
    def param(self):
        pass


from torch import empty
def upsampling(ratio,input_data):
    #its bad I don't like it :(
    out=empty(input_data.size()[0]*ratio,input_data.size()[1]*ratio)
    for i in range out.size()[0]:
        for j in range out.size()[1]:
            out[i,j]=input_data[int(i/ratio),int(j/ratio)]
            
class upsampling:
    def __init__(self,ratio):
        self.ratio=ratio
        self.ker=torch.empty((ratio,ratio))
        self.ker=self.ker.div(self.ker)
        
        
    def forward(input):
        (_,de,le,lo)=input.size()
        out=torch.unfold(input,1).mm(ker)
        output=torch.fold((de,le*ratio,lo*ratio),kernel_size=ker.size(),stride=2)
        return output
    
    def backward(grad):
        #return a convolution of grad and ker
        #reshape into line?
        (bs,oc,h,_)=grad.size()
        unfold_input = torch.nn.functional.unfold(grad, self.ratio, self.ratio)
        output = ((ker @ unfold_input) + self.bias.view(-1,1)).view(bs, oc, (h/self.ratio).floor(), -1)
        return output
        
        
    def param():
        #ker and grad but they are just 1 matrixes
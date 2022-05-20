from torch import empty
def upsampling(ratio,input_data):
    #its bad I don't like it :(
    out=empty(input_data.size()[0]*ratio,input_data.size()[1]*ratio)
    for i in range out.size()[0]:
        for j in range out.size()[1]:
            out[i,j]=input_data[int(i/ratio),int(j/ratio)]
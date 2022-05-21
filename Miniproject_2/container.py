class container:
    def __init__(self,*args):
        self.modules=[mod for mod in self.args]
        
        
    def add_module(mod):
        self.modules.append(mod)
       
        
    def seq(input):
        y=input
        for mod in self.modules:
            y=mod.forward(y)
        return y
    
    def back(grad):
        for mod in self.modules:
            grad=mod.backward(grad)
        return grad
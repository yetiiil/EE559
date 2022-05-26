import math

def kaiming_normal_(tensor):

    if 0 in tensor.shape:
        print("Initializing zero-element tensors is a no-op")
        return tensor

    num_input_fmaps = tensor.size(1)
    receptive_field_size = 1
    if tensor.dim() > 2:
        for s in tensor.shape[2:]:
            receptive_field_size *= s
    fan = num_input_fmaps * receptive_field_size
    gain = math.sqrt(2.0)
    std = gain / math.sqrt(fan)

    return tensor.normal_(0, std)

import io
import torch
import torch.nn as nn

def copy_weights(src: nn.Module, dst: nn.Module) -> nn.Module:
    # TODO: serialize src's state dict into a buffer, rewind, then load it into dst
    # Instatniate a buffer_io
    buffer_io = io.BytesIO()
    # Svae teh source weights 
    src_wts = src.state_dict()
    ## save the weights to torch
    torch.save(src_wts, buffer_io)
    ## After writing the data, the internal "cursor" of the buffer sits at the very end. If you try to read from it right now, you will get an empty read. Call .seek(0) on your buffer to reposition the cursor back to the beginning
    buffer_io.seek(0)
    dst_wts = torch.load(buffer_io)
    dst.load_state_dict(dst_wts)

    


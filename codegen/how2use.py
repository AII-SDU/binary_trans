import os
from config import *
from torch import nn
from scipy.ndimage import gaussian_filter1d
from torch.autograd import Variable
import torch
import numpy as np
import eval_utils as utils

import os

# 修改为相对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.dirname(current_dir)
os.environ['PYTHONPATH'] = project_path + os.pathsep + os.environ.get('PYTHONPATH', '')

# 使用相对路径
palmtree = utils.UsableTransformer(
    model_path=os.path.join(current_dir, "palmtreemodel/transformer.ep19"), 
    vocab_path=os.path.join(current_dir, "palmtreemodel/vocab")
)

# tokens has to be seperated by spaces.
if palmtree.is_torchscript():
    print("The model is a TorchScript model.")
else:
    print("The model is not a TorchScript model.")

text = ["mov rbp rdi", 
        "mov ebx 0x1", 
        "mov rdx rbx", 
        "call memcpy", 
        "mov [ rcx + rbx ] 0x0", 
        "mov rcx rax", 
        "mov [ rax ] 0x2e"]

# it is better to make batches as large as possible.
embeddings = palmtree.encode(text)
print("usable embedding of this basicblock:", embeddings)
print("the shape of output tensor: ", embeddings.shape)

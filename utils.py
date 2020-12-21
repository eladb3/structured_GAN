import torch
from torch import nn, optim
import torchvision
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import os
import torchvision.transforms.functional as TF
from tqdm.notebook import tqdm
from copy import deepcopy
import random
from tqdm import tqdm
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader
import time
from torchvision.datasets import CelebA
from torch.nn import functional as F
import numpy as np

device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
print(device, flush = True)

## plt

def plt_tensor(x):
    plt.imshow(TF.to_pil_image(x))
    
def get_local_time():
    lt = time.localtime()
    return f"{lt.tm_mday}_{lt.tm_mon}_{lt.tm_year}_{lt.tm_hour}_{lt.tm_min}_{lt.tm_sec}"

def rmdir_if_empty(path):
    if not os.path.isdir(path): return
    if [f for f in os.listdir(path) if not f.startswith('.')] == []:
        os.rmdir(path)

def plt_row_images(y):
    n = y.size(0)
    for i in range(n):
        plt.subplot(1,n,i+1)
        img = y[i, :, :, :].cpu()
        plt_tensor(img)

## data
def get_transformer(img_size, to_rgb = False, tanh_scale = True): 
    l = [
        transforms.ToTensor(),
        transforms.Resize(img_size),
    ]
    if to_rgb: l.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
    if tanh_scale: l.append(transforms.Lambda(lambda x: x * 2 - 1))
    myt = transforms.Compose(l)
    return myt

def get_celebA(batch_size = 256):
    ds = CelebA(
        root = "data/CelebA",
        split = 'train',
        transform = myt,
        download = True)
    return DataLoader(d, batch_size = batch_size, shuffle = True)


def get_lfw(batch_size, img_size):
    ds = torchvision.datasets.ImageFolder(
        root = "./data/lfw/lfw-deepfunneled",
        transform = get_transformer(img_size))
    return DataLoader(ds, batch_size = batch_size, shuffle = True)

def get_mnist(batch_size, img_size, split = "train"):
    t = get_transformer(img_size, to_rgb = True)
    train = split == "train"
    d = torchvision.datasets.MNIST(root="../DTN/data/MNIST", train = train, download = False, transform = t)
    return DataLoader(d, batch_size = batch_size, shuffle = True)


def get_data(name, batch_size, img_size):
    if name == 'mnist':
        return get_mnist(batch_size, img_size)
    elif name == 'lfw':
        return get_lfw(batch_size, img_size)


def get_total_grad_norms(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

##########

class Sym_ConvGeneral2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding= 0, sym_type = 'lr', transposed = False):
        super(Sym_ConvGeneral2d, self).__init__() 
        
        self.transposed = transposed
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.sym_type = sym_type

        if sym_type == 'diag': self.n_weights = ((kernel_size * kernel_size) // 2) + (kernel_size // 2)
        if sym_type == 'lr': self.n_weights = (kernel_size//2 + kernel_size%2) * kernel_size

        if transposed:
            self.weight = torch.nn.Parameter(torch.randn((in_channels, out_channels, self.n_weights)))
            self.register_buffer('w_kernel', torch.ones((in_channels, out_channels, kernel_size,kernel_size)))

        else:
            self.weight = torch.nn.Parameter(torch.randn((out_channels, in_channels, self.n_weights)))
            self.register_buffer('w_kernel', torch.ones((out_channels, in_channels, kernel_size,kernel_size)))


        self.bias = torch.nn.Parameter(torch.zeros((out_channels, )))
        self.triu_indices = self.w_kernel[0,0,:,:].triu() > 0
        self.lower_indices = self.w_kernel[0,0,:,:].tril() > 0

    def forward(self, x):
        #assert x.size(1) == self.in_channels, "Invalid in_dim"
        # self.w_kernel.detach_()
        w_kernel = torch.zeros_like(self.w_kernel).to(self.w_kernel.device)
        if self.sym_type == 'diag':
            w_kernel[:,:,self.triu_indices]= self.weight
            w_kernel[:, :,self.lower_indices] = w_kernel.transpose(2,3)[:,:,self.lower_indices]
        
        if self.sym_type == 'lr':
            lim = self.kernel_size//2 + self.kernel_size%2
            # print(self.kernel_size, lim, self.n_weights)
            # print(w_kernel[:,:, :, :lim].shape, self.weight.shape)
            w_kernel[:,:, :, :lim] = self.weight.view((self.weight.size(0), self.weight.size(1), -1, lim))
            w_kernel[:,:, :, lim:] = w_kernel[:,:, :, list(reversed(range(lim - self.kernel_size%2)))]
        # print(w_kernel)
        if self.transposed: return F.conv_transpose2d(x, w_kernel, bias = self.bias, stride = self.stride, padding = self.padding)
        return F.conv2d(x, w_kernel, bias = self.bias, stride = self.stride, padding = self.padding)

def gen_block_trans(cin, cout, kernel_size = 4, stride = 2, padding = 1, onlyconv =False, sym_conv = True):
    if sym_conv: conv = Sym_ConvGeneral2d(cin, cout, kernel_size = kernel_size, stride = stride, padding = padding, transposed = True)
    else: conv = nn.ConvTranspose2d(cin, cout, kernel_size = kernel_size, stride = stride, padding = padding)

    if onlyconv: return conv
    return nn.Sequential(conv,nn.BatchNorm2d(cout),nn.ReLU())

def gen_block(cin, cout, kernel_size = 4, stride = 2, padding = 1, onlyconv =False, sym_conv = True):
    if sym_conv: conv = Sym_ConvGeneral2d(cin, cout, kernel_size = kernel_size, stride = stride, padding = padding)
    # else: conv = nn.Conv2d(cin, cout, kernel_size = kernel_size, stride = stride, padding = padding)
    else: conv = SymmetricConv2d(cin, cout, kernel_size = kernel_size, stride = stride, padding = padding, symmetry = {'v':cout})

    if onlyconv: return conv
    return nn.Sequential(conv,nn.BatchNorm2d(cout),nn.LeakyReLU(0.2))

def init_weights(net):
    def init_weights_internal(n):
        try:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        except:
            pass
        net.apply(init_weights_internal)

def get_total_params(model):    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

###

class SymmetricConv2d(nn.Conv2d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride = 1,
            padding = 0,
            dilation = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',
            symmetry: dict = {},
            share_bias: bool = False
    ):     
        '''
        Args:
        symmetry (dict) - number of filters that are symmetric about the horizontal, 
                          vertical, or both axes
                          e.g. {'h':4, 'v': 2, 'hv':8} has 4 filters (2 filter pairs) that are 
                          horizontally symmetric, 2 filters (1 filter pair) which are vertically 
                          symmetric, and 8 filters (2 filter quadruples) that are symmetric 
                          about both axes
        share_bias (bool) - if True, symmetric filter pairs also share their biases
        '''   
        super(SymmetricConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)
        if self.groups > 1: raise ValueError(self.__str__() + ' does not support groups>1')
        if not bias: 
            self.share_bias = False
        else:
            self.share_bias = share_bias
        if symmetry is None: 
            # no symmetry, return a standard Conv2d
            self.symmetry = None
        else:
            # Set defaults for symmetric filters pairs
            symmetry = dict(symmetry) # make a copy
            symmetry.setdefault('h', 0)
            symmetry.setdefault('v', 0)
            symmetry.setdefault('hv', 0)
            self.symmetry = symmetry

            # sanity check: number of filters divisible by 2 resp. 4?
            for key, val in symmetry.items():
                    if (key in ['h','v']) and (val % 2 != 0):
                        raise ValueError('Number of symmetric h and v filters must be divisible by 2')
                    elif (key=='hv') and (val % 4 != 0):
                        raise ValueError('Number of symmetric hv filters must be divisible by 4')
            # sanity check: number of symmetric filters must be <= number of filters
            assert sum(list(symmetry.values())) <= self.out_channels, "Number of symmetric channels exceeds number of out channels"
            self.unique_out_channels = self.out_channels - symmetry['h']//2 - symmetry['v']//2 - 3*symmetry['hv']//4

            # Create only the unique weights 
            if self.transposed:
                self.weight = Parameter(torch.Tensor(
                    in_channels, self.unique_out_channels, *self.kernel_size))
            else:
                self.weight = Parameter(torch.Tensor(
                    self.unique_out_channels, in_channels, *self.kernel_size))

        self.reset_parameters()

    def forward(self, input):
        '''
        Starting from the unique weights, use torch.flip calls to create their 
        symmetric counterparts. Then concatenate all kernels and forward the resultant weights.
        '''
        s = self.symmetry
        weight = [self.weight]
        ix = 0
        if s['h'] > 0:
            weight.append(torch.flip(self.weight[ix:ix+s['h']//2,:,:,:], (3,)))
            ix += s['h']//2
        if s['v'] > 0:
            weight.append(torch.flip(self.weight[ix:ix+s['v']//2,:,:,:], (2,)))
            ix += s['v']//2
        if s['hv'] > 0:
            n = s['hv']//4
            weight.extend([torch.flip(self.weight[ix:ix + n,:,:,:], (3,)),
            torch.flip(self.weight[ix:ix + n,:,:,:], (2,)),
            torch.flip(self.weight[ix:ix + n,:,:,:], (2,3))])
            ix += n

        return self.conv2d_forward(input, torch.cat(weight, dim=0))

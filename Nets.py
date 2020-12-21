import torch
from torch import nn
from torch.nn import functional as F
from math import log2 as log
import utils

ngf = 128
H = 256



# Generator

class SymG(nn.Module):

    def __init__(self, z_size =  100, ztag_size = 5,  H =H, ngf = ngf, device = torch.device('cuda'), sym_conv = True):
        super(SymG, self).__init__()
        self.p =dict(z_size =  z_size, ztag_size = ztag_size,  ztag2_size = z_size - ztag_size,
                     H =H, device = device, n_noise = 512, nc = 3, ngf = ngf, sym_conv = sym_conv)
        self.F1 = SymG_F1(self.p)

        self.DCGAN = Sym_G_DCGAN5(self.p)
    
    def forward(self, x):   
        x = self.F1(x)
        x = x.permute((0,3,1,2))
        x = self.DCGAN(x)
        return x

class Sym_G_DCGAN5(nn.Module):
    def __init__(self, params):
        super(Sym_G_DCGAN5, self).__init__()
        self.p = params
        sym_conv = self.p['sym_conv']
        n_blocks = log(params['H']) - 3
        assert n_blocks // 1 == n_blocks
        n_blocks = int(n_blocks)
        ngf = params['ngf']
        self.blocks = nn.ModuleList()
        self.blocks.append(utils.gen_block_trans(params['n_noise'], ngf * 2 ** (n_blocks - 1), kernel_size = 5, padding = 2, sym_conv = sym_conv))
        for i in reversed(range(n_blocks)):
            j = 2**i
            if i == 0:
                self.blocks.append(utils.gen_block_trans(ngf * j,params['nc'], kernel_size = 5, padding = 2, stride = 2, onlyconv = True, sym_conv = sym_conv))
            else:
                self.blocks.append(utils.gen_block_trans(ngf * j,ngf*(j//2), kernel_size = 5, padding = 2, stride = 2, sym_conv = sym_conv))
        self.blocks.append(nn.Tanh())
        self.blocks = nn.Sequential(*self.blocks)
    
    def forward(self, x):   
        x = self.blocks(x)
        x = x[:,:,:-1,:-1]
        return x


class Sym_G_DCGAN(nn.Module):
    def __init__(self, params):
        super(Sym_G_DCGAN, self).__init__()
        self.p = params
        sym_conv = self.p['sym_conv']
        n_blocks = log(params['H']) - 3
        assert n_blocks // 1 == n_blocks
        n_blocks = int(n_blocks)
        ngf = params['ngf']
        self.blocks = nn.ModuleList()
        self.blocks.append(utils.gen_block_trans(params['n_noise'], ngf * 2 ** (n_blocks - 1), stride = 1, padding = 0, sym_conv = sym_conv))
        for i in reversed(range(n_blocks)):
            j = 2**i
            if i == 0:
                self.blocks.append(utils.gen_block_trans(ngf * j,params['nc'], onlyconv = True, sym_conv = sym_conv))
            else:
                self.blocks.append(utils.gen_block_trans(ngf * j,ngf*(j//2), sym_conv = sym_conv))
        self.blocks.append(nn.Tanh())
        self.blocks = nn.Sequential(*self.blocks)
    
    def forward(self, x):   
        x = self.blocks(x)
        return x


class SymG_F1(nn.Module):

    def __init__(self, params):
        super(SymG_F1, self).__init__()
        self.p = params
        self.fc_tag = nn.Linear(self.p['ztag_size'], 5120)
        self.fc_tag2 = nn.Linear(self.p['ztag2_size'], 7680)

    def forward(self, z):
        ztag, ztag2 = z[:, :self.p['ztag_size']], z[:, -self.p['ztag2_size']:]
        # ztag
        ztag = self.fc_tag(ztag) # (N, 5120)
        ztag = ztag.view(-1, 5, 2, 512)
        # antisym reflection
        zeros = torch.zeros((ztag.size(0), ztag.size(1), 1, ztag.size(3))).to(self.p['device'])
        ztag = torch.cat([ztag, zeros, -ztag[:, :, [1, 0], :]], dim = 2) # (N, 5, 5, 512)

        # ztag2
        ztag2 = self.fc_tag2(ztag2) # (N, 7680)
        ztag2 = ztag2.view(-1, 5, 3, 512)
        ztag2 = torch.cat([ztag2, ztag2[:, :, [1, 0], :]], dim = 2) # (N, 5, 5, 12)

        return ztag + ztag2



## Discriminator

class SymD(nn.Module):

    def __init__(self, ngf = ngf, H = H, sym_conv = True):
        super(SymD, self).__init__() 
        params = dict(ngf = ngf//2, nc = 3, H = H, out_channels = 512, sym_conv = sym_conv)
        self.D_net = Sym_D_DCGAN(params)
        self.fc = nn.Linear(512 * 5 * 3, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.D_net(x)
        # sym folding
        cols = x.size(3)
        lim = cols//2 + cols%2
        x = x[:, :, :, :lim] + x[:, :, :, [-i for i in range(1, lim+1)]]
        x = self.fc(x.reshape((x.size(0), -1)))
        x = self.sigmoid(x)
        return x

class Sym_D_DCGAN(nn.Module):

    def __init__(self, params):
        super(Sym_D_DCGAN, self).__init__()
        self.p = params
        sym_conv = self.p['sym_conv']
        n_blocks = log(params['H']) - 3
        assert n_blocks // 1 == n_blocks
        n_blocks = int(n_blocks)
        ngf = params['ngf']
        self.blocks = nn.ModuleList()
        self.blocks.append(utils.gen_block(params['nc'], ngf, sym_conv = sym_conv))
        for i in range(n_blocks):
            j =  2**i
            if i == n_blocks-1:
                self.blocks.append(utils.gen_block(ngf * j,params['out_channels'], onlyconv = True, padding = 2, sym_conv = sym_conv))
            else:
                self.blocks.append(utils.gen_block(ngf * j,ngf*(j*2), sym_conv = sym_conv))
        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x):   
        x = self.blocks(x)
        return x




## HyperNets

class F_net(nn.Module):
    """
    input:
        Image (3, 32, 32)
    output:
        1. theta_G
    """
    def __init__(self, H, n_g_params):
        super(F_net, self).__init__()
        
        # self.F = nn.Sequential()
        self.F_out_dim = 4209
        self.fc_z = nn.Linear(F_out_dim, z_size)
        self.fc_theta_g = nn.Linear(F_out_dim, n_g_params)

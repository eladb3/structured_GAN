import torch
import utils
from torch import nn, optim
import os
import pickle
import Nets
import time
import matplotlib.pyplot as plt

device = utils.device
z_size = 100
ztag_size = 5

def get_zN(z):
    zn = z.detach().clone()
    zn[:, :ztag_size] *= -1
    return zn

def mirror_img(img):
    return img.flip(3)

def get_z(n, z_size):
    return torch.rand((n, z_size)).to(device) * 2 - 1 
    r = torch.randn((n, z_size)).to(device)
    r += r.min()
    r /= r.max() #[0,1]
    r = (r*2)-1 #[-1, 1]
    return r
def get_rand_noise(like_tensor, noise_label_std = 0.12):
    r = torch.randn_like(like_tensor).abs().to(like_tensor.device) * noise_label_std
    r[r>0.3] = 0.3
    return r

def train(batch_size, Epochs = 2**32, 
            dataset_name = 'lfw',
            lr = 0.0001, noise_label_std = 0.12,
            z_size = 100, ztag_size = 5,img_size = 128,
            Dngf = 128, Gngf = 128,
            sym_conv = True,
            weights = {}, times = {},
            save = True, fast_save = False, cont = False, plt_ = True , hours = float("Inf"),
            show = False,):
    torch.autograd.set_detect_anomaly(True)
    
    base = f"./models/trainings/{utils.get_local_time()}"
    params = locals()
    os.makedirs(base)
    with open(f"{base}/params.txt", 'wt') as f: f.write(str(params))
    with open(f"{base}/params_dict", 'wb') as f: pickle.dump(params, f)
    hours = hours * (60 * 60) # sec in hour            
    
    if cont:
        g = torch.load(f"{cont}/g_net")
        D = torch.load(f"{cont}/D_net")
    else:
        g = Nets.SymG(z_size =  z_size, ztag_size = ztag_size,  H =img_size, ngf = Gngf,device = device, sym_conv = sym_conv)
        D = Nets.SymD(H = img_size, ngf = Dngf, sym_conv = sym_conv)
        utils.init_weights(g)
        utils.init_weights(D)
    g, D = g.to(device), D.to(device)
    opt_g = optim.Adam(g.parameters(), lr=  lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=  lr, betas=(0.5, 0.999))
    lossfn = nn.BCELoss()
    mseloss = nn.MSELoss()
    opts = {'g':opt_g, 'D':opt_D}
    dl = utils.get_data(dataset_name, batch_size, img_size) 

    def myplt(n = 1, path = None):
        z = get_z(n, z_size).to(device)
        g.eval()
        with torch.no_grad(): gz= g(z).cpu()
        gz = (gz+1)*0.5
        g.train()
        plt.figure(figsize=(15,10))
        utils.plt_row_images(gz)
        if path: plt.savefig(f"{path}/imgs.jpg")
        if show: plt.show()
        
    def printl(l):
        mlosses = {n:sum(l[n])/len(l[n]) for n in l if len(l[n]) > 0}
        print(mlosses)

    names = ['L_G', 'L_D', 'L_ALT', 'fake_probs', 'real_probs']
    weights = dict(weights)
    for n in  names: weights.setdefault(n, 1)
    times = dict(times)
    for n in  names: times.setdefault(n, 1)

    start_time = time.time()
    for e in range(Epochs):
        losses = {n:[] for n in names}
        norms = {n:[] for n in names + ['L_D_G']}

        for i, (x, y) in enumerate(dl):
            print(">", end = "")
            # x : (bs, 3, 32, 32)
            x = x.to(device)
            
            ones = torch.ones((x.size(0), 1)).to(device)
            zeros = torch.zeros_like(ones).to(device)
            z = get_z(x.size(0), z_size).to(device)

                
            for j in range(times['L_D']):
                # D step
                # z = get_z(x.size(0), z_size).to(device)
                real_probs = D(x)
                loss = lossfn(D(g(z)), zeros+get_rand_noise(zeros, noise_label_std) ) + lossfn(real_probs, ones-get_rand_noise(ones, noise_label_std) )
                loss *= weights['L_D']
                opts['D'].zero_grad() ; loss.backward() ; opts['D'].step()
                losses['L_D'].append(loss.cpu().item())
                losses['real_probs'].append(real_probs.mean().cpu().item())
                norms['L_D'].append(utils.get_total_grad_norms(D))
                norms['L_D_G'].append(utils.get_total_grad_norms(g))
           
            for j in range(times['L_G']):
                # G step
                # z = get_z(x.size(0), z_size).to(device)
                gz = g(z)
                fake_probs = D(gz)
                loss = lossfn(fake_probs, ones) 
                loss *= weights['L_G']
                opts['g'].zero_grad() ; loss.backward() ; opts['g'].step()
                losses['L_G'].append(loss.cpu().item())
                losses['fake_probs'].append(fake_probs.mean().cpu().item())
                norms['L_G'].append(utils.get_total_grad_norms(g))

            for j in range(times['L_ALT']):
                # Alternative loss
                # z = get_z(x.size(0), z_size).to(device)
                loss = mseloss(g(z), mirror_img(g(get_zN(z))))
                loss *= weights['L_ALT']
                opts['g'].zero_grad() ; loss.backward() ; opts['g'].step()
                losses['L_ALT'].append(loss.cpu().item())
                norms['L_ALT'].append(utils.get_total_grad_norms(g))

            if i%5 == 0: 
                printl(losses)
                printl(norms)
                myplt(4)
                if fast_save:
                    dirpath = f"{base}"
                    if not os.path.isdir(dirpath): os.mkdir(dirpath)
                    for name, model in [("g_net", g), ("D_net", D)]:
                        torch.save(model.cpu(), f"{dirpath}/{name}")
                        model.to(device)
                    print(f"CP -- models saved to {dirpath}/{name}")

        if save and e % 1 == 0:
            dirpath = f"{base}/{e}"
            if not os.path.isdir(dirpath): os.mkdir(dirpath)
            myplt(4, path = f"{base}/{e}")
            for name, model in [("g_net", g), ("D_net", D)]:
                torch.save(model.cpu(), f"{base}/{e}/{name}")
                model.to(device)
            print(f"CP -- models saved to {base}/{e}/{name}")
        print()
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> EPOCH {e} END")
        printl(losses)
        printl(norms)   
        myplt(4)
    
        
        if e > Epochs: break
        if (time.time() - start_time) > hours: break
        
    if save:
        for name, model in [("g_net", g), ("D_net", D)]:
            torch.save(model.cpu(), f"{base}/{name}")

    utils.rmdir_if_empty(base)

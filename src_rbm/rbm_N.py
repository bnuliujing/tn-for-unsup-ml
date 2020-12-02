import numpy as np
import torch
import time
import argparse
from math import sqrt
from numba import jit
import os
class RBM_nll(object):
  def __init__(self,config=None, n_visible=28*28, n_hidden=500, weights=None, bias_hidden=None, bias_visible=None, seed=2345,device='cpu'):

    self.n_visible = n_visible
    self.n_hidden = n_hidden
    self.device=device
    self.config=config
    torch.manual_seed(seed)
    if weights is None:
      self.weights = torch.randn(n_visible, n_hidden, dtype=torch.float64,device=device) / sqrt(n_visible * n_hidden)

    else:
      self.weights = weights
    if bias_hidden is None:
      self.bias_hidden = torch.randn(n_hidden, dtype=torch.float64,device=device) / sqrt(n_hidden)
    else:
      self.bias_hidden = bias_hidden
    if bias_visible is None:
      self.bias_visible = torch.randn(n_visible, dtype=torch.float64,device=device) / sqrt(n_visible)
    else:
      self.bias_visible = bias_visible
    self.weights.requires_grad=True
    self.bias_visible.requires_grad=True
    self.bias_hidden.requires_grad=True
    self.parameters=[self.weights,self.bias_hidden,self.bias_visible]

  def prob_hv(self,sample):
      pre_sigmoid_activation = sample @ self.weights + self.bias_hidden
      return torch.sigmoid(pre_sigmoid_activation)
  def prob_v(self,sample,lnz): #[N]
      prob_v=torch.exp(sample@self.bias_visible+torch.log(torch.exp(sample@self.weights+self.bias_hidden)+1).sum(dim=1)-lnz)
      return prob_v

  def lnz(self):
      part1 = (self.bias_visible.unsqueeze(0) @ self.config.unsqueeze(2)).squeeze()
      v1 = torch.exp((self.config.unsqueeze(1) @ self.weights).squeeze() + self.bias_hidden) + 1
      part2 = (torch.log(v1).sum(1))
      lnz = torch.logsumexp(part1 + part2, dim=0)
      return lnz

  def nll(self,input,lnz):
      lnz_model1 = (self.bias_visible.unsqueeze(0) @ input.unsqueeze(2)).mean(dim=0)
      lnz_model2 = torch.log(
          torch.exp(input @ self.weights + self.bias_hidden) + 1).sum(dim=1).mean(
          dim=0)
      logp = -1 * (lnz_model1 + lnz_model2)
      nll = logp + lnz
      return nll

def random_samples(bs,num):
    samples=torch.randint(0,2,(bs,num))
    return samples.to(torch.float64)

def train_backward(v,h,data=None,exact_config=None,lr=0.1,seed=50, epochs=50,device='cpu'):
    rbm = RBM_nll(data,exact_config, v, h, seed=seed,device=device)
    opt = torch.optim.Adam(rbm.parameters, lr)
    nll=[]
    M=len(data)
    N=len(exact_config)
    for epoch in range(epochs):
        t0=time.time()
        loss,lnz=rbm.nll()
        nll.append(loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(epoch,loss.item(),time.time()-t0)
    return nll

def train(v,h,data=None,exact_config=None,lr=0.1,seed=50, epochs=50,device='cpu'):
    rbm = RBM_nll(exact_config, v, h, seed=seed, device=device)
    opt = torch.optim.Adam(rbm.parameters, lr)
    nll=[]
    M=len(data)
    N=2**v
    for epoch in range(epochs):
        t0=time.time()
        if (len(nll)>600 and np.abs(nll[-2]-nll[-1])<2e-3 and np.abs(nll[-3]-nll[-2])<2e-3):
            break
        else:
            lnz=rbm.lnz()
            prob_hv_config = rbm.prob_hv(exact_config)  # [N,h]
            prop_v = rbm.prob_v(exact_config, lnz)  # [N]
            g0=torch.einsum('n,nh,nv->vh', prop_v,prob_hv_config,exact_config)
            g1=prop_v @ exact_config
            g2=prop_v @ prob_hv_config
            opt.zero_grad()
            loss = rbm.nll(data,lnz)
            prob_hv_data = rbm.prob_hv(data)  # [M,h]
            rbm.weights.grad = -1 * (data.t() @ prob_hv_data) / M + g0
            rbm.bias_visible.grad = -1 * data.mean(dim=0) + g1
            rbm.bias_hidden.grad = -1 * prob_hv_data.mean(dim=0) + g2
            opt.step()
            nll.append(loss.item())

        print(epoch,loss.item(),time.time()-t0)

    np.save('./config/sample%d_v%d_h%d_seed%d_600/weights.npy' % (M,v, h, seed), rbm.weights.data.numpy())
    np.save('./config/sample%d_v%d_h%d_seed%d_600/bias_v.npy' % (M,v, h, seed), rbm.bias_visible.data.numpy())
    np.save('./config/sample%d_v%d_h%d_seed%d_600/bias_h.npy' % (M,v, h, seed), rbm.bias_hidden.data.numpy())
    return nll



def main():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate, default: 0.1')
    parser.add_argument('--epochs', type=int, default=100,
                        help='epochs, default: 100')
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--h', type=int, default=500)

    args = parser.parse_args()

    @jit()
    def exact_config(D):
        config = np.empty((2 ** D, D))
        for i in range(2 ** D - 1, -1, -1):
            num = i
            for j in range(D - 1, -1, -1):
                config[i, D - j - 1] = num // 2 ** j
                if num - 2 ** j >= 0:
                    num -= 2 ** j

        return config
    nsamples=100
    seed=args.seed
    for v in [10,20]:
        config = torch.from_numpy(exact_config(args.v)).to(torch.float64).to(args.device)
        #config=torch.from_numpy(np.load('./config/rbm_v%d.npy' % v)).to(torch.float64).to(args.device)
        for k in range(5):
            torch.manual_seed(seed + k * 10)
            data = random_samples(nsamples, v).to(args.device)
            os.makedirs('./config/sample%d_v%d_h%d_seed%d_600' % (nsamples,v, args.h,  seed + k * 10))
            nll = train(v, args.h, data=data, exact_config=config, lr=args.lr, seed=seed + k * 10,epochs=args.epochs, device=args.device)
            print('#sample,nll:', nsamples, np.min(nll))
            np.savetxt('./config/sample%d_v%d_h%d_seed%d_600/loss_v%dh%dlr%sseed%d.txt' % (
                nsamples,v, args.h, seed + k * 10, v, args.h, args.lr, seed + k * 10), nll)

if __name__ == '__main__':
    main()

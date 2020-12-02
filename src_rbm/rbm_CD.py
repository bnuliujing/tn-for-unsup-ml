import numpy as np
import torch
import time
import argparse
from numba import jit
from math import sqrt
import os
class RBM_nll(object):
  def __init__(self, input=None,config=None, n_visible=28*28, n_hidden=500, weights=None, bias_hidden=None, bias_visible=None, seed=2345,device='cpu'):

    self.input = input
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
  def lnz(self):
      part1 = (self.bias_visible.unsqueeze(0) @ self.config.unsqueeze(2)).squeeze()
      v1 = torch.exp((self.config.unsqueeze(1) @ self.weights).squeeze() + self.bias_hidden) + 1
      part2 = (torch.log(v1).sum(1))
      lnz = torch.logsumexp(part1 + part2, dim=0)
      return lnz

  def nll(self):
      lnz_model1 = (self.bias_visible.unsqueeze(0) @ self.input.unsqueeze(2)).mean(dim=0)
      lnz_model2 = torch.log(
          torch.exp(self.input @ self.weights + self.bias_hidden) + 1).sum(dim=1).mean(
          dim=0)
      logp = -1 * (lnz_model1 + lnz_model2)
      lnz = self.lnz()
      nll = logp + lnz
      return nll,lnz
  def propup(self, vis):
    pre_sigmoid_activation = vis@self.weights + self.bias_hidden
    return torch.sigmoid(pre_sigmoid_activation)

  def sample_h_given_v(self, v0_sample):
    h1_mean = self.propup(v0_sample)
    h1_sample = torch.distributions.binomial.Binomial( 1, h1_mean).sample()
    return [h1_mean, h1_sample]

  def propdown(self, hid):
    pre_sigmoid_activation = hid@ (self.weights.t()) + self.bias_visible
    return torch.sigmoid(pre_sigmoid_activation)

  def sample_v_given_h(self, h0_sample):
    v1_mean = self.propdown(h0_sample)
    v1_sample = torch.distributions.binomial.Binomial( 1, v1_mean).sample()
    return [v1_mean, v1_sample]

  def gibbs_hvh(self, h0_sample):
    v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
    h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
    return [v1_mean, v1_sample,h1_mean, h1_sample]

  def get_cost_updates(self, lr=0.1, k=1):
    ph_mean, ph_sample = self.sample_h_given_v(self.input)

    for step in range(k):
      if step == 0:
        nv_means, nv_samples,nh_means, nh_samples = self.gibbs_hvh(ph_sample)
      else:
        nv_means, nv_samples,nh_means, nh_samples = self.gibbs_hvh(nh_samples)

    weights_grad=(self.input.t())@ph_mean - (nv_samples.t())@ nh_means
    bias_v_grad=torch.mean(self.input - nv_samples, dim=0)
    bias_h_grad=torch.mean(ph_mean - nh_means, dim=0)

    return -weights_grad,-bias_v_grad,-bias_h_grad

def random_samples(bs,num):
    samples=torch.randint(0,2,(bs,num))
    return samples.to(torch.float64)

def train(v,h,data=None,exact_config=None,k=1,lr=0.1,seed=50, epochs=50,device='cpu'):
    rbm = RBM_nll(data,exact_config, v, h, seed=seed,device=device)
    print("number of model parameters:", sum([np.prod(p.size()) for p in rbm.parameters]))
    opt = torch.optim.Adam(rbm.parameters, lr,betas=(0.9,0.9))
    nll=[]
    M=len(data)
    N=len(exact_config)
    for epoch in range(epochs):
        t0=time.time()
        loss,lnz=rbm.nll()
        nll.append(loss.item())
        if nll[-1]-np.log(M)<1e-6 or (len(nll)>400 and np.abs(nll[-2]-nll[-1])<2e-3 and np.abs(nll[-3]-nll[-2])<2e-3):
            break
        else:
            opt.zero_grad()
            rbm.weights.grad,rbm.bias_visible.grad,rbm.bias_hidden.grad=rbm.get_cost_updates(lr=1e-3,k=k)
            opt.step()
        print(epoch,loss.item(),time.time()-t0)
    np.save('./result/output_n%d_h%d_CD_k%d/sample%d_seed%d_lr%s/weights.npy' % (v, h,k, M, seed,lr), rbm.weights.data.numpy())
    np.save('./result/output_n%d_h%d_CD_k%d/sample%d_seed%d_lr%s/bias_v.npy' % (v, h,k, M, seed,lr), rbm.bias_visible.data.numpy())
    np.save('./result/output_n%d_h%d_CD_k%d/sample%d_seed%d_lr%s/bias_h.npy' % (v, h,k, M, seed,lr), rbm.bias_hidden.data.numpy())
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
    parser.add_argument('--v', type=int, default=20)
    parser.add_argument('--h', type=int, default=500)
    parser.add_argument('--k', type=int, default=1)

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


    nsamples=[2**i for i in range(1,11,1)]

    config = torch.from_numpy(exact_config(args.v)).to(torch.float64).to(args.device)
    seed=args.seed
    for i in nsamples:
        for k in range(1, 5):
            torch.manual_seed(seed + k * 10)
            data = random_samples(i, args.v).to(args.device)
            os.makedirs('./result/output_n%d_h%d_CD_k%d/sample%d_seed%d_lr%s' % (
                args.v, args.h, args.k, i, seed + k * 10, args.lr))
            nll = train(args.v, args.h, data=data, exact_config=config, lr=args.lr, k=args.k,seed=seed + k * 10,epochs=args.epochs, device=args.device)
            print('#sample,nll:', i, np.min(nll))
            np.savetxt('./result/output_n%d_h%d_CD_k%d/sample%d_seed%d_lr%s/loss_v%dh%dlr%sseed%d.txt' % (
                args.v, args.h, args.k, i, seed + k * 10, args.lr, args.v, args.h, args.lr, seed + k * 10), nll)


if __name__ == '__main__':
    main()

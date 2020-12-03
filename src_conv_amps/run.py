import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from convolution_mps import AutoregressiveMPS_sharing
import time
import os
from args import args
import math


# ------------------------------------------------------------------------------
def run_epoch(split, upto=None):
    t1 = time.time()
    torch.set_grad_enabled(split == 'train')  # enable/disable grad for efficiency of forwarding test batches
    model.train() if split == 'train' else model.eval()
    x = xtr if split == 'train' else xte
    N = x.shape[0]
    B = args.batch_size
    nsteps = N // B if upto is None else min(N // B, upto)
    lossfs = []
    t1 = time.time()
    for step in range(nsteps):
        xb = Variable(x[step * B:step * B + B]).to(args.device)

        if split == 'train':
            opt.zero_grad()
            logx_hat=model(xb)
            assert not torch.isnan(logx_hat).any()
            log_prob = logx_hat[:, :, 0] * xb[:, h:, h:(xb.shape[2] - h)].reshape(-1,784) + logx_hat[:, :, 1] * (1 - xb[:, h:, h:(xb.shape[2] - h)].reshape(-1, 784))
            assert not torch.isnan(log_prob).any()
            loss = -log_prob.sum(-1).mean()
            loss.backward()
            if args.clip_grad > 0:
                for p in params:
                    nn.utils.clip_grad_norm_(p, max_norm=args.max_norm)

            opt.step()

        else:
            logx_hat = model(xb)
            assert not torch.isnan(logx_hat).any()
            log_prob = logx_hat[:, :, 0] * xb[:, h:, h:(xb.shape[2] - h)].reshape(-1, 784) + logx_hat[:, :, 1] * (
                        1 - xb[:, h:, h:(xb.shape[2] - h)].reshape(-1, 784))
            assert not torch.isnan(log_prob).any()
            loss = -log_prob.sum(-1).mean()

        lossf = loss.data.item()
        lossfs.append(lossf)
    print("%s epoch average loss: %f, time: %f" % (split, np.mean(lossfs), time.time() - t1))
    return np.mean(lossfs)

# ------------------------------------------------------------------------------

if __name__ == '__main__':

    # --------------------------------------------------------------------------

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # load the dataset
    mnist = np.load(args.data_path)
    xtr, xte = mnist['train_data'], mnist['test_data']
    xtr = torch.from_numpy(xtr)
    xte = torch.from_numpy(xte)
    #padding
    h = math.floor((np.sqrt(1 + 2 * args.con) - 1) / 2)
    padding = torch.nn.ConstantPad2d((h, h, h, 0), 0)
    xtr=padding(xtr.view(xtr.shape[0],28,28))
    xte=padding(xte.view(xte.shape[0],28,28))

    layer=[]
    layer.append(AutoregressiveMPS_sharing(0, h, Dmax=args.Dmax, seed=args.seed, init_std=args.init, fixed_bias=args.fixed_bias, device=args.device))
    if args.net_depth>2:
        for i in range(args.net_depth - 2):
            layer.append(AutoregressiveMPS_sharing(1, h, Dmax=args.Dmax, seed=args.seed, init_std=args.init,fixed_bias=args.fixed_bias, device=args.device))
    layer.append(AutoregressiveMPS_sharing(2, h, Dmax=args.Dmax, seed=args.seed, init_std=args.init, fixed_bias=args.fixed_bias, device=args.device))
    model=nn.Sequential(*layer)
    params = list(model.parameters())
    model = model.to(args.device)

    opt = torch.optim.Adam(params, args.lr, weight_decay=args.weightDec)
    print("number of model parameters:", sum([np.prod(p.size()) for p in params]))
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_scheduler, gamma=0.1)

    # start the training
    loss_tr = []
    loss_te = []
    path='./output_multilayer/MNISTbs%dDmax%d_layer8_gene_softmax_scheduler' % (args.batch_size, args.Dmax)
    os.makedirs(path)
    with open(path+'/MNISTbs%dDmax%dlr%.gCon%dWeight_decay%g.txt' % (args.batch_size, args.Dmax, -np.log10(args.lr), args.con,-np.log10(args.weightDec)), 'a',
              newline='\n') as f:
        f.write('parameters:%s\n'%(args))

        f.write("number of model parameters:%d\n" % (sum([np.prod(p.size()) for p in params])))

        f.write('The number of pixels considered: %d,%d\n' % (2*h*(h+1),args.con))

        f.write('%s %s %s %s\n' % ('epoch', 'train_loss', 'test_loss', 'time'))
        s=0
        best_nll=1000
        for epoch in range(args.epoch):
            t0 = time.time()
            print("epoch %d" % (epoch))
            with torch.no_grad():
                l_te = run_epoch('test')
            if l_te.item()<best_nll:
                best_nll=l_te.item()
                torch.save(model.state_dict(), path + 'model.pt')
            l_tr = run_epoch('train')
            loss_tr.append(l_tr)
            loss_te.append(l_te)
            scheduler.step(epoch)
            ti = time.time() - t0
            print('time %f' % (ti))
            f.write('%d %f %f %f\n' % (epoch, l_tr, l_te, ti))
            if epoch>50:
                if  (np.abs(loss_te[-3]-loss_te[-2])<2e-3 and np.abs(loss_te[-2]-loss_te[-1])<2e-3) or  (loss_te[-3]-loss_te[-2]<0 and loss_te[-2]-loss_te[-1]<0) :
                    s+=1
                    if s==5:
                        break
            else:
                continue

    f.close()

    def sample( bs, random_start=False):
        """
        Sample images/spin configurations
        """
        n=784
        samples = torch.zeros([bs, 28,28], device=args.device)
        if random_start:
            samples[:,0,0] = torch.randint(2, size=(bs), dtype=torch.float, device=args.device)
        else:
            samples[:, 0,0] = 0.
        for idx in range(28):
            for jdx in range(28):
                sam = padding(samples)
                if idx ==jdx== 0:
                    continue
                else:
                    prob = model(sam)
                    samples[:, idx,jdx] = torch.bernoulli(torch.softmax(prob[:,idx*28+jdx], dim=1)[:, 0])
        return samples


    if args.sample>0:
        with torch.no_grad():
            model.load_state_dict(torch.load(path + 'model.pt'))
            samples=sample(args.sample)
        np.save(path + '/samples_bs%dDmax%dlr%.gCon%dWeight_decay%g.npy' % (args.batch_size, args.Dmax, -np.log10(args.lr), args.con, -np.log10(args.weightDec)),samples.cpu().data.numpy())
    print("optimization done. The best test nll is %f:"%best_nll)
    np.savetxt(path+'/MNISTtrainbs%dDmax%dlr%.gCon%dWeight_decay%g.txt' % (args.batch_size, args.Dmax, -np.log10(args.lr), args.con,-np.log10(args.weightDec)), loss_tr)
    np.savetxt(path+'/MNISTtestbs%dDmax%dlr%.gCon%dWeight_decay%g.txt' % (args.batch_size, args.Dmax, -np.log10(args.lr), args.con,-np.log10(args.weightDec)), loss_te)



import os
import argparse
import time
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from amps import AMPS


def main():
    # training setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='gpu id, default: 0')
    parser.add_argument('--n', type=int, default=20, help='system size, default: 20')
    parser.add_argument('--m', type=int, default=100, help='bond dimension, default: 100')
    parser.add_argument('--D', type=int, default=10, help='bond dimension, default: 10')
    parser.add_argument('--seed', type=int, default=1, help='random seed, default: 1')
    parser.add_argument('--epochs', type=int, default=10000, help='epochs, default: 10000')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate, default: 1e-3')
    args = parser.parse_args()
    print(args)

    # make dirs
    path = './results/capacity/'
    if not os.path.exists(path):
        os.makedirs(path)

    torch.manual_seed(args.seed)

    device = 'cuda:' + str(args.gpu)
    model = AMPS(n=args.n, bond_dim=args.D)
    model = model.to(device)

    randvec = torch.randint(2, size=(args.m, args.n), dtype=torch.float)
    randvec = randvec.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # train
    best_nll = 1000.
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        log_prob = model(randvec)
        nll_loss = -1.0 * torch.mean(log_prob)
        nll_loss.backward()
        # for p in model.parameters():
        #     nn.utils.clip_grad_norm_(p, max_norm=1)
        optimizer.step()
        if nll_loss.item() < best_nll:
            best_nll = nll_loss.item()
        print('Epoch %i\tNLL %.6f' % (epoch, best_nll))

    # logging
    print('Best NLL: %.4f, log(m): %.4f' % (best_nll, np.log(args.m)))
    # with open(path + 'n-20-mps.txt', 'a', newline='\n') as f:
    with open(path + 'm-100-mps.txt', 'a', newline='\n') as f:
        f.write('%i %i %i %i %.8f\n' % (args.n, args.m, args.D, args.seed, best_nll))


if __name__ == '__main__':
    main()

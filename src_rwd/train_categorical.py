import pickle
import os
import argparse
import time
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from model.softmax_model import ARMPS
from model.softmax_model_share import ARMPSShare


def main():
    # training setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='gpu id, default: 0')
    parser.add_argument('--D', type=int, default=10, help='bond dimension, default: 10')
    parser.add_argument('--seed', type=int, default=1, help='random seed, default: 1')
    parser.add_argument('--epochs', type=int, default=10000, help='epochs, default: 10000')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate, default: 1e-3')
    parser.add_argument('--data', type=str, choices=['biofam', 'spect', 'lymphography', 'tumor', 'votes', 'flare'], help='data name')
    parser.add_argument('--model', type=str, choices=['armps', 'armpsshare'], help='model name')
    parser.add_argument('--verbose', action='store_true', default=False, help='print training results')
    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)

    # # make dirs
    # path = './results/capacity/'
    # if not os.path.exists(path):
    #     os.makedirs(path)

    # load dataset
    if args.data == 'biofam':
        with open('datasets/' + args.data, 'rb') as f:
            data = pickle.load(f, encoding='latin1')[0].astype(int)
    else:
        with open('datasets/' + args.data, 'rb') as f:
            data = pickle.load(f)[0].astype(int)
    m, n = data.shape
    feature_dim = data.max() + 1

    device = 'cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu'
    if args.model == 'armps':
        model = ARMPS(n=n, bond_dim=args.D, feature_dim=feature_dim)
    elif args.model == 'armpsshare':
        model = ARMPSShare(n=n, bond_dim=args.D, feature_dim=feature_dim)
    else:
        raise Error('Unknown model name')
    model = model.to(device)

    data = torch.from_numpy(data)
    data = data.to(device)

    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    # train
    # best_nll = 1000.
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        log_prob = model(data)
        nll_loss = -1.0 * torch.mean(log_prob)
        nll_loss.backward()
        # for p in model.parameters():
        #     nn.utils.clip_grad_norm_(p, max_norm=1)
        optimizer.step()
        if args.verbose:
            print('# epoch: %i\tNLL: %.4f' % (epoch, nll_loss.item()))
    with open('results_' + args.data + '.txt', 'a', newline='\n') as f:
        f.write('%s %i %.2e %i %.4f\n' % (args.model, args.D, args.lr, args.epochs, nll_loss))

if __name__ == '__main__':
    main()
import os
import argparse
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from amps import AMPS
from sk import SKModel

torch.manual_seed(2020)


def main():
    # training setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='gpu id, default: 0')
    parser.add_argument('--n', type=int, default=20, help='SK model size, default: 20')
    parser.add_argument('--beta_init', type=float, default=0.1, help='initial beta, default: 0.1')
    parser.add_argument('--beta_final', type=float, default=5.0, help='final beta, default: 5.0')
    parser.add_argument('--beta_interval', type=float, default=0.1, help='interval of annealing, default: 0.1')
    parser.add_argument('--D', type=int, default=5, help='bond dimension, default: 5')
    parser.add_argument('--epochs', type=int, default=1000, help='epochs, default: 1000')
    parser.add_argument('--bs', type=int, default=10000, help='batch size, default: 10000')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate, default: 1e-3')
    args = parser.parse_args()

    # make dirs
    path = './results/sk-new/N%iD%iBS%iLR%.e/' % (args.n, args.D, args.bs, args.lr)
    if not os.path.exists(path):
        os.makedirs(path)

    # initialize model
    device = 'cuda:' + str(args.gpu)
    model = AMPS(n=args.n, bond_dim=args.D)
    model = model.to(device)
    sk = SKModel(n=args.n, device=device)

    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    beta = args.beta_init

    # main loop
    while beta <= args.beta_final + 1e-5:
        start_time = time.time()
        f_list = []
        f_std_list = []
        e_list = []
        s_list = []

        for step in range(args.epochs):
            optimizer.zero_grad()
            with torch.no_grad():
                samples = model.sample(args.bs, random_start=True)
            log_prob = model(samples)
            assert not samples.requires_grad
            assert log_prob.requires_grad
            with torch.no_grad():
                samples = samples * 2.0 - 1.0
                energy = -0.5 * torch.sum((samples @ sk.J) * samples, dim=1)
                loss = log_prob + beta * energy
            loss_reinforce = torch.mean(log_prob * (loss - loss.mean()))
            loss_reinforce.backward()
            for p in model.parameters():
                nn.utils.clip_grad_norm_(p, max_norm=1)
            optimizer.step()

            # print
            with torch.no_grad():
                free_energy_mean = loss.mean().item() / beta / args.n
                free_energy_std = loss.std().item() / beta / args.n
                entropy_mean = -1.0 * log_prob.mean().item() / args.n
                energy_mean = energy.mean().item() / args.n
                f_list.append(free_energy_mean)
                f_std_list.append(free_energy_std)
                e_list.append(energy_mean)
                s_list.append(entropy_mean)
                print('%.2f #%d free energy=%.6g std=%.6g energy=%.6g entropy=%.6g' %
                      (beta, step, free_energy_mean, free_energy_std, energy_mean, entropy_mean))

        # logging
        free_energy = np.mean(f_list[-100:])
        f_std = np.mean(f_std_list[-100:])
        e = np.mean(e_list[-100:])
        s = np.mean(s_list[-100:])
        with open(path + 'results.txt', 'a', newline='\n') as f:
            f.write('%.2f %.10f %.10f %.10f %.10f\n' % (beta, free_energy, f_std, e, s))

        # anealling
        beta += args.beta_interval


if __name__ == '__main__':
    main()

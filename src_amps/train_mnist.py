import os
import argparse
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from amps import AMPS, AMPSShare
from utils import train, test, adjust_learning_rate

torch.manual_seed(2020)


def main():
    # training setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='gpu id, default: 0')
    parser.add_argument('--D', type=int, default=30, help='bond dimension, default: 30')
    parser.add_argument('--epochs', type=int, default=100, help='epochs, default: 100')
    parser.add_argument('--bs', type=int, default=100, help='batch size, default: 100')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate, default: 0.1')
    parser.add_argument('--wd', type=float, default=0, help='weight decay, default: 0')
    parser.add_argument('--step_size', type=int, default=20, help='parameter step_size, default: 20')
    parser.add_argument('--gamma', type=float, default=0.1, help='parameter gamma, default: 0.1')
    parser.add_argument('--save-model', action='store_true', default=False, help='save the best model')
    parser.add_argument('--sample-image', action='store_true', default=False, help='sample iamges from model')
    args = parser.parse_args()
    print(args)

    # make dirs
    path = './results/mnist/D%iBS%iLR%.eWD%.eSZ%iGM%.e/' % (args.D, args.bs, args.lr, args.wd, args.step_size, args.gamma)
    if not os.path.exists(path):
        os.makedirs(path)

    # loading data
    mnist_data = np.load('../data/binarized_mnist.npz')
    train_data = torch.from_numpy(mnist_data['train_data'])
    test_data = torch.from_numpy(mnist_data['test_data'])
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.bs, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.bs)

    # initialize model
    device = 'cuda:' + str(args.gpu)
    model = AMPSShare(n=784, bond_dim=args.D)
    model = model.to(device)

    # set optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)

    # main loop
    best_nll = 1000.
    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(args, optimizer, epoch)

        start_time = time.time()
        nll_train = train(args, model, device, train_dataloader, optimizer, epoch)
        nll_test = test(model, device, test_dataloader)

        print('\nEpoch: %.3i\tTrain nll: %.2f\tTest nll: %.2f\tTime: %.2f' %
              (epoch, nll_train.item(), nll_test.item(), time.time() - start_time))

        # save current best model
        if nll_test.item() < best_nll:
            best_nll = nll_test.item()
            if args.save_model:
                torch.save(model.state_dict(), path + 'model.pt')
        print('\nCurrent best test nll: %.2f\n' % best_nll)

        # sample images
        if args.sample_image:
            with torch.no_grad():
                images = model.sample(8).cpu().numpy()
            images = images.reshape(-1, 28, 28)
            plt.figure(figsize=(8, 4))
            for i in range(8):
                plt.subplot(2, 4, i + 1)
                plt.imshow(images[i, :, :], cmap='gray')
                plt.axis('off')
            plt.savefig(path + 'samples-%.3i.png' % epoch, dpi=150)
            plt.close()

        # logging
        with open(path + 'results.txt', 'a', newline='\n') as f:
            f.write('%.3i %.4f %.4f %.4f\n' % (epoch, nll_train.item(), nll_test.item(), best_nll))


if __name__ == '__main__':
    main()

import torch
import torch.nn as nn


def train(args, model, device, train_dataloader, optimizer, epoch):
    model.train()
    nll_train = 0.
    for batch_idx, data in enumerate(train_dataloader):
        data = data.to(device)
        optimizer.zero_grad()
        log_prob = model(data)
        nll_loss = -1.0 * torch.mean(log_prob)
        nll_train += -1.0 * torch.sum(log_prob)
        nll_loss.backward()
        for p in model.parameters():
            nn.utils.clip_grad_norm_(p, max_norm=1)
        optimizer.step()
        print('Train Epoch: %i [%i/%i]\tNLL: %.6f' %
              (epoch, batch_idx * len(data), len(train_dataloader.dataset), nll_loss.item()))
    nll_train = nll_train / len(train_dataloader.dataset)
    return nll_train


def test(model, device, test_dataloader):
    model.eval()
    with torch.no_grad():
        nll_test = 0.
        for batch_idx, data in enumerate(test_dataloader):
            data = data.to(device)
            log_prob = model(data)
            nll_test += -1.0 * torch.sum(log_prob)
    nll_test = nll_test / len(test_dataloader.dataset)
    return nll_test


def adjust_learning_rate(args, optimizer, epoch):
    """
    https://github.com/pytorch/examples/blob/95d5fddfb578674e01802f1db1820d8ac1015f67/imagenet/main.py#L314
    Default: sets the learning rate to the initial LR decayed by 10 every 20 epochs
    """
    lr = args.lr * (args.gamma**(epoch // args.step_size))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

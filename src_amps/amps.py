import torch
import torch.nn as nn
import torch.nn.functional as F


class AMPS(nn.Module):
    def __init__(self, n=20, bond_dim=5, std=1e-8):
        super(AMPS, self).__init__()

        # Initialize AMPS model parameters, which is a (n, n, D, D, 2) tensor
        self.tensors = nn.Parameter(std * torch.randn(n, n, bond_dim, bond_dim, 2))

        # Initialize bias matrix, which is a (D, D, 2) identity matrix
        self.register_buffer('bias_mat', torch.stack([torch.eye(bond_dim), torch.eye(bond_dim)], dim=2))

        # Initialize masks
        self.register_buffer('mask', self.tensors.data.clone())
        self.mask.fill_(1)
        for i in range(n):
            self.mask[i, i + 1:, :, :] = 0

        # Set attributes
        self.n = n
        self.bond_dim = bond_dim
        self.std = std

    def forward(self, data):
        """
        Input: data/spin configurations, shape: (bs, n)
        Output: log prob of each sample, shape: (bs,)
        """

        self.tensors.data *= self.mask

        bs = data.shape[0]

        # local feature map, x_j -> [x_j, 1-x_j]
        embedded_data = torch.stack([data, 1.0 - data], dim=2)  # (bs, n, 2)

        logx_hat = torch.zeros_like(embedded_data)

        logx_hat[:, 0, :] = F.log_softmax((self.tensors[0, 0, :, :, :] + self.bias_mat)[0, 0, :], dim=0)

        mats = torch.einsum('nlri,bi->nblr', self.tensors[:, 0, :, :, :] + self.bias_mat, embedded_data[:, 0, :])
        left_vec = mats[:, :, 0, :].unsqueeze(2)  # (n, bs, 1, D)

        for idx in range(1, self.n):
            # compute p(s_2 | s_1) and so on
            logits = torch.einsum('nblr, nri->nbli', left_vec,
                                  (self.tensors[:, idx, :, :, :] + self.bias_mat)[:, :, 0, :]).squeeze(2)
            logx_hat[:, idx, :] = F.log_softmax(logits[idx, :, :], dim=1)
            mats = torch.einsum('nlri,bi->nblr', self.tensors[:, idx, :, :, :] + self.bias_mat, embedded_data[:,
                                                                                                              idx, :])
            left_vec = torch.einsum('nblr,nbrk->nblk', left_vec, mats)  # (n, bs, 1, D)

        # compute log prob
        log_prob = logx_hat[:, :, 0] * data + logx_hat[:, :, 1] * (1.0 - data)

        return log_prob.sum(-1)

    def sample(self, bs, random_start=False):
        """
        Sample images/spin configurations
        """

        self.tensors.data *= self.mask

        device = self.tensors.device
        samples = torch.empty([bs, self.n], device=device)

        # if random_start = True, force s_1 = -1/+1 randomly
        if random_start:
            samples[:, 0] = torch.randint(2, size=(bs, ), dtype=torch.float, device=device)
        else:
            samples[:, 0] = 0.

        for idx in range(self.n - 1):
            if idx == 0:
                # sample s_2 from p(s_2 | s_1)
                embedded_data = torch.stack([samples[:, 0], 1.0 - samples[:, 0]], dim=1)  # (bs, 2)
                mats = torch.einsum('nlri,bi->nblr', self.tensors[:, 0, :, :, :] + self.bias_mat, embedded_data)
                left_vec = mats[:, :, 0, :].unsqueeze(2)  # (n, bs, 1, D)
                logits = torch.einsum('nblr, nri->nbli', left_vec,
                                      (self.tensors[:, 1, :, :, :] + self.bias_mat)[:, :, 0, :]).squeeze(2)
                samples[:, 1] = torch.bernoulli(torch.softmax(logits[idx + 1, :, :], dim=1)[:, 0])
            else:
                # then sample s_3 from  p(s_3 | s_1, s_2) and so on
                embedded_data = torch.stack([samples[:, idx], 1.0 - samples[:, idx]], dim=1)  # (bs, 2)
                mats = torch.einsum('nlri,bi->nblr', self.tensors[:, idx, :, :, :] + self.bias_mat, embedded_data)
                #                 left_vec = torch.bmm(left_vec, mats)  # (bs, 1, D)
                left_vec = torch.einsum('nblr,nbrk->nblk', left_vec, mats)  # (n, bs, 1, D)
                logits = torch.einsum('nblr,nri->nbli', left_vec,
                                      (self.tensors[:, idx + 1, :, :, :] + self.bias_mat)[:, :, 0, :]).squeeze(2)
                samples[:, idx + 1] = torch.bernoulli(torch.softmax(logits[idx + 1, :, :], dim=1)[:, 0])

        return samples


class AMPSShare(nn.Module):
    def __init__(self, n=784, bond_dim=10, std=1e-8):
        super(AMPSShare, self).__init__()

        # Initialize AMPS model parameters, which is a (n, D, D, 2) tensor
        self.tensors = nn.Parameter(std * torch.randn(n, bond_dim, bond_dim, 2))

        # Initialize bias matrix, which is a (D, D, 2) identity matrix
        self.register_buffer('bias_mat', torch.stack([torch.eye(bond_dim), torch.eye(bond_dim)], dim=2))

        # Set attributes
        self.n = n
        self.bond_dim = bond_dim
        self.std = std

    def forward(self, data):
        """
        Input: data/spin configurations, shape: (bs, n)
        Output: log prob of each sample, shape: (bs,)
        """

        bs = data.shape[0]

        # local feature map, x_j -> [x_j, 1-x_j]
        embedded_data = torch.stack([data, 1.0 - data], dim=2)  # (bs, n, 2)

        logx_hat = torch.zeros_like(embedded_data)
        logx_hat[:, 0, :] = F.log_softmax((self.tensors[0, :, :, :] + self.bias_mat)[0, 0, :], dim=0)

        mats = torch.einsum('lri,bi->blr', self.tensors[0, :, :, :] + self.bias_mat, embedded_data[:, 0, :])
        left_vec = mats[:, 0, :].unsqueeze(1)  # (bs, 1, D)

        for idx in range(1, self.n):
            # compute p(s_2 | s_1) and so on
            logits = torch.einsum('blr, ri->bli', left_vec,
                                  (self.tensors[idx, :, :, :] + self.bias_mat)[:, 0, :]).squeeze(1)
            logx_hat[:, idx, :] = F.log_softmax(logits, dim=1)
            mats = torch.einsum('lri,bi->blr', self.tensors[idx, :, :, :] + self.bias_mat, embedded_data[:, idx, :])
            left_vec = torch.bmm(left_vec, mats)  # (bs, 1, D)

        # compute log prob
        log_prob = logx_hat[:, :, 0] * data + logx_hat[:, :, 1] * (1.0 - data)

        return log_prob.sum(-1)

    def sample(self, bs, random_start=False):
        """
        Sample images/spin configurations
        """

        device = self.tensors.device
        samples = torch.empty([bs, self.n], device=device)

        # if random_start = True, force s_1 = -1/+1 randomly
        if random_start:
            samples[:, 0] = torch.randint(2, size=(bs, ), dtype=torch.float, device=device)
        else:
            samples[:, 0] = 0.

        for idx in range(self.n - 1):
            if idx == 0:
                # sample s_2 from p(s_2 | s_1)
                embedded_data = torch.stack([samples[:, 0], 1.0 - samples[:, 0]], dim=1)  # (bs, 2)
                mats = torch.einsum('lri,bi->blr', self.tensors[0, :, :, :] + self.bias_mat, embedded_data)
                left_vec = mats[:, 0, :].unsqueeze(1)  # (bs, 1, D)
                logits = torch.einsum('blr, ri->bli', left_vec,
                                      (self.tensors[1, :, :, :] + self.bias_mat)[:, 0, :]).squeeze(1)
                samples[:, 1] = torch.bernoulli(torch.softmax(logits, dim=1)[:, 0])
            else:
                # then sample s_3 from  p(s_3 | s_1, s_2) and so on
                embedded_data = torch.stack([samples[:, idx], 1.0 - samples[:, idx]], dim=1)  # (bs, 2)
                mats = torch.einsum('lri,bi->blr', self.tensors[idx, :, :, :] + self.bias_mat, embedded_data)
                left_vec = torch.bmm(left_vec, mats)  # (bs, 1, D)
                logits = torch.einsum('blr, ri->bli', left_vec,
                                      (self.tensors[idx + 1, :, :, :] + self.bias_mat)[:, 0, :]).squeeze(1)
                samples[:, idx + 1] = torch.bernoulli(torch.softmax(logits, dim=1)[:, 0])

        return samples


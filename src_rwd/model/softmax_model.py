import torch
import torch.nn as nn
import torch.nn.functional as F


class ARMPS(nn.Module):
    def __init__(self, n=20, bond_dim=5, feature_dim=2, std=1e-8):
        super(ARMPS, self).__init__()

        # Initialize ARMPS model parameters, which is a (n, n, D, D, d) tensor
        self.tensors = nn.Parameter(std * torch.randn(n, n, bond_dim, bond_dim, feature_dim))

        # Initialize bias matrix, which is a (D, D, d) identity matrix
        self.register_buffer('bias_mat', torch.stack([torch.eye(bond_dim)] * feature_dim, dim=2))

        # We use a mask for parallel MPS contraction
        self.register_buffer('mask', self.tensors.data.clone())
        self.mask.fill_(1)
        for i in range(n):
            self.mask[i, i + 1:, :, :] = 0

        # Set attributes
        self.n = n
        self.bond_dim = bond_dim
        self.feature_dim = feature_dim
        self.std = std

    def forward(self, data):
        """
        data: shape (bs, n)
        return: log probability of each sample
        """
        assert data.dtype == torch.int64
        assert data.max() < self.feature_dim
        bs = data.shape[0]

        self.tensors.data *= self.mask

        # local feature map, one hot encoding
        embedded_data = F.one_hot(data).float()  # (bs, n, feature_dim)
        logx_hat = torch.zeros_like(embedded_data)
        logx_hat[:, 0, :] = F.log_softmax((self.tensors[0, 0, :, :, :] + self.bias_mat)[0, 0, :], dim=0)
        mats = torch.einsum('nlri,bi->nblr', self.tensors[:, 0, :, :, :] + self.bias_mat, embedded_data[:, 0, :])
        left_vec = mats[:, :, 0, :].unsqueeze(2)  # (n, bs, 1, D)

        for idx in range(1, self.n):
            # compute p(s_2 | s_1) and so on
            tmp = torch.einsum('nblr, nri->nbli', left_vec,
                               (self.tensors[:, idx, :, :, :] + self.bias_mat)[:, :, 0, :]).squeeze(2)
            logx_hat[:, idx, :] = F.log_softmax(tmp[idx, :, :], dim=1)
            mats = torch.einsum('nlri,bi->nblr', self.tensors[:, idx, :, :, :] + self.bias_mat, embedded_data[:,
                                                                                                              idx, :])
            left_vec = torch.einsum('nblr,nbrk->nblk', left_vec, mats)  # (n, bs, 1, D)

        # compute log prob
        log_prob = -1.0 * F.nll_loss(logx_hat.permute(0, 2, 1), data, reduction='none')
        return log_prob.sum(-1)

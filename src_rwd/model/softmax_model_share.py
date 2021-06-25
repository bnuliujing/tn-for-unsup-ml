import torch
import torch.nn as nn
import torch.nn.functional as F


class ARMPSShare(nn.Module):
    def __init__(self, n=20, bond_dim=5, feature_dim=2, std=1e-8):
        super(ARMPSShare, self).__init__()

        # Initialize ARMPS model parameters, which is a (n, D, D, d) tensor
        self.tensors = nn.Parameter(std * torch.randn(n, bond_dim, bond_dim, feature_dim))

        # Initialize bias matrix, which is a (D, D, d) identity matrix
        self.register_buffer('bias_mat', torch.stack([torch.eye(bond_dim)] * feature_dim, dim=2))

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

        # local feature map, one hot encoding
        embedded_data = F.one_hot(data).float()  # (bs, n, feature_dim)
        logx_hat = torch.zeros_like(embedded_data)
        logx_hat[:, 0, :] = F.log_softmax((self.tensors[0, :, :, :] + self.bias_mat)[0, 0, :], dim=0)

        mats = torch.einsum('lri,bi->blr', self.tensors[0, :, :, :] + self.bias_mat, embedded_data[:, 0, :])
        left_vec = mats[:, 0, :].unsqueeze(1)  # (bs, 1, D)

        for idx in range(1, self.n):
            # compute p(s_2 | s_1) and so on
            tmp = torch.einsum('blr, ri->bli', left_vec, (self.tensors[idx, :, :, :] + self.bias_mat)[:,
                                                                                                      0, :]).squeeze(1)
            logx_hat[:, idx, :] = F.log_softmax(tmp, dim=1)
            mats = torch.einsum('lri,bi->blr', self.tensors[idx, :, :, :] + self.bias_mat, embedded_data[:, idx, :])
            left_vec = torch.bmm(left_vec, mats)  # (bs, 1, D)

        # compute log prob
        log_prob = -1.0 * F.nll_loss(logx_hat.permute(0, 2, 1), data, reduction='none')

        return log_prob.sum(-1)

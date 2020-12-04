import torch
import math
import numpy as np
from scipy.special import logsumexp


class SKModel():
    def __init__(self, n=20, seed=2050, device='cpu'):
        self.n = n
        self.seed = seed
        self.device = device
        if self.seed > 0:
            torch.manual_seed(self.seed)
        self.J = torch.randn(self.n, self.n) / math.sqrt(self.n)
        self.J = torch.triu(self.J, diagonal=1)
        self.J = self.J + self.J.t()
        self.J_np = self.J.numpy()  # numpy, np.float32
        self.J = self.J.to(self.device)
        print((self.J).mean(), (self.J * self.J).mean())

    def exact(self, beta):
        assert self.n <= 20
        n_total = int(math.pow(2, self.n))
        energy_arr = []
        self.J_np = self.J_np.astype(np.float64)
        for idx in range(n_total):
            s = np.binary_repr(idx, width=self.n)
            s = np.array(list(s)).astype(np.float64) * 2 - 1
            energy_arr.append(-0.5 * s.T @ self.J_np @ s)
        energy_arr = np.array(energy_arr)
        lnz = logsumexp(-beta * energy_arr)
        F = -1.0 * lnz / beta
        prob_arr = np.exp(-1.0 * beta * energy_arr - lnz)
        E = np.sum(prob_arr * energy_arr)
        S = beta * (E - F)
        print('beta: %.2f\tf: %.6f\te: %.6f\ts: %.6f' % (beta, F / self.n, E / self.n, S / self.n))
        return F / self.n, E / self.n, S / self.n

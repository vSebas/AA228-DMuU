import numpy as np

class MDP:
    def __init__(self, gamma=0.95):
        self.gamma = gamma
        self.S = None; self.A = None
        self.T = None; self.R = None
        self.state_to_idx = None; self.idx_to_state = None
        self.action_to_idx = None; self.idx_to_action = None

    def mle(self, data: np.ndarray):
        """Maximum-likelihood model from (s,a,r,sp) tuples."""
        self.S = np.unique(np.concatenate([data[:, 0], data[:, 3]])).astype(np.int64)
        self.A = np.unique(data[:, 1]).astype(np.int64)

        self.state_to_idx = {int(s): i for i, s in enumerate(self.S)}
        self.idx_to_state = {i: s for s, i in self.state_to_idx.items()}
        self.action_to_idx = {int(a): i for i, a in enumerate(self.A)}
        self.idx_to_action = {i: a for a, i in self.action_to_idx.items()}

        S_len, A_len = len(self.S), len(self.A)
        N_sas = np.zeros((S_len, A_len), dtype=np.float64)
        N_sasp = np.zeros((S_len, A_len, S_len), dtype=np.float64)
        R_sum = np.zeros((S_len, A_len), dtype=np.float64)

        for s, a, r, sp in data:
            si = self.state_to_idx[int(s)]
            ai = self.action_to_idx[int(a)]
            spi = self.state_to_idx[int(sp)]
            N_sasp[si, ai, spi] += 1.0
            N_sas[si, ai] += 1.0
            R_sum[si, ai] += float(r)

        self.T = N_sasp / np.maximum(N_sas[:, :, None], 1.0)
        self.R = R_sum / np.maximum(N_sas, 1.0)

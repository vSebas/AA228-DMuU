#!/usr/bin/env python3

import numpy as np

class MDP:
    def __init__(self, gamma=0.95):
        # mappings
        self.state_to_idx = None
        self.idx_to_state = None
        self.action_to_idx = None
        self.idx_to_action = None
        
        # model in INDEX space
        self.S = None
        self.A = None
        self.T = None # shape (S,A,S)
        self.R = None # shape (S,A)
        self.gamma = gamma

    # maximum likelihood estimate
    def mle(self, data):
        self.S = np.unique(np.concatenate([data[:,0], data[:,3]])) # (s, sp)
        self.A = np.unique(data[:,1])

        self.state_to_idx = {s:i for i, s in enumerate(self.S)}
        self.action_to_idx = {a:i for i, a in enumerate(self.A)}

        self.idx_to_state = {i:s for s,i in self.state_to_idx.items()}
        self.idx_to_action = {i:a for a,i in self.action_to_idx.items()}

        S_len, A_len = len(self.S), len(self.A)

        N_sas  = np.zeros((S_len, A_len))        # sum over sp of transition counts
        N_sasp = np.zeros((S_len, A_len, S_len)) # transition counts
        R_sum  = np.zeros((S_len, A_len))        # reward totals

        for s, a, r, sp in data:
            si  = self.state_to_idx[s]
            ai  = self.action_to_idx[a]
            spi = self.state_to_idx[sp]
            N_sasp[si, ai, spi] += 1
            N_sas[si, ai] += 1
            R_sum[si, ai] += r

        self.T = N_sasp / np.maximum(N_sas[:, :, None], 1) # (S,A,S)
        self.R = R_sum / np.maximum(N_sas[:, :,], 1)       # (S,A)
#!/usr/bin/env python3

import numpy as np

class QLearning:
    def __init__(self, S, A, alpha=0.2, gamma=1):
        self.S = S
        self.A = A
        self.Q = np.zeros((S.shape[0], A.shape[0]))
        self.gamma = gamma
        self.alpha = alpha

        self.state_to_idx = {s:i for i,s in enumerate(self.S)}
        self.action_to_idx = {a:i for i,a in enumerate(self.A)}
        
        self.idx_to_state = {i:s for s,i in self.state_to_idx.items()}
        self.idx_to_action = {i:a for a,i in self.action_to_idx.items()}

    def Qlearning_update(self, s, a, r, s_next):
        return self.Q[s,a] + self.alpha*(r + self.gamma*np.max(self.Q[s_next]) - self.Q[s,a])
    
    def simulate(self, data, k_max):
        
        for k in range(k_max):

            for s,a,r,sp in data:
                s_i = self.state_to_idx[s]
                sp_i = self.state_to_idx[sp]
                a_i = self.action_to_idx[a]
                self.Q[s_i,a_i] = self.Qlearning_update(s_i, a_i, r, sp_i)

        max_actions = np.argmax(self.Q, axis=1)  # axis=1 regresa idx de la columna en la que el elemento max de cada row se encuentra.

        pi = {self.idx_to_state[s]:self.idx_to_action[max_actions[s]] for s in range(self.S.shape[0])}

        return self.Q.max(axis=1), pi

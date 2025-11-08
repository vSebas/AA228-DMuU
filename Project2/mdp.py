#!/usr/bin/env python3

class MDP:
    def __init__(self, S, A, T, R, gamma):
        self.S = S          # state space
        self.A = A          # action space
        self.T = T          # transition function
        self.R = R          # reward function
        self.gamma = gamma  # discount factor
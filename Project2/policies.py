#!/usr/bin/env python3
from mdp import MDP, PolicyMDP
import numpy as np

# for policy evaluation, bellman update
# Q_U(s,a) one-step lookahead
def lookahead(MDP, U, s, a): # iterative
    # If reward depends only on (s,a), keep mdp.R(s,a); otherwise use mdp.R(s,a,s_next)
    return MDP.R(s, a) + MDP.gamma * sum(MDP.T(s, a, s_next) * U[s_next] for s_next in MDP.S)

# Given a state, and future utilities, improve the value function
# (TU)(s) = max_a Q_U(s,a)
def bellman_backup(MDP, U, s):
    return max([lookahead(MDP, U, s, a) for a in MDP.A])

# value function policies
# π_U(s) = argmax_a Q_U(s,a)
def greedy_policy(MDP, U):
    pi = {}
    for s in MDP.S:
        vals = [lookahead(MDP, U, s, a) for a in MDP.A]
        pi[s] = MDP.A[np.argmax(vals)]

    return pi

# Repeated iteration of the bellman backup is guaranteed to converge to the optimal value function
def value_iteration(k_max, MDP):
    U = {s: 0 for s in MDP.S}

    # Keep backing up all states until the value function converges.
    for _ in range(k_max):
        U = {s:bellman_backup(MDP, U, s) for s in MDP.S}
    
    # Once you have a stable (or approximately optimal) U, the greedy policy uses the same lookahead expression to choose the argmax action:
    # extract greedy policy
    return U, greedy_policy(MDP, U)


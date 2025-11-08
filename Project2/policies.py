#!/usr/bin/env python3

import numpy as np
from mdp import MDP

def lookahead(MDP, U):
    """
    Return Q-function after executing one-step bellman lookahead
    Bellman update for iterative policy evaluation

    Parameters
    ----------
    MDP: MDP Class
        problem to solve
    U: array_like
        previous value function
        Vector of |S| dimensions

    Returns
    -------
    Q : ndarray
        Array with updated action value function of shape |S|x|A|
    """
    return MDP.R + MDP.gamma*MDP.T@U    # (S,A) + (S,A,S)@(S,) -> (S,A) shape

def bellman_backup(MDP, U, k_max, eps=1e-3):
    """
    Improve value function by applying a bellman backup/update for iterative policy evaluation

    Parameters
    ----------
    MDP: MDP Class
        problem to solve
    U: array_like
        previous value function
        Vector of |S| dimensions
    k_max: integer
        Max number of iterations to compute
    eps: float
        max deviation of the estimated value function from the optimal value function
        (final estimate, after convergence)
        delta is computed through eps to stop iterations

    Returns
    -------
    U : ndarray
        Array with updated max value function of shape |S|
    """

    # Keep backing up all states until the value function converges.

    for k in range(k_max):
        U_old = U
        # Compute one lookahead
        Q = lookahead(MDP, U)

        # Per state, get the max value function out of all actions (cols)
        U = Q.max(axis=1)   # (S,) shape

        residual = np.max(np.abs(U - U_old))      # Bellman residual
        delta = eps * (1.0 - MDP.gamma) / MDP.gamma # Book shows e = delta*gamma/(1-gamma), just solve for delta
        if residual < delta:
            print(f"Stopped iterations at {k:5d}, with delta={delta:.3e}")
            break


    return U

def value_iteration(MDP, k_max=10000):
    """
    Perform value iteration to converge to the optimal value function.
    After computing the optimal value function, the greedy value function is obtained
    Returns optimal values and policies per state

    Parameters
    ----------
    MDP: MDP Class
        problem to solve
    k_max: integer
        Max number of iterations to compute

    Returns
    -------
    U : ndarray
        Array with optmial max value function of shape |MDP.S|
    pi: ndarray
        Array with optimal value function policies of shape |MDP.S| 
    
    """
    U = np.zeros_like(MDP.S)
    U = bellman_backup(MDP, U, k_max)

    # Extract greedy value function policy
        # Once you have a stable (or approximately optimal) U, 
        # the greedy policy uses the same lookahead expression to choose the argmax action:
    Q = lookahead(MDP, U)
    max_actions = np.argmax(Q, axis=1)  # axis=1 regresa idx de la columna en la que el elemento max de cada row se encuentra. 
    pi = {MDP.idx_to_state[s]:MDP.idx_to_action[max_actions[s]] for s in range(MDP.S.shape[0])}

    return U, pi
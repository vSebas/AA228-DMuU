import numpy as np

def lookahead(mdp, U):
    """One-step lookahead for the Bellman *optimality* backup."""
    return mdp.R + mdp.gamma * (mdp.T @ U)   # (S,A) + (S,A,S)@(S,) -> (S,A)

def bellman_backup(mdp, U, k_max, eps=1e-3):
    """Value-iteration style optimality backup with epsilon stopping rule."""
    for k in range(k_max):
        U_old = U.copy()
        Q = lookahead(mdp, U_old)
        U = Q.max(axis=1)
        residual = np.max(np.abs(U - U_old))
        delta = eps * (1.0 - mdp.gamma) / mdp.gamma
        if residual < delta:
            print(f"VI stop at iter {k} (residual {residual:.3e} < {delta:.3e})")
            break
    return U

def value_iteration(mdp, k_max=10_000, eps=1e-3):
    U = np.zeros(len(mdp.S), dtype=float)
    U = bellman_backup(mdp, U, k_max, eps)
    Q = lookahead(mdp, U)
    a_idx = np.argmax(Q, axis=1)
    pi = {mdp.idx_to_state[i]: mdp.idx_to_action[int(a_idx[i])] for i in range(len(mdp.S))}
    return U, pi

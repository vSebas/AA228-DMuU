import numpy as np

from scipy.spatial import cKDTree

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

def knn_fill_states(Q_full, S_seen, base=(500, 100), fallback_rows=None):
    """
    Fill rows of Q_full (states never seen) using nearest-neighbor in (pos, vel).
    - base = (n_pos_bins, n_vel_bins)
    - S_seen: raw ids of seen states (1-based)
    - fallback_rows: Q rows aligned with S_seen order (Q_seen)
    """
    nS_full = Q_full.shape[0]
    seen_set = set(int(s) for s in S_seen)

    pos_seen = ((S_seen - 1) % base[0]).astype(np.int64)
    vel_seen = ((S_seen - 1) // base[0]).astype(np.int64)
    tree = cKDTree(np.column_stack([pos_seen, vel_seen]))

    for sid in range(1, nS_full + 1):
        if sid not in seen_set:
            pos = (sid - 1) % base[0]
            vel = (sid - 1) // base[0]
            _, j = tree.query([pos, vel])  # index in S_seen / fallback_rows
            Q_full[sid - 1, :] = fallback_rows[j, :]
    return Q_full

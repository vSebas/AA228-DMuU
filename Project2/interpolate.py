import numpy as np
from scipy.spatial import cKDTree

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

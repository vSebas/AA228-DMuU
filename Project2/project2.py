#!/usr/bin/env python3

import numpy as np
import util
from QLearning import QLearning
from mdp import MDP
from policies import value_iteration
from scipy.spatial import cKDTree

def run_small():
    data = util.load_dataset("datasets/small.csv")

    mdp = MDP()
    mdp.mle(data)

    # Transistion models are conditional probabilities distributions, so they must sum up to one over all states
    print("Row sums of T by (s,a) should be 1 or 0 if unseen:")
    row_sums = mdp.T.sum(axis=2)
    print("min row sum:", row_sums[row_sums>0].min(), "max row sum:", row_sums.max())

    U, pi = value_iteration(mdp)

    util.save("small", mdp.state_to_idx, U, pi)

def run_medium():
    data = util.load_dataset("datasets/medium.csv")

    S_raw = np.unique(np.concatenate([data[:,0], data[:,3]])).astype(int) # (s, sp)
    A = np.unique(data[:,1]).astype(int)

    model = QLearning(S_raw, A)
    _, _ = model.simulate(data, 1000)      # runs the updates
    Q_seen = np.asarray(model.Q)     # <-- grab the table here

    # Implement k-nearest to fill missing states
    nS_full = 50000
    nA = len(A)
    Q_full = np.zeros((nS_full, nA), dtype=float)

    # Map raw id -> row in Q_seen
    st2row = model.state_to_idx  # dict: raw_id -> 0..|S_raw|-1

    # 1) Copy learned Q into the full 50k table at raw-id positions
    for s in S_raw:
        Q_full[s - 1, :] = Q_seen[st2row[s], :]

    # 2) Build KDTree over visited coords (pos, vel) in the SAME ORDER as S_raw
    #    coords[k] must align with Q_seen[k]
    pos_seen = (S_raw - 1) % 500
    vel_seen = (S_raw - 1) // 500
    coords_seen = np.column_stack([pos_seen, vel_seen])  # shape (|S_raw|, 2)
    tree = cKDTree(coords_seen)

    # 3) Fill missing states via nearest neighbor (raw-id space)
    #    Unseen mask: rows in Q_full still all zeros AND raw id not in st2row
    seen_set = set(S_raw.tolist())
    for sid in range(1, nS_full + 1):
        if sid not in seen_set:
            pos = (sid - 1) % 500
            vel = (sid - 1) // 500
            _, nearest = tree.query([pos, vel])        # index into S_raw / Q_seen
            Q_full[sid - 1, :] = Q_seen[nearest, :]

    # 4) Greedy policy. If grader expects 1..7, add +1 offset.
    pi_full = np.argmax(Q_full, axis=1) + 1
    # pi = {model.idx_to_state[s]:model.idx_to_action[pi_full[s]] for s in range(model.S.shape[0])}

    rows = []
    for s_id in range(nS_full):
        a_id = pi_full[s_id]
        u_val = np.argmax(Q_full, axis=1)[s_id]
        rows.append([int(s_id)+1, float(u_val), int(a_id)])

    out = np.array(rows, dtype=float)

    np.savetxt(
        "output/results_medium.txt",
        out,
        fmt=["%d", "%.8f", "%d"],
        header="state\tvalue*\taction*",
        comments=""
    )

    np.savetxt(
        "output/medium.policy",
        out[:,2],
        fmt=["%d"]
    )

def main():
    # run_small()
    run_medium()


if __name__ == "__main__":
    main()

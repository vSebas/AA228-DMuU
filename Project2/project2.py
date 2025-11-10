#!/usr/bin/env python3
import argparse
import numpy as np

from mdp import MDP
from policies import value_iteration
from qlearning import QLearning
from interpolate import knn_fill_states
from utils import (
    load_dataset,
    write_policy_actions_only,
    write_results_table,
)

# Small: Model-based MLE + Value Iteration
def run_small(data_path="datasets/small.csv"):
    data = load_dataset(data_path)

    mdp = MDP(gamma=0.95)
    mdp.mle(data)

    # each (s,a) row in T should sum to 1 if seen
    # row_sums = mdp.T.sum(axis=2)
    # seen = row_sums > 0
    # if np.any(seen):
    #     print("T row-sums (seen): min=", row_sums[seen].min(), "max=", row_sums[seen].max())

    U, pi = value_iteration(mdp, k_max=10_000)

    # small: write both policy and diagnostics
    actions = np.array([pi[mdp.idx_to_state[i]] for i in range(len(mdp.S))], dtype=int)
    write_policy_actions_only(actions, f"output/small.policy")
    write_results_table(
        state_ids=mdp.S.astype(int),
        values=U,
        actions=actions,
        path=f"output/results_small.txt",
        header="state\tvalue*\taction*",
    )
    print("small: policy and results saved")

# Medium: Q-learning + K-nearest neighbors
def run_medium(data_path="datasets/medium.csv"):
    data = load_dataset(data_path).astype(float)

    # actions: from 1..7 to 0..6 for indexing
    data[:, 1] -= 1
    data = data.astype(np.int64)

    S_raw = np.unique(np.concatenate([data[:, 0], data[:, 3]])).astype(np.int64)
    A = np.arange(7, dtype=np.int64)  # 0..6

    agent = QLearning(S_raw, A)
    Q_seen = agent.simulate(data, n_passes=300, gamma=1.0, alpha=None, tol=1e-5)

    # Build full 50k×7 Q and copy observed rows
    nS_full, nA = 50_000, 7
    Q_full = np.zeros((nS_full, nA), dtype=float)
    N_full = np.zeros((nS_full, nA), dtype=np.int64)

    sid2row = agent.state_to_idx
    for s in S_raw:
        i_full, i_seen = int(s) - 1, sid2row[int(s)]
        Q_full[i_full, :] = Q_seen[i_seen, :]
        N_full[i_full, :] = agent.N[i_seen, :]

    # Fill UNSEEN states via KNN in (pos,vel) grid
    Q_full = knn_fill_states(Q_full, S_raw, base=(500, 100), fallback_rows=Q_seen)

    # Greedy policy with masking of never-tried actions (only on seen states)
    pi0 = np.empty(nS_full, dtype=np.int64)
    seen_set = set(int(s) for s in S_raw)
    for i in range(nS_full):
        if (i + 1) in seen_set:
            q = Q_full[i, :].copy()
            mask = (N_full[i, :] == 0)
            if mask.all():
                pi0[i] = int(np.argmax(q))
            else:
                q[mask] = -1e12
                pi0[i] = int(np.argmax(q))
        else:
            pi0[i] = int(np.argmax(Q_full[i, :]))  # NN-filled row

    # back to 1..7 for export
    pi_out = (pi0 + 1).astype(np.int64)
    write_policy_actions_only(pi_out, f"output/medium.policy")

    # diagnostics (value* = max_a Q)
    U_star = np.max(Q_full, axis=1)
    write_results_table(
        state_ids=np.arange(1, nS_full + 1),
        values=U_star,
        actions=pi_out,
        path=f"output/results_medium.txt",
        header="state\tvalue*\taction*",
    )
    print("medium: policy and results saved")

# Large: Q-learning
def run_large(data_path="datasets/large.csv",
              n_states_full=302_020, n_actions_full=9, gamma=0.95):
    data = load_dataset(data_path)

    # Normalize actions to 0..(n_actions_full-1) if they’re 1..n
    a_min, a_max = int(data[:, 1].min()), int(data[:, 1].max())
    if a_min == 1 and a_max == n_actions_full:
        data[:, 1] -= 1
    data = data.astype(np.int64)

    S_raw = np.unique(np.concatenate([data[:, 0], data[:, 3]])).astype(np.int64)
    A = np.arange(n_actions_full, dtype=np.int64)  # 0..8

    agent = QLearning(S_raw, A)
    Q_seen = agent.simulate(data, n_passes=60, gamma=gamma, alpha=None, tol=1e-5)

    # Full table & copy
    Q_full = np.zeros((n_states_full, n_actions_full), dtype=float)
    sid2row = agent.state_to_idx
    for s in S_raw:
        Q_full[int(s) - 1, :] = Q_seen[sid2row[int(s)], :]

    # Greedy policy with masking of never-tried actions on seen states
    pi0 = np.empty(n_states_full, dtype=np.int64)
    seen = set(int(s) for s in S_raw)
    for i in range(n_states_full):
        if (i + 1) in seen:
            counts = agent.N[sid2row[i + 1], :]
            q = Q_full[i, :].copy()
            if counts.sum() > 0:
                q[counts == 0] = -1e12
            pi0[i] = int(np.argmax(q))
        else:
            # unseen state → safest fallback = most frequent action in data
            most_common = int(np.bincount(data[:, 1], minlength=n_actions_full).argmax())
            pi0[i] = most_common

    pi_out = (pi0 + 1).astype(np.int64)
    write_policy_actions_only(pi_out, f"output/large.policy")

    # optional diagnostics
    U_star = np.max(Q_full, axis=1)
    write_results_table(
        state_ids=np.arange(1, n_states_full + 1),
        values=U_star,
        actions=pi_out,
        path=f"output/results_large.txt",
        header="state\tvalue*\taction*",
    )
    print("large: policy and results saved")

def main():
    run_small("datasets/small.csv")
    run_medium("datasets/medium.csv")
    run_large("datasets/large.csv")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse, os, sys
import numpy as np
from collections import defaultdict
import random

# -------------------------
# Batch Q-learning (tabular)
# -------------------------
class QLearningLarge:
    def __init__(self, S_raw, A_ids):
        self.S_raw = np.asarray(S_raw, dtype=np.int64)          # raw state ids seen
        self.A_ids = np.asarray(A_ids, dtype=np.int64)          # raw action ids seen (0..8 after normalization)
        self.nS = len(self.S_raw)
        self.nA = len(self.A_ids)
        self.sid2row = {int(s): i for i, s in enumerate(self.S_raw)}

        # Q and visit counts
        self.Q = np.zeros((self.nS, self.nA), dtype=np.float32)
        self.N = np.zeros((self.nS, self.nA), dtype=np.int64)

    def simulate(self, data, n_passes=50, gamma=0.95, tol=1e-5, shuffle_each=True):
        """
        Offline Q-learning with alpha(s,a)=1/N(s,a). No terminals assumed for LARGE.
        data: ndarray/list of tuples (s,a,r,sp) with actions normalized to 0..8
        """
        # make a light list of tuples for cheap shuffles
        data = list(map(lambda t: (int(t[0]), int(t[1]), float(t[2]), int(t[3])), data))

        for _ in range(n_passes):
            if shuffle_each:
                random.shuffle(data)

            max_abs_delta = 0.0
            for (s, a, r, sp) in data:
                i = self.sid2row[s]
                j = self.sid2row.get(sp, None)
                # bootstrap always (no terminals in LARGE per spec)
                max_next = 0.0 if j is None else float(np.max(self.Q[j, :]))
                target   = r + gamma * max_next
                delta    = target - self.Q[i, a]

                self.N[i, a] += 1
                step = 1.0 / self.N[i, a]          # decaying α
                self.Q[i, a] += step * delta

                if abs(delta) > max_abs_delta:
                    max_abs_delta = abs(delta)

            if max_abs_delta < tol:
                break

        return self.Q  # (nS, nA)

# -------------------------
# Utilities
# -------------------------
def load_large_csv(path):
    # Expect CSV with columns s,a,r,sp. Some dumps are ints+floats mixed—load as float then cast.
    raw = np.loadtxt(path, delimiter=",", skiprows=1) if path.endswith(".csv") else np.loadtxt(path, delimiter=",")
    # columns: s, a, r, sp
    s  = raw[:, 0].astype(np.int64)
    a  = raw[:, 1].astype(np.int64)
    r  = raw[:, 2].astype(np.float32)
    sp = raw[:, 3].astype(np.int64)
    data = np.stack([s, a, r, sp], axis=1)
    return data

def write_policy_actions_only(actions_1based, outfile):
    # Exactly one integer per line, newline-terminated
    with open(outfile, "w") as f:
        for a in actions_1based:
            f.write(f"{int(a)}\n")

def write_policy_sid_action(actions_1based, outfile):
    # "state_id,action" per line, state ids start at 1
    with open(outfile, "w") as f:
        for sid, a in enumerate(actions_1based, start=1):
            f.write(f"{sid},{int(a)}\n")

# -------------------------
# Main pipeline
# -------------------------
def run_large(
    large_csv="datasets/large.csv",
    out_policy="large.policy",
    actions_only=True,
    n_states_full=302_020,
    n_actions_full=9,
    gamma=0.95,
    n_passes=50,
    tol=1e-5,
    seed=0
):
    rng = np.random.default_rng(seed)
    random.seed(seed)

    # 1) Load dataset
    data = load_large_csv(large_csv)

    # 2) Normalize actions to 0..(n_actions_full-1) if dataset is 1..9
    #    Detect range and shift only if needed.
    a_min, a_max = int(data[:,1].min()), int(data[:,1].max())
    if a_min == 1 and a_max == n_actions_full:
        data[:,1] -= 1  # now 0..8

    # 3) Build state/action universes from data
    S_raw = np.unique(np.concatenate([data[:,0], data[:,3]])).astype(np.int64)
    A_ids = np.arange(n_actions_full, dtype=np.int64)  # 0..8

    # 4) Train tabular offline Q-learning
    agent = QLearningLarge(S_raw, A_ids)
    Q_seen = agent.simulate(data, n_passes=n_passes, gamma=gamma, tol=tol, shuffle_each=True)

    # 5) Build full Q (302020 x 9), fill observed rows; unseen states → fallback
    Q_full = np.zeros((n_states_full, n_actions_full), dtype=np.float32)
    seen = set(int(s) for s in S_raw)
    sid2row = agent.sid2row
    for s in S_raw:
        Q_full[s - 1, :] = Q_seen[sid2row[int(s)], :]

    # For completely unseen states, choose a conservative default action (e.g., most frequent action in data)
    # Compute most frequent action in the dataset (after normalization)
    most_common_a = int(np.bincount(data[:,1].astype(np.int64), minlength=n_actions_full).argmax())
    if most_common_a < 0 or most_common_a >= n_actions_full: most_common_a = 0

    # 6) Greedy policy with masking of never-tried actions (on seen states)
    pi_full_0based = np.empty(n_states_full, dtype=np.int64)
    for i in range(n_states_full):
        if (i + 1) in seen:
            counts = agent.N[sid2row[i + 1], :]
            if counts.sum() == 0:
                # seen state but no actions updated (rare) → argmax as-is
                pi_full_0based[i] = int(np.argmax(Q_full[i, :]))
            else:
                q = Q_full[i, :].copy()
                q[counts == 0] = -1e12  # mask never-tried actions in that state
                pi_full_0based[i] = int(np.argmax(q))
        else:
            # totally unseen state → fall back to most common action
            pi_full_0based[i] = most_common_a

    # 7) Back to 1..9 for export
    pi_out = (pi_full_0based + 1).astype(np.int64)

    # 8) Write policy in the format your FAQ requires
    if actions_only:
        write_policy_actions_only(pi_out, out_policy)
    else:
        write_policy_sid_action(pi_out, out_policy)

    # 9) Sanity checks
    # - correct length
    n_lines = sum(1 for _ in open(out_policy, "r"))
    assert n_lines == n_states_full, f"policy lines {n_lines} != {n_states_full}"
    # - correct action range
    assert 1 <= pi_out.min() and pi_out.max() <= n_actions_full, "actions out of range"

    print(f"Wrote {out_policy} with {n_lines} lines.")

# CLI
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="datasets/large.csv", help="path to large.csv")
    p.add_argument("--out", default="large.policy", help="output policy filename")
    p.add_argument("--actions_only", action="store_true", help="write one action per line (default)")
    p.add_argument("--sid_action",   action="store_true", help="write 'state,action' per line")
    p.add_argument("--passes", type=int, default=50)
    p.add_argument("--tol", type=float, default=1e-5)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    actions_only = True
    if args.sid_action: actions_only = False
    if args.actions_only: actions_only = True
    run_large(
        large_csv=args.csv,
        out_policy=args.out,
        actions_only=actions_only,
        n_passes=args.passes,
        tol=args.tol,
        gamma=args.gamma,
        seed=args.seed
    )

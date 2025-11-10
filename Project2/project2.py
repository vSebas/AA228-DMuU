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
    data = util.load_dataset("datasets/medium.csv").astype(float)

    # normalize columns: convert actions 1..7 → 0..6
    data[:, 1] -= 1
    data = data.astype(np.int64)

    S_raw = np.unique(np.concatenate([data[:,0], data[:,3]])).astype(np.int64)
    A     = np.unique(data[:,1]).astype(np.int64)     # expect 7 actions: 0..6 or 1..7
    nA    = len(A)

    # Train tabular batch Q
    model = QLearning(S_raw, A)
    Q_seen = model.simulate(data, n_passes=200, gamma=1.0, alpha=None, tol=1e-5)

    # Build full (50k, 7) table and a matching visit-count table
    nS_full = 50000
    Q_full  = np.zeros((nS_full, nA), dtype=float)
    N_full  = np.zeros((nS_full, nA), dtype=np.int64)

    st2row = model.state_to_idx
    for s in S_raw:
        i_full = int(s) - 1
        i_seen = st2row[int(s)]
        Q_full[i_full, :] = Q_seen[i_seen, :]
        N_full[i_full, :] = model.N[i_seen, :]

    # Nearest-neighbor fill for states never seen
    pos_seen = ((S_raw - 1) % 500).astype(np.int64)
    vel_seen = ((S_raw - 1) // 500).astype(np.int64)
    tree     = cKDTree(np.column_stack([pos_seen, vel_seen]))
    seen_set = set(int(s) for s in S_raw)

    for sid in range(1, nS_full + 1):
        if sid not in seen_set:
            pos = (sid - 1) % 500
            vel = (sid - 1) // 500
            _, j = tree.query([pos, vel])   # j indexes into S_raw/Q_seen order
            Q_full[sid - 1, :] = Q_seen[j, :]     # counts remain 0 (unseen state)

    # Greedy policy with action masking:
    # if a state-action was never visited, don’t let it win the argmax unless all are unvisited.
    pi_full = np.empty(nS_full, dtype=np.int64)
    for i in range(nS_full):
        counts = N_full[i, :]
        if counts.sum() == 0:
            # totally unseen state → fall back to NN-filled Q row
            a_idx = int(np.argmax(Q_full[i, :]))
        else:
            q = Q_full[i, :].copy()
            q[counts == 0] = -1e12  # mask never-tried actions
            a_idx = int(np.argmax(q))
        pi_full[i] = a_idx

    # If the grader expects 1..7, offset here:
    pi_out = (pi_full + 1).astype(np.int64)

    # True value*: max_a Q(s,a)  (NOT the action index)
    U_star = np.max(Q_full, axis=1)

    # --- WRITE FILES ---
    # policy: EXACTLY one action per line (1..7), 50,000 lines, no headers.
    policy_path = "output/medium.policy"
    with open(policy_path, "w") as f:
        for a in pi_out:                 # pi_out has length 50_000
            f.write(f"{int(a)}\n")

    # Optional: results file for your own diagnostics (not used by grader)
    results_path = "output/results_medium.txt"
    with open(results_path, "w") as f:
        f.write("state\tvalue*\taction*\n")
        for sid in range(1, nS_full + 1):
            f.write(f"{sid}\t{U_star[sid-1]:.6f}\t{int(pi_out[sid-1])}\n")

    # Sanity checks
    import subprocess, shlex
    n_lines = int(subprocess.check_output(shlex.split(f"wc -l {policy_path}")).split()[0])
    assert n_lines == 50000, n_lines
    assert 1 <= pi_out.min() and pi_out.max() <= 7


def main():
    # run_small()
    run_medium()


if __name__ == "__main__":
    main()

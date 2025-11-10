import numpy as np
import os

def load_dataset(csv_path):
    """Load (s,a,r,sp) with header"""
    with open(csv_path, "r") as fh:
        headers = fh.readline().strip().split(",")
    cols = [i for i, h in enumerate(headers) if h.startswith(("s", "a", "r", "sp"))]
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1, usecols=cols)
    if data.ndim == 1:
        data = np.expand_dims(data, 0)
    return data

def write_policy_actions_only(actions_1based, path):
    """Write one integer action per line (no header)."""
    with open(path, "w") as f:
        for a in actions_1based:
            f.write(f"{int(a)}\n")
    # sanity
    n_lines = sum(1 for _ in open(path, "r"))
    assert n_lines == len(actions_1based), f"line count {n_lines} != {len(actions_1based)}"
    assert 1 <= int(np.min(actions_1based)) and int(np.max(actions_1based)) >= 1

def write_results_table(state_ids, values, actions, path, header="state\tvalue*\taction*"):
    rows = np.column_stack([state_ids, values, actions])
    np.savetxt(path, rows, fmt=["%d", "%.8f", "%d"], header=header, comments="")

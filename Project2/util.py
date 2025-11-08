import matplotlib.pyplot as plt
import numpy as np

def load_dataset(csv_path):
    """Load dataset from a CSV file.

    Args:
         csv_path: Path to CSV file containing dataset.

    Returns:
        mdp: Numpy array of MDP tuple (s, a, r, sp).
    """

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    mdp_tuple_cols = [i for i in range(len(headers)) if headers[i].startswith(('s','a','r','sp'))]
    inputs = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=mdp_tuple_cols)

    if inputs.ndim == 1:
        inputs = np.expand_dims(inputs, -1)

    return inputs


def plot(x, y, theta, save_path, correction=1.0):
    """Plot dataset and fitted logistic regression parameters.

    Args:
        x: Matrix of training examples, one per row.
        y: Vector of labels in {0, 1}.
        theta: Vector of parameters for logistic regression model.
        save_path: Path to save the plot.
        correction: Correction factor to apply, if any.
    """
    # Plot dataset
    plt.figure()
    plt.plot(x[y == 1, -2], x[y == 1, -1], 'bx', linewidth=2)
    plt.plot(x[y == 0, -2], x[y == 0, -1], 'go', linewidth=2)

    # Plot decision boundary (found by solving for theta^T x = 0)
    x1 = np.arange(min(x[:, -2]), max(x[:, -2]), 0.01)
    x2 = -(theta[0] / theta[2] + theta[1] / theta[2] * x1
           + np.log((2 - correction) / correction) / theta[2])
    plt.plot(x1, x2, c='red', linewidth=2)
    plt.xlim(x[:, -2].min()-.1, x[:, -2].max()+.1)
    plt.ylim(x[:, -1].min()-.1, x[:, -1].max()+.1)

    # Add labels and save to disk
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.savefig(save_path)

def save(mapping, U, pi):
    # === Combine policy and value into one output file ===
    rows = []

    # Ensure deterministic ordering by sorting state IDs
    for s_id in sorted(pi.keys()):
        a_id = pi[s_id]
        u_val = U[mapping[s_id]]
        rows.append([int(s_id), float(u_val), int(a_id)])

    out = np.array(rows, dtype=float)

    np.savetxt(
        "results_small.txt",
        out,
        fmt=["%d", "%.8f", "%d"],
        header="state\tvalue*\taction*",
        comments=""
    )

    np.savetxt(
        "small.policy",
        out[:,2],
        fmt=["%d"]
    )
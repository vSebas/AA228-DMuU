import numpy as np
import random

class QLearning:
    """Batch (offline) tabular Q-learning with α(s,a)=1/N(s,a)."""
    def __init__(self, S_raw, A_ids):
        self.S_raw = np.asarray(S_raw, dtype=np.int64)
        self.A_ids = np.asarray(A_ids, dtype=np.int64)
        self.nS = len(self.S_raw); self.nA = len(self.A_ids)
        self.state_to_idx = {int(s): i for i, s in enumerate(self.S_raw)}
        self.Q = np.zeros((self.nS, self.nA), dtype=float)
        self.N = np.zeros((self.nS, self.nA), dtype=np.int64)

    def simulate(self, data, n_passes=200, gamma=1.0, alpha=None, tol=1e-5, shuffle_each=True, terminal_states=None):
        if terminal_states is None: terminal_states = set()
        data = list(map(lambda t: (int(t[0]), int(t[1]), float(t[2]), int(t[3])), data))

        for _ in range(n_passes):
            if shuffle_each: random.shuffle(data)
            max_abs_delta = 0.0

            for (s, a, r, sp) in data:
                i = self.state_to_idx[s]
                j = self.state_to_idx.get(sp, None)

                if (sp in terminal_states) or (j is None):
                    max_next = 0.0
                else:
                    max_next = float(np.max(self.Q[j, :]))

                target = r + gamma * max_next
                delta = target - self.Q[i, a]

                if alpha is None:
                    self.N[i, a] += 1
                    step = 1.0 / self.N[i, a]
                else:
                    step = alpha

                self.Q[i, a] += step * delta
                if abs(delta) > max_abs_delta: max_abs_delta = abs(delta)

            if max_abs_delta < tol: break

        return self.Q

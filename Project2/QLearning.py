import numpy as np
import random

class QLearning:
    def __init__(self, S_raw, A):
        self.S_raw = np.asarray(S_raw, dtype=np.int64)            # raw ids seen in data
        self.A = np.asarray(A, dtype=np.int64)                    # e.g., 0..6
        self.nS = len(self.S_raw)
        self.nA = len(self.A)
        self.state_to_idx = {int(s): i for i, s in enumerate(self.S_raw)}
        self.Q = np.zeros((self.nS, self.nA), dtype=float)
        self.N = np.zeros((self.nS, self.nA), dtype=np.int64)     # visit counts

    def simulate(
        self,
        data,
        n_passes=100,
        gamma=1.0,
        alpha=None,                    # if None → use 1/N(s,a)
        tol=1e-5,                      # early stop threshold on max |ΔQ|
        shuffle_each_pass=True,
        terminal_states=None           # set of raw state ids; if s' ∈ terminal → no bootstrap
    ):
        if terminal_states is None:
            terminal_states = set()

        # make sure we can shuffle in-place cheaply
        data = list(map(tuple, data))

        for _ in range(n_passes):
            if shuffle_each_pass:
                random.shuffle(data)

            max_abs_delta = 0.0

            for (s, a, r, sp) in data:
                s = int(s); a = int(a); sp = int(sp)
                i = self.state_to_idx[s]
                j = self.state_to_idx.get(sp, None)

                if (sp in terminal_states) or (j is None):
                    max_next = 0.0
                else:
                    max_next = float(np.max(self.Q[j, :]))

                target = r + gamma * max_next
                delta  = target - self.Q[i, a]

                # step size: either fixed alpha or 1/N(s,a)
                if alpha is None:
                    self.N[i, a] += 1
                    step = 1.0 / self.N[i, a]
                else:
                    step = alpha

                self.Q[i, a] += step * delta
                if abs(delta) > max_abs_delta:
                    max_abs_delta = abs(delta)

            if max_abs_delta < tol:
                break

        return self.Q  # (nS, nA)

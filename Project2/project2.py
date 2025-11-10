#!/usr/bin/env python3

import numpy as np
import util
from QLearning import QLearning
from mdp import MDP
from policies import value_iteration

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

    S = np.unique(np.concatenate([data[:,0], data[:,3]])) # (s, sp)
    A = np.unique(data[:,1])

    model = QLearning(S, A)
    U, pi = model.simulate(data, 100)

    util.save("medium", model.state_to_idx, U, pi)

def main():
    # run_small()
    run_medium()


if __name__ == "__main__":
    main()

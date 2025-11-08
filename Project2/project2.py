#!/usr/bin/env python3

import numpy as np
import util
from mdp import MDP
from policies import value_iteration

def small_dataset():
    data = util.load_dataset("small.csv")

    mdp = MDP()
    mdp.mle(data)

    # Transistion models are conditional probabilities distributions, so they must sum up to one over all states
    print("Row sums of T by (s,a) should be 1 or 0 if unseen:")
    row_sums = mdp.T.sum(axis=2)
    print("min row sum:", row_sums[row_sums>0].min(), "max row sum:", row_sums.max())

    U, pi = value_iteration(mdp)

    util.save(mdp.state_to_idx,U,pi)

def main():
    small_dataset()

if __name__ == "__main__":
    main()

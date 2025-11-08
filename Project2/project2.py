#!/usr/bin/env python3

import numpy as np
import model
import util
from mdp import MDP
from policies import value_iteration

def main():
    mdp_data = util.load_dataset("small.csv").astype(float)  # or int for state/action columns
    mdp = model.construct_model(mdp_data)
    U, pi = value_iteration(10, mdp)
    
    print(U)
    print(pi)

if __name__ == "__main__":
    main()
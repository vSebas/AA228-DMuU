#!/usr/bin/env python3

import util

def main():
    mdp_data = util.load_dataset("small.csv")
    print(mdp_data)

if __name__ == "__main__":
    main()
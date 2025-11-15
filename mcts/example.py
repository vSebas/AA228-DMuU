# toy_mcts_example.py

import numpy as np
from mcts_recursive import MCTS

class LineWorldMDP:
    """
    Simple 1D grid:
        positions: 0, 1, 2, ..., goal_pos
        start at 0, goal at goal_pos

    Actions:
        -1: move left  (if state > 0)
        +1: move right (if state < goal_pos)

    Rewards:
        +1.0 when reaching the goal
        -0.1 per step otherwise
    """

    def __init__(self, goal_pos=4):
        self.goal_pos = goal_pos

    def actions(self, state):
        """Return list of valid actions from this state."""
        if state == self.goal_pos:
            # terminal state: no actions
            return []

        acts = []
        if state > 0:
            acts.append(-1)
        if state < self.goal_pos:
            acts.append(1)
        return acts

    def step(self, state, action):
        """
        Apply action to state.

        Returns:
            next_state, reward
        """
        next_state = state + action
        # clip to valid range just in case
        next_state = max(0, min(self.goal_pos, next_state))

        if next_state == self.goal_pos:
            reward = 1.0
        else:
            reward = -0.1

        return next_state, reward

    def rollout_policy(self, state):
        """Random policy over available actions."""
        acts = self.actions(state)
        if not acts:
            return None
        return np.random.choice(acts)


if __name__ == "__main__":
    np.random.seed(0)

    mdp = LineWorldMDP(goal_pos=4)

    # Tune these if you want:
    iters = 2000
    max_depth = 1
    c = 1.4
    gamma = 0.5

    mcts = MCTS(model=mdp, iters=iters, max_depth=max_depth, c=c, gamma=gamma)

    root_state = 0
    best_action = mcts.get_best_root_action(root_state)

    print(f"Best action from state {root_state}: {best_action}")

    a = mcts.get_best_root_action(root_state)

    print("Optimal action now:", a)

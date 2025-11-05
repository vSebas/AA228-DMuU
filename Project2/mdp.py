#!/usr/bin/env python3

class MDP:
    def __init__(self, S, A, T, R, gamma):
        self.S = S          # state space
        self.A = A          # action space
        self.T = T          # transition function
        self.R = R          # reward function
        self.gamma = gamma  # discount factor

    # def getStates(self):
    #     return self.S

    # def getPossibleActions(self, state):
    #     return self.actions.get(state, [])

    # def getTransitionProb(self, state, action, nextState):
    #     return self.transitionProbabilities.get(state, {}).get(action, {}).get(nextState, 0)

    # def getReward(self, state, action, nextState):
    #     return self.rewards.get(state, {}).get(action, {}).get(nextState, 0)

# class PolicyMDP(MDP):
#     def __init__(self, S, A, T, R, gamma, policy):
#         super().__init__(S, A, T, R, gamma)
#         self.pi = policy  # fixed policy for the MDP
#         self.k_max = 1000  # maximum iterations for policy evaluation
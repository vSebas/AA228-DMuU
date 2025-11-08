#!/usr/bin/env python3

from mdp import MDP
import numpy as np

def construct_model(data):
    # mdp_data rows: [s, a, r, sp]

    S_ids = np.unique(np.concatenate([data[:,0], data[:,3]]))  # include s and sp
    A_ids = np.unique(data[:,1])

    state_to_idx = {s_id:i for i, s_id in enumerate(S_ids)}
    action_to_idx = {a_id:i for i, a_id in enumerate(A_ids)}

    S = len(S_ids)
    A = len(A_ids)

    # Counts
    N_sas = np.zeros((S, A))                    # count of (s,a)
    N_sasp = np.zeros((S, A, S))                # count of (s,a,sp)
    R_sum = np.zeros((S, A))                    # sum of rewards per (s,a)

    # Accumulate
    for s_id, a_id, r, sp_id in data:
        s  = state_to_idx[s_id]
        a  = action_to_idx[a_id]
        sp = state_to_idx[sp_id]
        N_sas[s,a]    += 1
        N_sasp[s,a,sp] += 1
        R_sum[s,a]    += r


    # data = util.load_dataset("small.csv")
    # unique_states = np.unique(data[:,0])
    # unique_actions = np.unique(data[:,1])
    # unique_next_states = np.unique(data[:,3])
    # unique_transitions, transition_counts = np.unique(np.delete(data, 2, axis=1), return_counts=True, axis=0) # remove column r 
    # reward_totals = np.zeros((unique_states.size, unique_actions.size))
    # T = np.zeros((unique_states.size, unique_actions.size, unique_next_states.size)) 
    # R = np.zeros_like(reward_totals) 
    
    # for s, a, r, sp in data: reward_totals[int(s)-1,int(a)-1] = r 
    
    # for t, n in zip(unique_transitions, transition_counts): 
    #     s = int(t[0])-1
    #     a = int(t[1])-1 
    #     sp = int(t[2])-1, 
    #     T[s,a,sp] = n/max(unique_next_states.size,1) 
    #     R[s,a] = reward_totals[s,a] / max(unique_next_states.size, 1)

    # Construct Maximum-Likelihood MDP
    T = N_sasp / np.maximum(N_sas[:, :, None], 1)
    R = R_sum / np.maximum(N_sas[:, :,], 1)

    model = MDP(S_ids, A_ids, T, R, 0.3)

    return model
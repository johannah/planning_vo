import numpy as np
import matplotlib.pyplot as plt
from IPython import embed

rdn = np.random.RandomState(12)

data_file = '../data/human_road.npz'
npruns = np.load(data_file)

chunks_per_episode = 500
max_chunk_length = 201

all_states = np.array(npruns['states'])
all_actions = np.array(npruns['actions'])
all_tracks = npruns['roads']
num_subgoals_per_trace = 5
# name of states [img,speed,pos0,pos1,angle,steer,gas,brake]
# action array (steering, gas)

def get_sequences(inds, tracks, astates, aactions):
    state_seqs = np.zeros((chunks_per_episode*astates.shape[0],max_chunk_length,11))
    cnt = 0
    details = []
    all_subgoals = []
    for ep_num, (ind, track, states, actions) in enumerate(zip(inds, tracks, astates, aactions)):
        initial_obs = np.array(track[0][1:])
        # remove noop actions before getting started
        first = actions.sum(axis=1).nonzero()[0][0]
        actions = actions[first:,:]
        states = np.insert(states[first:,1:], 0, initial_obs, axis=0)
        last = actions.shape[0]-1
        starts = rdn.choice(np.arange(first, (last-max_chunk_length)-1), chunks_per_episode, replace=False)
        #end_plus = rdn.randint(min_chunk_length, max_chunk_length, chunks_per_episode)
        ends = starts+max_chunk_length
        #ends = np.clip(starts+end_plus, 0, last)
        for st,en in zip(starts, ends):
            details.append([ind, st, en, en-st])
            diff_y = np.insert(np.diff(states[st:en,1]), 0, 0.0)
            diff_x = np.insert(np.diff(states[st:en,2]), 0, 0.0)
            abs_subgoals = sorted(list(rdn.choice(np.arange(st+10, en-10, 10), num_subgoals_per_trace, replace=False)))
            suy = states[abs_subgoals,1]-states[st,1]
            sux = states[abs_subgoals,0]-states[st,0]
            subgoals = np.array([suy,sux]).T
            all_subgoals.append(subgoals)
            # name of states [pos0,pos1,speed,angle,steer,gas,brake,diffy,diffx,steering,throttle]
            state_seqs[cnt,:en-st,:] = np.hstack((states[st:en,1:], states[st:en,0][:,None], diff_y[:,None], diff_x[:,None], actions[st:en]))
            cnt+=1

    print("finished with cnt", cnt, state_seqs.shape, len(details))
    return state_seqs,np.array(details), np.array(all_subgoals)

all_inds = np.arange(all_states.shape[0])
num_train = int(all_inds.shape[0]*.85)
train_inds = rdn.choice(all_inds, num_train, replace=False)
test_inds = [a for a in all_inds if a not in train_inds]

test_sequences = get_sequences(test_inds, all_tracks[test_inds], all_states[test_inds], all_actions[test_inds])
train_sequences = get_sequences(train_inds, all_tracks[train_inds], all_states[train_inds], all_actions[train_inds])
np.savez("../data/train_2d_controller", states=train_sequences[0], details=train_sequences[1], subgoals=train_sequences[2])
np.savez("../data/test_2d_controller", states=test_sequences[0], details=test_sequences[1], subgoals=test_sequences[2])

embed()



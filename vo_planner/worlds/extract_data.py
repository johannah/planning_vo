import numpy as np
import matplotlib.pyplot as plt
from IPython import embed

rdn = np.random.RandomState(12)

data_file = 'human_road.npz'
npruns = np.load(data_file)

chunks_per_episode = 500
max_chunk_length = 200
min_chunk_length = 100

all_states = np.array(npruns['states'])
all_actions = np.array(npruns['actions'])
all_tracks = npruns['roads']

# name of states [img,speed,pos0,pos1,angle,steer,gas,brake] 
# action array (steering, gas)

def get_sequences(inds, astates, aactions):
    action_seqs = np.zeros((chunks_per_episode*astates.shape[0],max_chunk_length,2))
    state_seqs = np.zeros((chunks_per_episode*astates.shape[0],max_chunk_length,7))
    cnt = 0
    details = []
    for ep_num, (ind, states, actions) in enumerate(zip(inds,astates, aactions)):
        first = actions.sum(axis=1).nonzero()[0][0]
        last = actions.shape[0]-1
        starts = rdn.choice(np.arange(first, last-min_chunk_length), chunks_per_episode, replace=False)
        end_plus = rdn.randint(min_chunk_length, max_chunk_length, chunks_per_episode)
        ends = np.clip(starts+end_plus, 0, last)
        for st,en in zip(starts, ends):
            details.append([ind, st, en, en-st])
            action_seqs[cnt,:en-st] = actions[st:en]
            state_seqs[cnt,:en-st] = states[st:en,1:]
            cnt+=1
    return action_seqs,state_seqs,np.array(details)

all_inds = np.arange(all_states.shape[0])
num_train = int(all_inds.shape[0]*.85)
train_inds = rdn.choice(all_inds, num_train, replace=False)  
test_inds = [a for a in all_inds if a not in train_inds]

test_sequences = get_sequences(test_inds, all_states[test_inds], all_actions[test_inds])
train_sequences = get_sequences(train_inds, all_states[train_inds], all_actions[train_inds])
np.savez("train_2d_controller", actions=train_sequences[0], states=train_sequences[1], details=train_sequences[2])
np.savez("test_2d_controller", actions=test_sequences[0], states=test_sequences[1], details=test_sequences[2])

embed()



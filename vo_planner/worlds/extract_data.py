import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
data_file = 'human_road.npz'
npruns = np.load(data_file)
all_states = npruns['states']
all_actions = npruns['actions']
all_tracks = npruns['roads']

# name of states [img,speed,pos0,pos1,angle,steer,gas,brake] 
states = np.array(all_states[0])
# action array (steering, gas)
actions = np.array(all_actions[0])

embed()


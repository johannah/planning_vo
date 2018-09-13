import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from copy import deepcopy
import time
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
import shutil
from model_car_env import ModelCarEnv, plot_results, savedir
rdn = np.random.RandomState(33)
# TODO one-hot the action space?

def random_agent(env):
    last_state, lead_in_pred = env.reset()
    index, time, pos0, pos1, pred_y, h1_tm1, c1_tm1, h2_tm1, c2_tm1 = last_state
    finished = False
    pred_trace = np.zeros((env.max_step, pred_y.shape[0], 2))
    pred_trace[:lead_in_pred.shape[0]] = lead_in_pred
    reward = 0;
    while not finished:
        # need action to be of shape (1,2)
        action = env.actions[rdn.choice(env.action_inds)][None,:]
        last_state = [index,time,pos0,pos1,pred_y,h1_tm1,c1_tm1,h2_tm1,c2_tm1]
        next_state, r, finished = env.step(last_state, action=action)
        index, time, pos0, pos1, pred_y, h1_tm1, c1_tm1, h2_tm1, c2_tm1 = next_state
        # you have the option to correct position here before it goes into the
        # next state if you assume that you receive a gps waypoint - this might
        # be useful in training with collected data
        pred_trace[index] = [pos0, pos1]
        reward+=r

    basename = 'di_%010d_random' %env.data_index
    plot_path = os.path.join(savedir, basename+'.png')
    subgoals = env.ptst[env.sgt_ind][:,0]
    title='reward {}'.format(reward)
    plot_results(env.ptst, pred_trace, subgoals, plot_path, title=title, lead_in=env.lead_in)
    print("finished with reward=%s"%reward)

if __name__ == '__main__':
    tenv = ModelCarEnv()
    for i in range(5):
        random_agent(tenv)



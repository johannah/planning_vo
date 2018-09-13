import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from copy import deepcopy
import time
import os, sys
import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.init as init
from IPython import embed
import shutil
import torch # package for building functions with learnable parameters
import torch.nn as nn # prebuilt functions specific to neural networks
from torch.autograd import Variable # storing data while learning
from mdn_lstm import mdnLSTM
from utils import save_checkpoint, plot_strokes, get_dummy_data, load_data
from sample_2d_car import predict
rdn = np.random.RandomState(33)
torch.manual_seed(139)
savedir = 'predictions'

class ModelCarEnv():
    def __init__(self, data_path='../data/train_2d_controller.npz',
                       device='cpu',
                       number_mixtures=20, hidden_size=1024,
                       batch_size=1, lead_in=15,
                       model_load_path='saved_models/model_000000006500000.pkl',
                       random_state_number=44):

        # time index (axis=0)
        self.rdn = np.random.RandomState(random_state_number)
        # data path can be testing or training data
        self.data_path = data_path
        self.finished = True
        # return data as numpy array
        self.x, self.y, self.subgoals, self.keys, self.pts = load_data(self.data_path, cut=True)
        # ughhh TODO one hot the action data in the simulation model
        self.steering_actions = [-1, -0.5, 0, 0.5, 1]
        self.throttle_actions = [-.5, .2, .4, .6, .8]
        actions = []
        for s in self.steering_actions:
            for t in self.throttle_actions:
                actions.append((s,t))

        self.actions = np.array(actions)
        self.action_inds = np.arange(len(self.actions))
        self.max_step = self.y.shape[0]
        self.data_indexes = np.arange(self.x.shape[1], dtype=np.int)
        self.trace_length = self.x.shape[0]
        self.input_size = self.x.shape[2]
        self.output_size = self.y.shape[2]
        self.lead_in = lead_in
        # build model
        self.model_load_path = model_load_path
        if not os.path.exists(self.model_load_path):
            print("Provided model path does not exist!")
            print(self.model_load_path)
            sys.exit()
        else:
            lstm_dict = torch.load(self.model_load_path)
        try:
            self.batch_size = batch_size
            self.number_mixtures = number_mixtures
            self.hidden_size = hidden_size
            self.device = device
            self.lstm = mdnLSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                                number_mixtures=self.number_mixtures).to(self.device)
            self.lstm.load_state_dict(lstm_dict['state_dict'])
            print("sucessfully loaded model: %s" %self.model_load_path)
        except Exception:
            print('unable to load model')
            print(e)

    def reset(self):
        # load true trace and subgoals
        h1_tm1 = Variable(torch.zeros((self.batch_size, self.hidden_size))).to(self.device)
        c1_tm1 = Variable(torch.zeros((self.batch_size, self.hidden_size))).to(self.device)
        h2_tm1 = Variable(torch.zeros((self.batch_size, self.hidden_size))).to(self.device)
        c2_tm1 = Variable(torch.zeros((self.batch_size, self.hidden_size))).to(self.device)
        self.data_index = self.rdn.choice(self.data_indexes, 1)[0]
        # data is [time in trace, trace number, features]
        self.xt = self.x[:,self.data_index:self.data_index+1]
        self.yt = self.y[:,self.data_index:self.data_index+1]
        self.ptst = self.pts[:,self.data_index:self.data_index+1]
        self.xt_pt = Variable(torch.FloatTensor(self.xt))
        self.yt_pt = Variable(torch.FloatTensor(self.yt))
        self.sgt_ind = list(self.subgoals[:,self.data_index, 0])
        # time required to complete circuit
        self.time = 0
        # position always starts at 0,0
        self.index = 0
        self.finished = False
        # get diff from the real state x
        pred = np.zeros((self.lead_in+1, self.batch_size, 2))
        for i in range(self.lead_in):
            # use gt for action because this is the lead in
            true_pred = np.array([self.yt[self.index, 0, self.keys['y_diff0']],
                                  self.yt[self.index, 0, self.keys['y_diff1']]])[None,:]

            true_action = np.array([self.xt[self.index,0,self.keys['x_steering']],
                                    self.xt[self.index,0,self.keys['x_throttle']]])[None,:]
            pred[self.index,:,0] = true_pos0 = self.ptst[self.index, 0, 0]
            pred[self.index,:,1] = true_pos1 = self.ptst[self.index, 0, 1]
            last_state = [self.index,self.time,true_pos0,true_pos1,true_pred,h1_tm1,c1_tm1,h2_tm1,c2_tm1]
            next_state, r, self.finished = self.step(last_state, action=true_action)
            self.index, uncounted_time, unused_pos0, unused_pos1, unused_pred, h1_tm1, c1_tm1, h2_tm1, c2_tm1 = next_state
        pred[self.index,:,0] = true_pos0 = self.ptst[self.index, 0, 0]
        pred[self.index,:,1] = true_pos1 = self.ptst[self.index, 0, 1]
        next_state = [self.index,self.time,true_pos0,true_pos1,self.yt[self.index],h1_tm1,c1_tm1,h2_tm1,c2_tm1]
        return next_state, pred

    def step(self, last_state, action, use_center=False):
        """
        last_state is list with [index, time, pos0, pos1, last_x, h1_tm1, c1_tm1, h2_tm1, c2_tm1]
        action is list with [steering, throttle]
        use_center is bool; true means use the mean of the mdn, otherwise sample
        returns list text_state, float reward, bool finished
        """
        if self.finished:
            return  last_state, -999, self.finished
        st = time.time()
        index, t, pos0, pos1, last_y, h1_tm1, c1_tm1, h2_tm1, c2_tm1 = last_state
        # concat last prediction and action into state
        x = Variable(torch.FloatTensor(np.hstack((last_y, action)))).to(self.device)
        predict_output = predict(self.lstm, x, h1_tm1, c1_tm1, h2_tm1, c2_tm1, trdn=self.rdn, use_center=use_center)
        pred, h1_tm1, c1_tm1, h2_tm1, c2_tm1 = predict_output
        # TODO properly calculate time required
        # prevent zeros - don't count brake for now
        steering,throttle = action[0]
        t += 1.0/(max([throttle,0.0])+1e-5)
        #print('before', index, [pos0,pos1] , self.ptst[index])
        index+=1
        pos0 += pred[0,0]
        pos1 += pred[0,1]
        reward, finished = self.calculate_reward(index, [pos0,pos1], t)
        next_state = [index, t, pos0, pos1, pred, h1_tm1, c1_tm1, h2_tm1, c2_tm1]
        et = time.time()
        return next_state, reward, finished

    def calculate_distance_penalty(self, true_tuple, pred_tuple):
        # measure absolute position error
        # dont do square root to save time?
        offset0 = pred_tuple[0]-true_tuple[0]
        offset1 = pred_tuple[1]-true_tuple[1]
        dist = np.log(offset0**2 + offset1**2)
        #print('tp', true_tuple, pred_tuple)
        return -dist

    def calculate_reward(self, index, pred_y, dt):
        finished = False
        reward = 0
        if index in self.sgt_ind:
            true_tuple = self.ptst[index,0]
            reward = self.calculate_distance_penalty(true_tuple, pred_y)
            # offset for being quick?
            #reward += dt/float(index)
        # if we are at last index
        if index >= self.max_step-1:
            finished = True
            self.finished = True
        #print(index,reward,finished)
        return reward, finished

def plot_results(true_trace, pred_trace, subgoals, path='ex.png', title='', lead_in=0):
    plt.figure()
    pc = 'mediumslateblue'
    lc = 'purple'
    sc = 'g'
    tc = 'orangered'
    gc = 'salmon'
    plt.plot(true_trace[:,0,0], true_trace[:,0,1], c=tc, linewidth=.7, alpha=.5, label='true')
    for xx in range(pred_trace.shape[1]):
        p = pred_trace[:,xx]
        if not xx:
            plt.plot(p[:,0], p[:,1], c=pc, linewidth=1.2, alpha=.8, label='pred')
            plt.plot(p[:lead_in,0], p[:lead_in,1], c=lc, linewidth=.8, alpha=.8, label='lead in')
        else:
            plt.plot(p[:,0], p[:,1], c=pc, linewidth=1.2, alpha=.8)
            plt.plot(p[:lead_in,0], p[:lead_in,1], c=lc, linewidth=.8, alpha=.8)
        plt.scatter(p[:,0], p[:,1], c=pc, s=.8, alpha=.6)

    plt.scatter(subgoals[:,0], subgoals[:,1], c=gc, marker='x', s=10, alpha=1, label='goal')
    plt.scatter([true_trace[0,0,0]], [true_trace[0,0,1]], c=sc, marker='o', s=20, alpha=.9, label='start')
    plt.title(title)
    plt.legend()
    plt.savefig(path)
    plt.close()

def complete_perfect_run():
    # function for sanity checking env
    env = ModelCarEnv()
    last_state = env.reset()
    index, time, pos0, pos1, pred_y, h1_tm1, c1_tm1, h2_tm1, c2_tm1 = last_state
    finished = False
    reward = 0
    pred_trace = np.zeros((env.max_step, pred_y.shape[0], 2))
    while not finished:
        # use gt for action and prediction - overwriting a bunch
        true_pred = np.array([env.yt[index, 0, env.keys['y_diff0']],
                              env.yt[index, 0, env.keys['y_diff1']]])[None,:]

        true_action = np.array([env.xt[index,0,env.keys['x_steering']],
                                env.xt[index,0,env.keys['x_throttle']]])[None,:]
        true_pos0 = env.ptst[index, 0, 0]
        true_pos1 = env.ptst[index, 0, 1]
        last_state = [index,time,true_pos0,true_pos1,true_pred,h1_tm1,c1_tm1,h2_tm1,c2_tm1]
        next_state, r, finished = env.step(last_state, action=true_action)
        index, time, unused_pos0, unused_pos1, unused_pred, h1_tm1, c1_tm1, h2_tm1, c2_tm1 = next_state
        pred_trace[index] = [unused_pos0, unused_pos1]
        reward+=r

    basename = os.path.join(savedir, 'di%06d_perfect' %env.data_index)
    plot_path = os.path.join(savedir, basename+'.png')
    subgoals = env.ptst[env.sgt_ind][:,0]
    title='reward {}'.format(reward)
    plot_results(env.ptst, np.array(pred_trace), subgoals, plot_path, title=title)
    print("finished perfect run with reward - should be near zero", reward)
    embed()

if __name__ == '__main__':
    complete_perfect_run()
    embed()



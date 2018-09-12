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
# TODO one-hot the action space?

torch.manual_seed(139)
# David's blog post:
# https://github.com/hardmaru/pytorch_notebooks/blob/master/mixture_density_networks.ipynb
# input data should be - timestep, batchsize, features!
# Sampling example from tfbldr kkastner
# https://github.com/kastnerkyle/tfbldr/blob/master/examples/handwriting/generate_handwriting.py


class ModelCarEnv():
    def __init__(self, data_path='../data/train_2d_controller.npz', device='cpu',
                       number_mixtures=20, hidden_size=1024, batch_size=100, lead_in=4,
                       model_load_path='mdn_2d_models/model_000000005360000.pkl', random_state_number=44):

        # index in x feature space containing last action steering
        self.x_steering_index = 3
        # index in x feature space containing last action throttle
        self.x_throttle_index = 4
        self.y_diff0_index = 0
        self.y_diff1_index = 1
        self.rdn = np.random.RandomState(random_state_number)
        # data path can be testing or training data
        self.data_path = data_path
        # return data as numpy array
        self.x, self.y, self.subgoals = load_data(self.data_path, cut=True)
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
        h1_tm1 = Variable(torch.zeros((1, self.hidden_size))).to(DEVICE)
        c1_tm1 = Variable(torch.zeros((1, self.hidden_size))).to(DEVICE)
        h2_tm1 = Variable(torch.zeros((1, self.hidden_size))).to(DEVICE)
        c2_tm1 = Variable(torch.zeros((1, self.hidden_size))).to(DEVICE)
        self.data_index = self.rdn.choice(self.data_indexes, 1)[0]
        # data is [time in trace, trace number, features]
        self.xt = self.x[:,self.data_index:self.data_index+1]
        self.yt = self.y[:,self.data_index:self.data_index+1]
        self.xt_pt = Variable(torch.FloatTensor(self.xt))
        self.yt_pt = Variable(torch.FloatTensor(self.yt))
        self.sgt_ind = list(self.subgoals[self.data_index, :, 0])
        self.sgt_diff = self.subgoals[self.data_index, :, 1:]
        self.index = 0
        # time required to complete circuit
        self.time = 0
        pos0 = 0; pos1 = 0
        #for i in range(self.lead_in):
        for i in range(199):
            # use gt for action because this is the lead in

            #true_action = [self.xt[self.index+1,0,self.x_steering_index],
            #               self.xt[self.index+1,0,self.x_throttle_index]]
            #last_state = [self.index,self.time,pos0,pos1,self.xt_pt[self.index],h1_tm1,c1_tm1,h2_tm1,c2_tm1]
            #next_state, reward, finished = self.step(last_state, action=true_action)
            # update position with diffs (this time from true value)
            pos0 += self.yt[self.index, 0, self.y_diff0_index]
            pos1 += self.yt[self.index, 0, self.y_diff1_index]
            self.index+=1
            print(pos0, pos1)
            #self.index, uncounted_time, unused_pos0, unused_pos1, unused_pred_next_state, h1_tm1, c1_tm1, h2_tm1, c2_tm1 = next_state

        next_state = [self.index,self.time,pos0,pos1,self.xt_pt[self.index],h1_tm1,c1_tm1,h2_tm1,c2_tm1]
        return next_state

    def step(self, last_state, action, use_center=False):
        """
        last_state is list with [index, time, pos0, pos1, last_x, h1_tm1, c1_tm1, h2_tm1, c2_tm1]
        action is list with [steering, throttle]
        use_center is bool; true means use the mean of the mdn, otherwise sample
        returns list text_state, float reward, bool finished
        """
        index, t, pos0, pos1, last_x, h1_tm1, c1_tm1, h2_tm1, c2_tm1 = last_state
        predict_output = predict(self.lstm, last_x, h1_tm1, c1_tm1, h2_tm1, c2_tm1, trdn=self.rdn, use_center=use_center)
        pred, h1_tm1, c1_tm1, h2_tm1, c2_tm1 = predict_output

        # TODO properly calculate time required
        # prevent zeros - don't count brake for now
        steering,throttle = action
        t += 1.0/(max([throttle,0.0])+1e-5)
        index+=1
        #print('predict', pred[0,0],pos0)
        pos0 += pred[0,0]
        pos1 += pred[0,1]
        #print('after', pred[0,0],pos0)
        reward, finished = self.calculate_reward(index, [pos0,pos1], t)
        # hack because i didn't train full state information
        # update predicted ydiff, xdiff
        next_x = self.xt[index]
        next_x[:,-2:] = torch.FloatTensor(pred).to(DEVICE)
        next_state = [index, t, pos0, pos1, next_x, h1_tm1, c1_tm1, h2_tm1, c2_tm1]
        return next_state, reward, finished

    def calculate_distance_penalty(self, pred_tuple, true_tuple):
        # measure absolute position error
        print(pred_tuple, true_tuple)
        dist = np.sqrt((pred_tuple[0]-true_tuple[0])**2 + (pred_tuple[1]-true_tuple[1])**2)
        return -dist

    def calculate_reward(self, index, pred_y, dt):
        finished = False
        reward = 0
        if index in self.sgt_ind:
            tind = self.sgt_ind.index(index)
            reward = self.calculate_distance_penalty(pred_y, self.sgt_diff[tind])
            #reward += dt/float(index)
            # if we are at last index
        if index >= self.max_step:
            finished = True
        return reward, finished

def complete_perfect_run():
    env = ModelCarEnv()
    last_state = env.reset()
    index, time, pos0, pos1, pred_next_state, h1_tm1, c1_tm1, h2_tm1, c2_tm1 = last_state
    finished = False
    reward = 0
    while not finished:
        # use gt for action because this is the lead in
        # the last action cant be gotten from x because
        true_action = [env.xt[index+1,0,env.x_steering_index],
                       env.xt[index+1,0,env.x_throttle_index]]
        true_next_state = env.xt_pt[index]
        last_state = [index,time,pos0,pos1,true_next_state,h1_tm1,c1_tm1,h2_tm1,c2_tm1]
        next_state, r, finished = env.step(last_state, action=true_action)
        index, time, pos0, pos1, pred_next_state, h1_tm1, c1_tm1, h2_tm1, c2_tm1 = next_state
        pos0 += pos0; pos1 += pos1
        reward+=r
    print("finished with  reward", r)
    embed()




if __name__ == '__main__':
    import argparse
    data_batch_size = 32
    seq_length = 200
    hidden_size = 1024
    number_mixtures = 20
    train_losses, test_losses, train_cnts, test_cnts = [], [], [], []

    img_savedir = 'predictions'
    cnt = 0
    default_model_loadname = 'mdn_2d_models/model_000000000080000.pkl'
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_loadname', default=default_model_loadname)
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('-uc', '--use_center', action='store_true', default=False, help='use means instead of sampling')
    parser.add_argument('-tf', '--teacher_force', action='store_true', default=False)
    parser.add_argument('--training', action='store_true', default=False, help='generate from training set rather than test set')
    parser.add_argument('-n', '--num',default=300, help='length of data to generate')
    parser.add_argument('-bs', '--batch_size',type=int,default=1, help='number of each sample to generate')
    parser.add_argument('-bn', '--batch_num',type=int, default=0, help='index into batch from teacher force to use')
    parser.add_argument('--whole_batch', action='store_true', default=False, help='plot an entire batch')
    parser.add_argument('--num_plot', default=10, type=int, help='number of examples from training and test to plot')
    parser.add_argument('-l', '--lead_in', default=5, type=int, help='number of examples to teacher force before running')

    args = parser.parse_args()

    if args.cuda:
        DEVICE = 'cuda'
    else:
        DEVICE = 'cpu'

    complete_perfect_run()
    embed()



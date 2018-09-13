import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from copy import deepcopy
import time
import os, sys
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.init as init
from IPython import embed
import shutil
import torch # package for building functions with learnable parameters
from torch.autograd import Variable # storing data while learning
rdn = np.random.RandomState(33)
# TODO one-hot the action space?

torch.manual_seed(139)

def plot_losses(train_cnts, train_losses, test_cnts, test_losses, name='loss_example.png'):
    plt.figure(figsize=(3,3))
    plt.plot(train_cnts, train_losses, label='train loss', lw=3)
    plt.plot(test_cnts, test_losses, label='test loss', lw=1)
    plt.legend()
    plt.savefig(name)
    plt.close()

def split_strokes(points):
    points = np.array(points)
    strokes = []
    b = 0
    for e in range(len(points)):
        if points[e, 2] == 1.:
            strokes += [points[b: e + 1, :2].copy()]
            b = e + 1
    strokes += [points[b: e + 1, :2].copy()]
    return strokes

def get_dummy_data(v_x, v_y):
    for i in range(v_x.shape[1]):
        v_x[:,i] = v_x[:,0]
        v_y[:,i] = v_y[:,0]
    return v_x, v_y

def plot_strokes(strokes_x_in, strokes_y_in, lead_in=0, name='example.png',pen=True):
    strokes_x = deepcopy(strokes_x_in)
    f, ax1 = plt.subplots(1,1, figsize=(6,3))

    if pen: # pen up pen down is third channel
        strokes_x[:, :2] = np.cumsum(strokes_x[:, :2], axis=0)
        ax1.scatter(strokes_x[:,0], -strokes_x[:,1], c='b', s=2, label='pred')
        for stroke in split_strokes(strokes_x):
            ax1.plot(stroke[:,0], -stroke[:,1], c='b', linewidth=1)

        if np.abs(strokes_y_in).sum()>0:
            strokes_y = deepcopy(strokes_y_in)
            strokes_y[:, :2] = np.cumsum(strokes_y[:, :2], axis=0)
            ax1.scatter(strokes_y[:,0], -strokes_y[:,1], c='g', s=2, label='gt')
            for stroke in split_strokes(strokes_y):
                ax1.plot(stroke[:,0], -stroke[:,1], c='g', linewidth=1)
    else:
        # no pen indicator
        for i in range(strokes_x.shape[1]):
            strokes_xi = np.cumsum(deepcopy(strokes_x[:,i]), axis=0)
            if not i:
                ax1.plot(strokes_xi[:,0], -strokes_xi[:,1], c='b', label='%s pred'%strokes_x.shape[1], linewidth=.5, alpha=0.5)
            else:
                ax1.plot(strokes_xi[:,0], -strokes_xi[:,1], c='b', linewidth=.5, alpha=.5)
            ax1.scatter(strokes_xi[:,0], -strokes_xi[:,1], c='b', s=.2, alpha=.5)
        if np.abs(strokes_y_in).sum()>0:
            strokes_y = deepcopy(strokes_y_in)
            strokes_y = np.cumsum(strokes_y, axis=0)
            ax1.scatter(strokes_y[:,0,0], -strokes_y[:,0,1], c='g', s=.9)
            ax1.plot(strokes_y[:,0,0], -strokes_y[:,0,1], c='g', label='gt', linewidth=2, alpha=.9)
        if lead_in:
            ax1.scatter([strokes_y[lead_in,0,0]], [-strokes_y[lead_in,0,1]], c='r', marker='o', s=5, label='lead in')

    plt.legend()
    print('plotting %s'%name)
    plt.savefig(name)
    plt.close()

def save_checkpoint(state, filename='model.pkl'):
    print("starting save of {}".format(filename))
    torch.save(state, filename)
    print("finishing save of {}".format(filename))

def load_data(load_path, cut=False):
    data = np.load(load_path)
    # name of features [pos0,pos1,speed,angle,steer,gas,brake,diffy,diffx,steering,throttle]
    # shape is [trace number, sequence, features]
    estates = data['states']
    # details is [episode, start, end, length of sequence from episode]
    edetails = data['details']
    subgoals = np.swapaxes(data['subgoals'], 0,1)
    # shuffle the indexes
    indexes = np.arange(estates.shape[0])
    rdn.shuffle(indexes)
    # change to [timestep, batch, features]
    states = estates[indexes].swapaxes(1,0)
    details = edetails[indexes]
    # AS OF Sept 12 this is old - have different format
    # change data type and shape, move from numpy to torch
    # note that we need to convert all data to np.float32 for pytorch
    #   0    1     2    3      4    5   6     7    8      9         10
    # [pos0,pos1,speed,angle,steer,gas,brake,diffy,diffx,steering,throttle]
    #x_tensor = torch.from_numpy(np.float32(states[:-1,:,[3,4,9,10,2,7,8]]))
    #y_tensor = torch.from_numpy(np.float32(states[1:,:,[2,7,8]]))
    #x_variable = Variable(x_tensor, requires_grad=True)
    #y_variable = Variable(y_tensor, requires_grad=False)
    # dont cut off for the rl case in which we need the last state
#    if cut:
#        x = np.array(states[:,:,[2,3,4,9,10,7,8]], dtype=np.float32)
#    else:
#        x = np.array(states[:-1,:,[2,3,4,9,10,7,8]], dtype=np.float32)
#    # one time step ahead to predict diffs
#    y = np.array(states[1:,:,[7,8]], dtype=np.float32)

    #                     0   1     2     3     4      5        6
    # # name of states [pos0,pos1,yaw,diffy,diffx,steering,throttle]
    # assume actions (5,6) are given
    #x                   0       1      2        3
    # # name of states [diff0,diff1,steering,throttle]
    # indexes
    keys = {
            'x_diff0':0,
            'x_diff1':1,
            'x_steering':2,
            'x_throttle':3,
            'y_diff0':0,
            'y_diff1':1}
    pts = states[:,:,[0,1]]
    if cut:
        x = np.array(states[:,:,[3,4,5,6]], dtype=np.float32)
    else:
        x = np.array(states[:-1,:,[3,4,5,6]], dtype=np.float32)
    # one time step ahead to predict diffs
    y = np.array(states[1:,:,[3,4]], dtype=np.float32)
    return x, y, subgoals, keys, pts


class DataLoader():
    def __init__(self, train_load_path, test_load_path, batch_size=32, random_number=394):
        self.rdn = np.random.RandomState(random_number)
        self.batch_size = batch_size
        self.x,self.y, _, _, _ = load_data(train_load_path)
        self.num_batches = self.x.shape[1]//self.batch_size
        self.batch_array = np.arange(self.x.shape[1])
        self.valid_x,self.valid_y,_,_,_ = load_data(test_load_path)

    def validation_data(self):
        max_idx = min(self.batch_size, self.valid_x.shape[1])
        return self.valid_x[:,:max_idx], self.valid_y[:,:max_idx]

    def next_batch(self):
        batch_choice = self.rdn.choice(self.batch_array, self.batch_size,replace=False)
        return self.x[:,batch_choice], self.y[:,batch_choice]

def plot_traces(trues_e, tf_predicts_e, predicts_e, filename):
    ugty = np.cumsum(trues_e[:,0])
    ugtx = np.cumsum(trues_e[:,1])
    tfy = np.cumsum(tf_predicts_e[:,0])
    tfx = np.cumsum(tf_predicts_e[:,1])
    py = np.cumsum(predicts_e[:,0])
    px = np.cumsum(predicts_e[:,1])

    xmin = np.min([px.min(), tfx.min(), ugtx.min()])-10
    xmax = np.max([px.max(), tfx.max(), ugtx.max()])+10
    ymin = np.min([py.min(), tfy.min(), ugty.min()])-10
    ymax = np.max([py.max(), tfy.max(), ugty.max()])+10
    f,ax=plt.subplots(1,3, figsize=(9,3))
    ## original coordinates
    ax[0].scatter(ugtx,ugty, c=np.arange(ugty.shape[0]))
    ax[0].set_xlim([xmin,xmax])
    ax[0].set_ylim([ymin,ymax])
    ax[0].set_title("target")
    ax[1].scatter(tfx,tfy, c=np.arange(tfy.shape[0]))
    ax[1].set_xlim([xmin,xmax])
    ax[1].set_ylim([ymin,ymax])
    ax[1].set_title('teacher force predict')
    ax[2].scatter(px, py, c=np.arange(py.shape[0]))
    ax[2].set_xlim([xmin,xmax])
    ax[2].set_ylim([ymin,ymax])
    ax[2].set_title('predict')
    plt.savefig(filename)
    plt.close()





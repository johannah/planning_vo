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
from vo_lstm import LSTM
rdn = np.random.RandomState(33)
# TODO one-hot the action space?

torch.manual_seed(139)
#
def save_checkpoint(state, filename='model.pkl'):
    print("starting save of {}".format(filename))
    torch.save(state, filename)
    print("finishing save of {}".format(filename))

def load_data(load_path):
    train_data = np.load(load_path)
    # name of features [pos0,pos1,speed,angle,steer,gas,brake,diffy,diffx,steering,throttle]
    # shape is [trace number, sequence, features]
    estates = train_data['states']
    # details is [episode, start, end, length of sequence from episode]
    edetails = train_data['details']
    # shuffle the indexes
    indexes = np.arange(estates.shape[0])
    rdn.shuffle(indexes)
    # change to [timestep, batch, features]
    states = estates[indexes].swapaxes(1,0)
    details = edetails[indexes]
    # change data type and shape, move from numpy to torch
    # note that we need to convert all data to np.float32 for pytorch
    #   0    1     2    3      4    5   6     7    8      9         10
    # [pos0,pos1,speed,angle,steer,gas,brake,diffy,diffx,steering,throttle]
    x_tensor = torch.from_numpy(np.float32(states[:-1,:,[3,4,9,10,2,7,8]]))
    y_tensor = torch.from_numpy(np.float32(states[1:,:,[2,7,8]]))
    x_variable = Variable(x_tensor, requires_grad=True)
    y_variable = Variable(y_tensor, requires_grad=False)
    return x_tensor, y_tensor

def teacher_force_predict(lstm, hidden_size, DEVICE, mse_loss, x, y):
    # not done
    bs = x.shape[1]
    h1_tm1 = Variable(torch.zeros((bs, hidden_size))).to(DEVICE)
    c1_tm1 = Variable(torch.zeros((bs, hidden_size))).to(DEVICE)
    h2_tm1 = Variable(torch.zeros((bs, hidden_size))).to(DEVICE)
    c2_tm1 = Variable(torch.zeros((bs, hidden_size))).to(DEVICE)
    outputs = []
    # one batch of x
    for i in np.arange(0,x.shape[0]):
        output, h1_tm1, c1_tm1, h2_tm1, c2_tm1 = lstm(x[i].to(DEVICE), h1_tm1, c1_tm1, h2_tm1, c2_tm1)
        outputs+=[output]
    y_pred = torch.stack(outputs, 0)
    mse_loss = ((y_pred-y.to(DEVICE))**2)
    losses = list(mse_loss.data.numpy())
    return y_pred, losses

def predict(lstm, hidden_size, DEVICE, mse_loss, x, y):
    # not done
    bs = x.shape[1]
    h1_tm1 = Variable(torch.zeros((bs, hidden_size))).to(DEVICE)
    c1_tm1 = Variable(torch.zeros((bs, hidden_size))).to(DEVICE)
    h2_tm1 = Variable(torch.zeros((bs, hidden_size))).to(DEVICE)
    c2_tm1 = Variable(torch.zeros((bs, hidden_size))).to(DEVICE)
    outputs = []
    input_data = x[0]
    # one batch of x
    for i in np.arange(x.shape[0]):
        output, h1_tm1, c1_tm1, h2_tm1, c2_tm1 = lstm(input_data.to(DEVICE), h1_tm1, c1_tm1, h2_tm1, c2_tm1)
        outputs+=[output]
        if i < x.shape[0]-1:
            input_data = x[i+1]
            # replace the offset with the predicted offset
            #if i < 10:
            #    print(input_data[:,[4,5,6]])
            #    print(output)
            #    print('-------------',i+1)
            input_data[:,[4,5,6]] = output
    y_pred = torch.stack(outputs, 0)
    #print(y_pred.shape)
    mse_loss = ((y_pred-y)**2)
    losses = list(mse_loss.data.numpy())
    return y_pred, losses

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





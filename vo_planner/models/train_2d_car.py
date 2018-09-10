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
from utils import save_checkpoint, plot_losses, plot_strokes, get_dummy_data, DataLoader
rdn = np.random.RandomState(33)
# TODO one-hot the action space?

torch.manual_seed(139)
# David's blog post:
# https://github.com/hardmaru/pytorch_notebooks/blob/master/mixture_density_networks.ipynb
# target is nput data shifted by one time step
# input data should be - timestep, batchsize, features!

def train(x, y, validation=False):
    optim.zero_grad()
    bs = x.shape[1]
    h1_tm1 = Variable(torch.zeros((bs, hidden_size))).to(DEVICE)
    c1_tm1 = Variable(torch.zeros((bs, hidden_size))).to(DEVICE)
    h2_tm1 = Variable(torch.zeros((bs, hidden_size))).to(DEVICE)
    c2_tm1 = Variable(torch.zeros((bs, hidden_size))).to(DEVICE)
    outputs = []
    x = x.to(DEVICE)
    y = y.to(DEVICE)
    # one batch of x
    for i in np.arange(0,x.shape[0]):
        xin = x[i]
        output, h1_tm1, c1_tm1, h2_tm1, c2_tm1 = lstm(xin, h1_tm1, c1_tm1, h2_tm1, c2_tm1)
        outputs+=[output]
    y_pred = torch.stack(outputs, 0)
    y_pred_flat = y_pred.reshape(y_pred.shape[0]*y_pred.shape[1],y_pred.shape[2])
    y1_flat = y[:,:,0]
    y2_flat = y[:,:,1]
    #y3_flat = y[:,:,2]
    y1_flat = y1_flat.reshape(y1_flat.shape[0]*y1_flat.shape[1])[:,None]
    y2_flat = y2_flat.reshape(y2_flat.shape[0]*y2_flat.shape[1])[:,None]
    #y3_flat = y3_flat.reshape(y3_flat.shape[0]*y3_flat.shape[1])[:,None]
    #out_pi, out_mu1, out_mu2, out_sigma1, out_sigma2, out_corr, out_eos = lstm.get_mixture_coef(y_pred_flat)
    out_pi, out_mu1, out_mu2, out_sigma1, out_sigma2, out_corr = lstm.get_mixture_coef(y_pred_flat)
    #loss = lstm.get_lossfunc(out_pi, out_mu1, out_mu2, out_sigma1, out_sigma2, out_corr, out_eos, y1_flat, y2_flat, y3_flat)
    loss = lstm.get_lossfunc(out_pi, out_mu1, out_mu2, out_sigma1, out_sigma2, out_corr, y1_flat, y2_flat)
    if not validation:
        loss.backward()
        for p in lstm.parameters():
            p.grad.data.clamp_(min=-grad_clip,max=grad_clip)
        optim.step()
    rloss = loss.cpu().data.numpy()
    return y_pred, rloss

def loop(data_loader, num_epochs=1000, save_every=1000, train_losses=[], test_losses=[], train_cnts=[], test_cnts=[], dummy=False):
    print("starting training loop for data with %s batches"%data_loader.num_batches)
    st = time.time()
    if len(train_losses):
        # resume cnt from last save
        last_save = train_cnts[-1]
        cnt = train_cnts[-1]
    else:
        last_save = 0
        cnt = 0

    for e in range(num_epochs):
        ecnt = 0
        tst = round((time.time()-st)/60., 0)
        if not e%1 and e>0:
            print("starting epoch %s, %s mins, loss %s, seen %s, last save at %s" %(e, tst, train_losses[-1], cnt, last_save))
        batch_loss = []
        for b in range(data_loader.num_batches):
            xnp, ynp = data_loader.next_batch()
            x = Variable(torch.FloatTensor(xnp))
            y = Variable(torch.FloatTensor(ynp))
            y_pred, loss = train(x,y,validation=False)
            train_cnts.append(cnt)
            train_losses.append(loss)
            if cnt%100:
                valy_pred, val_mean_loss = train(v_x,v_y,validation=True)
                test_losses.append(val_mean_loss)
                test_cnts.append(cnt)
            if cnt-last_save >= save_every:
                last_save = cnt
                # find test loss
                print('epoch: {} saving after example {} train loss {} test loss {}'.format(e,cnt,loss,val_mean_loss))
                state = {
                        'train_cnts':train_cnts,
                        'train_losses':train_losses,
                        'test_cnts':  test_cnts,
                        'test_losses':test_losses,
                        'state_dict':lstm.state_dict(),
                        'optimizer':optim.state_dict(),
                         }
                basename = os.path.join(savedir, '%s_%015d'%(model_save_name,cnt))
                plot_losses(train_cnts, train_losses, test_cnts, test_losses, name=basename+'_loss.png')
                save_checkpoint(state, filename=basename+'.pkl')

            cnt+= x.shape[1]
            ecnt+= x.shape[1]


if __name__ == '__main__':
    import argparse

    batch_size = 32
    seq_length = 200
    hidden_size = 1024
    savedir = 'mdn_2d_models'
    number_mixtures = 20
    grad_clip = 5
    train_losses, test_losses, train_cnts, test_cnts = [], [], [], []

    cnt = 0
    default_model_loadname = 'mdn_2d_models/model_000000000190304.pkl'
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('--dummy', action='store_true', default=False)
    parser.add_argument('-po', '--plot', action='store_true', default=False)
    parser.add_argument('-m', '--model_loadname', default=default_model_loadname)
    parser.add_argument('-ne', '--num_epochs',default=300, help='num epochs to train')
    parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float, help='learning_rate')
    parser.add_argument('-se', '--save_every',default=20000, help='how often in epochs to save training model')
    parser.add_argument('--limit', default=-1, type=int, help='limit training data to reduce convergence time')
    parser.add_argument('-l', '--load', action='store_true', default=False, help='load model to continue training or to generate. model path is specified with -m')
    parser.add_argument('-v', '--validate', action='store_true', default=False, help='test results')

    args = parser.parse_args()

    if args.cuda:
        DEVICE = 'cuda'
    else:
        DEVICE = 'cpu'
        print("using DEVICE: %s" %DEVICE)

    save_every = args.save_every
    data_loader = DataLoader(train_load_path='../data/train_2d_controller.npz',
                             test_load_path='../data/train_2d_controller.npz',
                             batch_size=batch_size)

    v_xnp, v_ynp = data_loader.validation_data()

    v_x = Variable(torch.FloatTensor(v_xnp))
    v_y = Variable(torch.FloatTensor(v_ynp))
    input_size = v_x.shape[2]
    output_shape = v_y.shape[2]
    lstm = mdnLSTM(input_size=input_size, hidden_size=hidden_size, number_mixtures=number_mixtures).to(DEVICE)
    optim = torch.optim.Adam(lstm.parameters(), lr=args.learning_rate)

    model_save_name = 'model'
    if args.load:
        if not os.path.exists(args.model_loadname):
            print("load model: %s does not exist"%args.model_loadname)
            sys.exit()
        else:
            print("loading %s" %args.model_loadname)
            lstm_dict = torch.load(args.model_loadname)
            lstm.load_state_dict(lstm_dict['state_dict'])
            optim.load_state_dict(lstm_dict['optimizer'])
            train_cnts = lstm_dict['train_cnts']
            train_losses = lstm_dict['train_losses']
            test_cnts = lstm_dict['test_cnts']
            test_losses = lstm_dict['test_losses']

    loop(data_loader, save_every=save_every, num_epochs=args.num_epochs, train_losses=[], test_losses=[], train_cnts=[], test_cnts=[], dummy=args.dummy)

    embed()


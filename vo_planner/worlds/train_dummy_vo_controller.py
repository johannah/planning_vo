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
rdn = np.random.RandomState(33)
# TODO one-hot the action space?

torch.manual_seed(139)
# David's blog post:
# https://github.com/hardmaru/pytorch_notebooks/blob/master/mixture_density_networks.ipynb
# target is input data shifted by one time step
# input data should be - timestep, batchsize, features!

class LSTM(nn.Module):
    def __init__(self, input_size=9, output_size=2, hidden_size=128):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm1 = nn.LSTMCell(input_size, hidden_size)
        self.lstm2 = nn.LSTMCell(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
    def forward(self, xt, h1_tm1, c1_tm1, h2_tm1, c2_tm1):
        h1_t, c1_t = self.lstm1(xt, (h1_tm1, c1_tm1))
        h2_t, c2_t = self.lstm2(h1_t, (h2_tm1, c2_tm1))
        output = self.linear(h2_t)
        return output, h1_t, c1_t, h2_t, c2_t


def train(x, y, e, do_save=False):
    optim.zero_grad()
    bs = x.shape[1]
    h1_tm1 = Variable(torch.zeros((bs, hidden_size))).to(DEVICE)
    c1_tm1 = Variable(torch.zeros((bs, hidden_size))).to(DEVICE)
    h2_tm1 = Variable(torch.zeros((bs, hidden_size))).to(DEVICE)
    c2_tm1 = Variable(torch.zeros((bs, hidden_size))).to(DEVICE)
    outputs = []
    losses = []
    # one batch of x
    for i in np.arange(0,x.shape[0]):
        xin = x[i]
        output, h1_tm1, c1_tm1, h2_tm1, c2_tm1 = lstm(xin, h1_tm1, c1_tm1, h2_tm1, c2_tm1)
        outputs+=[output]
    y_pred = torch.stack(outputs, 0)
    mse_loss = ((y_pred-y)**2).mean()
    mse_loss.backward()
    losses.append(mse_loss.data)
    clip = 10
    for p in lstm.parameters():
        p.grad.data.clamp_(min=-clip,max=clip)
    optim.step()
    if do_save:
        ll = mse_loss.cpu().data.numpy()
        print('saving after example {} loss {}'.format(e,ll))
        state = {'epoch':e,
                'loss':ll,
                'state_dict':lstm.state_dict(),
                'optimizer':optim.state_dict(),
                 }
        filename = os.path.join(savedir, '%s_%015d.pkl'%(model_save_name,e))
        save_checkpoint(state, filename=filename)
    return y_pred, np.mean(losses)

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
    # [pos0,pos1,speed,angle,steer,gas,brake,diffy,diffx,steering,throttle]
    x_tensor = torch.from_numpy(np.float32(states[:-1,:,[3,4,9,10,7,8]]))
    y_tensor = torch.from_numpy(np.float32(states[1:,:,[7,8]]))
    x_variable = Variable(x_tensor, requires_grad=True)
    y_variable = Variable(y_tensor, requires_grad=False)
    return x_tensor, y_tensor

def teacher_force_predict(x, y):
    # not done
    optim.zero_grad()
    bs = x.shape[1]
    h1_tm1 = Variable(torch.zeros((bs, hidden_size))).to(DEVICE)
    c1_tm1 = Variable(torch.zeros((bs, hidden_size))).to(DEVICE)
    h2_tm1 = Variable(torch.zeros((bs, hidden_size))).to(DEVICE)
    c2_tm1 = Variable(torch.zeros((bs, hidden_size))).to(DEVICE)
    outputs = []
    # one batch of x
    for i in np.arange(0,x.shape[0]):
        output, h1_tm1, c1_tm1, h2_tm1, c2_tm1 = lstm(x[i], h1_tm1, c1_tm1, h2_tm1, c2_tm1)
        outputs+=[output]
    y_pred = torch.stack(outputs, 0)
    mse_loss = ((y_pred-y)**2)
    losses = list(mse_loss.data.numpy())
    return y_pred, losses

def predict(x, y):
    # not done
    optim.zero_grad()
    bs = x.shape[1]
    h1_tm1 = Variable(torch.zeros((bs, hidden_size)))
    c1_tm1 = Variable(torch.zeros((bs, hidden_size)))
    h2_tm1 = Variable(torch.zeros((bs, hidden_size)))
    c2_tm1 = Variable(torch.zeros((bs, hidden_size)))
    outputs = []
    input_data = x[0]
    # one batch of x
    for i in np.arange(0,x.shape[0]-1):
        output, h1_tm1, c1_tm1, h2_tm1, c2_tm1 = lstm(input_data, h1_tm1, c1_tm1, h2_tm1, c2_tm1)
        outputs+=[output]
        input_data = x[i+1]
        # replace the offset with the predicted offset
        input_data[:,-2:] = output
    # do final prediction
    output, h1_tm1, c1_tm1, h2_tm1, c2_tm1 = lstm(input_data, h1_tm1, c1_tm1, h2_tm1, c2_tm1)
    outputs+=[output]

    y_pred = torch.stack(outputs, 0)
    #print(y_pred.shape)
    mse_loss = ((y_pred-y)**2)
    losses = list(mse_loss.data.numpy())
    return y_pred, losses

def loop(xvar,yvar,num_epochs=1000,cnt=0,save_every=10000):
    print("starting training loop")
    st = time.time()
    aloss = []
    last_save = cnt
    for e in range(num_epochs):
        ecnt = 0
        if not e%1 and e>0:
            tst = round((time.time()-st)/60., 0)
            print("starting epoch %s, %s mins after start, loss %s, seen %s" %(e, tst, aloss[-1], cnt))
        batch_loss = []
        for bst in np.arange(0, (xvar.shape[1]-batch_size)+1, batch_size, dtype=np.int):
            xd = xvar[:,bst:bst+batch_size].to(DEVICE)
            yd = yvar[:,bst:bst+batch_size].to(DEVICE)
            if cnt-last_save > save_every:
                print("SAVE TRUE")
                y_pred, mean_loss = train(xd, yd, cnt, True)
                last_save = cnt
            else:
                y_pred, mean_loss = train(xd, yd, cnt, False)
            cnt+=batch_size
            ecnt+=batch_size
            batch_loss.append(mean_loss)
        ## get leftovers
        num_left = xvar.shape[1]-ecnt
        if num_left:
            xd = xvar[:,ecnt:]
            yd = yvar[:,ecnt:]
            y_pred, mean_loss = train(xd, yd, cnt, True)
            batch_loss.append(mean_loss)
            cnt+=num_left
        aloss.append(np.mean(batch_loss))

def valid_loop(function, xvar, yvar):
    aloss = []
    cnt = 0
    vdshape = (yvar.shape[0], yvar.shape[1], yvar.shape[2])
    trues = np.zeros(vdshape)
    predicts = np.zeros(vdshape)
    batch_loss = []
    ecnt = 0
    for bst in np.arange(0, (xvar.shape[1]-batch_size)+1, batch_size, dtype=np.int):
        xd = xvar[:,bst:bst+batch_size].to(DEVICE)
        yd = yvar[:,bst:bst+batch_size].to(DEVICE)
        y_pred, losses = function(xd, yd)

        predicts[:,bst:bst+batch_size,:] = y_pred.detach().numpy()
        trues[:,bst:bst+batch_size,:] = yd.detach().numpy()
        cnt+=batch_size
        ecnt+=batch_size
        batch_loss.extend(losses)
    # get leftovers
    num_left = xvar.shape[1]-ecnt
    if num_left:
        xd = xvar[:,ecnt:]
        yd = yvar[:,ecnt:]
        y_pred, losses = function(xd, yd)
        predicts[:,ecnt:,:] = y_pred.detach().numpy()
        trues[:,ecnt:,:] = yd.detach().numpy()
        cnt+=num_left
        batch_loss.extend(losses)
    return trues, predicts, batch_loss

def plot_results(cnt, vx_tensor, vy_tensor, name='test'):
    # check that feature array for offset is correct from training set
    tf_trues, tf_predicts, tf_batch_loss = valid_loop(teacher_force_predict, vx_tensor, vy_tensor)
    trues, predicts, pbatch_loss = valid_loop(predict, vx_tensor, vy_tensor)
    if not os.path.exists(img_savedir):
        os.makedirs(img_savedir)
    for e in range(trues.shape[1]):
        #gt_y = np.cumsum(vy_tensor[:,e,-2].detach().numpy())
        #gt_x = np.cumsum(vy_tensor[:,e,-1].detach().numpy())

        ugty = np.cumsum(trues[:,e,0])
        ugtx = np.cumsum(trues[:,e,1])
        tfy = np.cumsum(tf_predicts[:,e,0])
        tfx = np.cumsum(tf_predicts[:,e,1])
        py = np.cumsum(predicts[:,e,0])
        px = np.cumsum(predicts[:,e,1])

        xmin = np.min([px.min(), tfx.min(), ugtx.min()])
        xmax = np.min([px.max(), tfx.max(), ugtx.max()])
        ymin = np.min([py.min(), tfy.min(), ugty.min()])
        ymax = np.max([py.max(), tfy.max(), ugty.max()])
        f,ax=plt.subplots(1,3, figsize=(9,3))
        ## original coordinates
        ax[0].scatter(ugty,ugtx, c=np.arange(ugty.shape[0]))
        ax[0].set_xlim([xmin,xmax])
        ax[0].set_ylim([ymin,ymax])
        ax[0].set_title("target")
        ax[1].scatter(tfy,tfx, c=np.arange(tfy.shape[0]))
        ax[1].set_xlim([xmin,xmax])
        ax[1].set_ylim([ymin,ymax])
        ax[1].set_title('teacher force predict')
        ax[2].scatter(py, px, c=np.arange(py.shape[0]))
        ax[2].set_xlim([xmin,xmax])
        ax[2].set_ylim([ymin,ymax])
        ax[2].set_title('predict')
        filename = os.path.join(img_savedir, 'model_epoch_%015d_%s_%05d.png'%(cnt,name,e))
        plt.savefig(filename)
        plt.close()

if __name__ == '__main__':
    import argparse
    hidden_size = 1024
    batch_size = 10
    output_size = 2
    savedir = 'models'
    img_savedir = 'predictions'
    cnt = 0
    model_save_name = 'model'
    model_load_path = 'models/model_epoch_000000100000150.pkl'
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('-dp', '--do_plot', action='store_true', default=True)
    parser.add_argument('-po', '--plot_only', action='store_true', default=False)
    parser.add_argument('-m', '--model_loadname', default=model_load_path)
    parser.add_argument('-se', '--save_every',
                        default=1e7, help='how often to save training model')
    parser.add_argument('--limit', default=-1, type=int, help='limit training data to reduce convergence time')
    parser.add_argument('-l', '--load', action='store_true', default=False, help='load model to continue training or to generate. model path is specified with -m')
    parser.add_argument('-v', '--validate', action='store_true', default=False, help='test results')

    args = parser.parse_args()

    if args.cuda:
        DEVICE = 'cuda'
    else:
        DEVICE = 'cpu'

    if args.limit != -1:
        model_save_name += "_limit_%04d"%args.limit
    mse_loss = nn.MSELoss().to(DEVICE)

    save_every = args.save_every
    # load train and test set
    x_tensor, y_tensor = load_data("train_2d_controller.npz")
    valid_x_tensor, valid_y_tensor = load_data("test_2d_controller.npz")
    input_size = x_tensor.shape[2]
    lstm = LSTM(input_size=input_size, output_size=output_size, hidden_size=hidden_size).to(DEVICE)
    optim = torch.optim.Adam(lstm.parameters(), lr=1e-5)
    limit = min(x_tensor.shape[1], args.limit)
    if args.load:
        if not os.path.exists(model_load_path):
            print("load model: %s does not exist"%model_load_path)
            sys.exit()
        else:
            print("loading %s" %model_load_path)
            lstm_dict = torch.load(model_load_path)
            lstm.load_state_dict(lstm_dict['state_dict'])
            optim.load_state_dict(lstm_dict['optimizer'])
            cnt = lstm_dict['epoch'] # epoch is actually count
            # todo losses

    if not args.plot_only:
        loop(x_tensor[:,:limit], y_tensor[:,:limit], save_every=save_every, cnt=cnt)

    if args.do_plot or args.plot_only:
        plot_results(cnt, valid_x_tensor[:,:batch_size], valid_y_tensor[:,:batch_size], name='test')
        plot_results(cnt,       x_tensor[:,:batch_size],      y_tensor[:,:batch_size], name='train')

    embed()


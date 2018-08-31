import torch 
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
from rudder_utils import TriangularValueEncoding, generate_sample, truncated_normal, RudderLSTM

from IPython import embed

def train(e,do_save=False):
    o.zero_grad()
    bhsize = batch_size, hidden_size
    # TODO - these should be started with truncated_normal_init
    h1_tm1 = Variable(torch.FloatTensor(np.zeros(bhsize)), requires_grad=False)
    c1_tm1 = Variable(torch.FloatTensor(np.zeros(bhsize)), requires_grad=False)
    h2_tm1 = Variable(torch.FloatTensor(np.zeros(bhsize)), requires_grad=False)
    c2_tm1 = Variable(torch.FloatTensor(np.zeros(bhsize)), requires_grad=False)
    outputs = []
    # one batch of x
    sam = generate_sample(max_timestep,n_features,ending_frames,rdn)
    for ts in range(max_timestep):
        dval = torch.tensor(ts, dtype=torch.int32)
        tse = timestep_encoder.encode_value(dval)[None,:]
        out = rudder(sam['actions'][ts], sam['states'][ts], tse,  h1_tm1, c1_tm1, h2_tm1, c2_tm1)
        output, h1_tm1, c1_tm1, h2_tm1, c2_tm1 = out
        outputs+=[output]
    y_pred = torch.stack(outputs, 0)
    embed()
    mse_loss = ((y_pred-sam['rewards'])**2).mean()
    mse_loss.backward()
    #clip = 10
    #for p in rnn.parameters():
    #    p.grad.data.clamp_(min=-clip,max=clip)

    #o.step()
    #if not e%100:
    #    ll = mse_loss.cpu().data.numpy()
    #    print('saving epoch {} loss {}'.format(e,ll))
    #    if np.isnan(ll[0]):
    #        embed()
    #    state = {'epoch':e, 
    #            'loss':ll,
    #            'state_dict':rnn.state_dict(), 
    #            'optimizer':o.state_dict(), 
    #             }
    #    filename = os.path.join(savedir, 'model_epoch_%06d.pkl'%e)
    #    save_checkpoint(state, filename=filename)
    return y_pred


#
# Set up an example environment
#
lstm_input_size = 20
lr = 1e-5
max_timestep = 50
n_mb = 1
n_features = 13
n_actions = 2
ending_frames = 10
hidden_size = 8
rnd_seed = 123
triangle_size = 10
torch.manual_seed(rnd_seed)
rdn = np.random.RandomState(rnd_seed)
batch_size = 1
timestep_encoder = TriangularValueEncoding(max_value=max_timestep, 
                                          triangle_span=int(max_timestep/triangle_size))

rudder = RudderLSTM(hidden_size=hidden_size)
o = optim.Adam(rudder.parameters(), lr=lr)


for e in range(1):
    train(e)
embed()

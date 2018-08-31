import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from IPython import embed

class TriangularValueEncoding(object):
    def __init__(self, max_value, triangle_span):
        """ from TeLL - Encodes an integer value with range [0, max_value] as multiple activations between 0 and 1 via triangles of
        width triangle_span;
        LSTM profits from having an integer input with large range split into multiple input nodes; This class encodes
        an integer as multiple nodes with activations of range [0,1]; Each node represents a triangle of width
        triangle_span; These triangles are distributed equidistantly over the integer range such that 2 triangles
        overlap by 1/2 width and the whole integer range is covered; For each integer to encode, the high of the
        triangle at this integer position is taken as node activation, i.e. max. 2 nodes have an activation > 0 for each
        integer value;
        Values are encoded via encode_value(value) and returned as float32 tensorflow tensor of length self.n_nodes;
        Parameters
        ----------
        max_value : int
            Maximum value to encode
        triangle_span : int
            Width of each triangle
        """
        # round max_value up to a multiple of triangle_span
        if max_value % triangle_span != 0:
            max_value = ((max_value // triangle_span) + 1) * triangle_span
        # Calculate number of overlapping triangle nodes
        n_nodes_half = int(max_value // triangle_span)
        n_nodes = n_nodes_half * 2 + 1
        self.n_nodes_python = n_nodes
        self.n_nodes = torch.tensor(n_nodes, dtype=torch.int32)
        # Template for tensor
        self.coding1 = torch.zeros((n_nodes,), dtype=torch.float32)
        self.coding2 = torch.zeros((n_nodes,), dtype=torch.float32)
        self.n_nodes_half = torch.tensor(n_nodes_half, dtype=torch.int32)
        self.max_value = torch.tensor(max_value, dtype=torch.int32)
        self.triangle_span = torch.tensor(triangle_span, dtype=torch.int32)
        self.triangle_span_float = torch.tensor(triangle_span, dtype=torch.float32)
        self.half_triangle_span = torch.tensor(self.triangle_span / 2.0, dtype=torch.int32)
        self.half_triangle_span_float = torch.tensor(self.triangle_span.float() / 2.0, dtype=torch.float32)
        
    def encode_value(self, value):
        """ Encode value as multiple triangle node activations
        
        Parameters
        ----------
        value : int tensor
            Value to encode as integer tensorflow tensor
        
        Returns
        ----------
        float32 tensor
            Encoded value as float32 tensor of length self.n_nodes
        """
        value_float = value.float()
        tabs = torch.abs(value_float-self.half_triangle_span_float)
        modr = torch.fmod(tabs, self.triangle_span_float)

        index1 = (value / self.triangle_span).int()
        index2 = ((value + self.half_triangle_span)//(self.triangle_span)) + self.n_nodes_half
        # reinitialize coding
        self.coding1*=0.0
        self.coding2*=0.0
        if index1<self.coding1.shape[0]:
            act1 = (0.5-(modr/self.triangle_span_float))*2.0
            self.coding1[index1] = act1
        if index2<self.coding2.shape[0]:
            act2 = (modr/self.triangle_span_float)*2.0
            self.coding2[index2] = act2
        return self.coding1 + self.coding2

def generate_sample(max_timestep,n_features,ending_frames,rnd_gen):
    """Create sample episodes from our example environment"""
    # Create random actions
    actions = np.asarray(rnd_gen.randint(low=0, high=2, size=(max_timestep,)), dtype=np.float32)
    actions_onehot = torch.zeros((max_timestep, 2), dtype=torch.float32)
    inds = np.arange(actions.shape[0])
    actions_onehot[inds[actions == 0], 0] = 1
    actions_onehot[inds[actions == 1], 1] = 1
    actions += actions - 1
    # Create states to actions, make sure agent stays in range [-6, 6]
    states = np.zeros_like(actions)
    for i, a in enumerate(actions):
        if i == 0:
            states[i] = a
        else:
            states[i] = np.clip(states[i - 1] + a, a_min=-int(n_features / 2), a_max=int(n_features / 2))
    
    # Check when agent collected a coin (=is at position 2)
    coin_collect = np.asarray(states == 2, dtype=np.float32)
    # Move all reward to position 50 to make it a delayed reward example
    coin_collect[-1] = np.sum(coin_collect)
    coin_collect[:-1] = 0
    rewards = torch.FloatTensor(coin_collect)
    # Padd end of game sequences with zero-states
    astates = np.asarray(states, np.int) + int(n_features / 2)
    states_onehot = torch.zeros((len(rewards) + ending_frames, n_features), dtype=torch.float32)
    states_onehot[np.arange(len(rewards)), astates] = 1
    actions_onehot = torch.cat((actions_onehot, torch.zeros_like(actions_onehot[:ending_frames])),0)
    rewards = torch.cat((rewards, torch.zeros_like(rewards[:ending_frames])),0)
    # Return states, actions, and rewards
    return dict(states=states_onehot[:, None, :], actions=actions_onehot[:, None, :], rewards=rewards[:, None, None])

def truncated_normal(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)

class RudderLSTM(nn.Module):
    def __init__(self, hidden_size=128, head_output_size=16, 
                       lstm_input_size=42,
                       action_input_size=2, state_input_size=13, 
                       time_input_size=21):
        super(RudderLSTM,self).__init__()
        self.action_head = nn.Linear(action_input_size, head_output_size)
        self.state_head = nn.Linear(state_input_size,  head_output_size)
        self.time_head = nn.Linear(time_input_size, head_output_size)
        self.reward_redistribution_layer = nn.Linear(head_output_size*3,lstm_input_size)
        # (1,1,big) going into lstm
        self.lstm1 = nn.LSTMCell(42,hidden_size)
        self.lstm2 = nn.LSTMCell(hidden_size,hidden_size)
        self.linear = nn.Linear(hidden_size,1)

    def forward(self, action_hot, state_hot, time_tri, h1_tm1, c1_tm1, h2_tm1, c2_tm1):
        ao = self.action_head(action_hot)
        so = self.state_head(state_hot)
        to = self.time_head(time_tri)
        concat_in = torch.cat((ao,so,to),-1)
        xt = self.reward_redistribution_layer(concat_in)
        # h1 is 1,hidden_size
        h1_t, c1_t = self.lstm1(xt,(h1_tm1,c1_tm1))
        h2_t, c2_t = self.lstm2(h1_t, (h2_tm1, c2_tm1))
        output = self.linear(h2_t)
        return output, h1_t, c1_t, h2_t, c2_t


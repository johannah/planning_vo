import numpy as np
import sys, os
from IPython import embed

rdn = np.random.RandomState(3354)
if not len(sys.argv) == 2:
    print('please pass name of npz file with all data')
    print('passed', sys.argv)
    sys.exit()
fname = sys.argv[1]
raw = np.load(fname)
states = raw['states']; names = raw['bagnames']; k = raw['keys'].item()
print('loaded data of shape', states.shape)
num = states.shape[1]
ts = states.shape[0]
raw_feats = states.shape[2]
xdiff = np.insert(np.diff(states[:,:,k['ml_x']], axis=0), 0, 0.0, axis=0)[:,:,None]
ydiff = np.insert(np.diff(states[:,:,k['ml_y']], axis=0), 0, 0.0, axis=0)[:,:,None]

dsoxdiff = np.insert(np.diff(states[:,:,k['dso_x']], axis=0), 0, 0.0, axis=0)[:,:,None]
dsoydiff = np.insert(np.diff(states[:,:,k['dso_y']], axis=0), 0, 0.0, axis=0)[:,:,None]

# one hot dso status
# stat good is one where it is good
stat_good = np.zeros((ts, num, 1), dtype=np.float32)
# stat bad is one where it is bad
stat_bad = np.ones((ts, num, 1), dtype=np.float32)
good_dso = states[:,:,k['dso_stat']].astype(np.int) == 1
stat_good[good_dso] = 1
stat_bad[good_dso] = 0
# add in our new states
cnt = raw_feats
k['ml_xdiff'] = cnt
cnt+=1
k['ml_ydiff'] = cnt
cnt+=1
k['dso_xdiff'] = cnt
cnt+=1
k['dso_ydiff'] = cnt
cnt+=1
k['stat_good'] = cnt
cnt+=1
k['stat_bad'] = cnt

more_states = np.concatenate((states, xdiff, ydiff,
                              dsoxdiff, dsoydiff,
                              stat_good, stat_bad), axis=2)
# break into train and test set
narray = np.arange(num)
nmask = np.ones(num)
num_test = int(num*.15)
test_inds = rdn.choice(narray, num_test, replace=False)
nmask[test_inds] = 0
train_inds = narray[nmask == 1]
test_data = more_states[:,test_inds]
train_data = more_states[:,train_inds]

bname = os.path.split(fname)[1]
train_name = bname.replace('.npz', '_train.npz')
test_name = bname.replace('.npz', '_test.npz')
np.savez(train_name, states=train_data, keys=k, bagnames=names[train_inds], inds=train_inds)
np.savez(test_name, states=test_data, keys=k, bagnames=names[test_inds], inds=test_inds)


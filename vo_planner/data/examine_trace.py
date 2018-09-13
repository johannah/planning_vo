import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
import os, sys
from glob import glob
search = 'icra2019_vehicle_data*.npz'
# set to false to remove pandas dependency
add_bagnames = True
files = sorted(glob(search))
if not len(files):
    print("no files at", search)
    sys.exit()
else:
    #raw = np.load('icra2019_vehicle_data_2018-09-12_0643_files.npz')
    # TODO sort on date, then size
    if len(files) > 1:
        print('i dont know which file to use.....')
        print('set index = to the file you want to use, then type exit()')
        print(files)
        embed()
    else:
        index = 0
    fname = files[index]
    raw = np.load(fname)
states = raw['states']
names = raw['bagnames']
k = raw['keys'].item()

good = []

def plot_trace(i, t, path):
    f, ax = plt.subplots(1,2, figsize=(18,10))
    true = t[:,[k['ml_x'], k['ml_y']]]
    pred = t[:,[k['dso_x'], k['dso_y']]]
    stat = t[:, k['dso_stat']].astype(np.int32)
    ax[0].plot(true[:,0], true[:,1], c='b', lw=3, label='true')
    good = pred[stat==1]
    num_good = good.shape[0]
    ax[0].scatter(pred[:,0], pred[:,1], c='y', s=3, label='bad dso')
    ax[0].scatter(good[:,0], good[:,1], c='g', s=3, label='tracking dso')
    ax[0].legend()
    aa = np.arange(true.shape[0])
    st=ax[1].plot(aa,t[:,k['steering']], label='steer')
    le=ax[1].plot(aa,t[:,k['throttle']], label='throttle')
    ax[1].legend()
    plt.suptitle("index %s  - %d good pts" %(i,num_good))
    plt.savefig(path)
    plt.close()
    if num_good > 0 :
        print(path, num_good)
    return num_good

img_dir = 'imgs'
if not os.path.exists(img_dir):
    os.makedirs(img_dir)
good_cnt = 0
for i in range(states.shape[1]):
    if add_bagnames:
        bagname = names[i]
        fname = os.path.join(img_dir, '%strace_%04d.png' %(bagname,i))
    else:
        fname = os.path.join(img_dir, 'trace_%04d.png' %(i))
    good_cnt+=plot_trace(i, states[:,i], fname)
print('all good', good_cnt)


import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
import os, sys
import pandas as pd

raw = np.load('icra2019_vehicle_data_0643_files_2018-09-12.npz')
states = raw['states']
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
    #ax[0].legend()
    aa = np.arange(true.shape[0])
    ax[1].plot(aa,t[:,k['steering']], label='steer') 
    ax[1].plot(aa,t[:,k['throttle']], label='throttle') 
    plt.title("index %s  - %d good pts" %(i,num_good))
    plt.savefig(path)
    plt.close()
    if num_good > 0 :
        print(path, num_good)
    return num_good
 

#    plt.figure() 
#    true = t[:,[k['ml_x'], k['ml_y']]]
#    pred = t[:,[k['dso_x'], k['dso_y']]]
#    stat = t[:, k['dso_stat']].astype(np.int32)
#    plt.plot(true[:,0], true[:,1], c='b', lw=3, label='true')
#    good = pred[stat==1]
#    num_good = good.shape[0]
#    plt.scatter(pred[:,0], pred[:,1], c='y', s=3, label='bad dso')
#    plt.scatter(good[:,0], good[:,1], c='g', s=3, label='tracking dso')
#    plt.legend()
#    plt.title("index %s  - %d good pts" %(i,num_good))
#    plt.savefig(path)
#    plt.close()
#    if num_good > 0 :
#        print(path, num_good)
#    return num_good
    
pan = pd.read_csv('icra2019_vehicle_data_0642_files_2018-09-12.csv')
img_dir = 'imgs'
if not os.path.exists(img_dir):
    os.makedirs(img_dir)
good_cnt = 0
for i in range(states.shape[1]):
    bagname = pan.loc[0,'bag_name']
    fname = os.path.join(img_dir, '%strace_%04d.png' %(bagname,i))
    good_cnt+=plot_trace(i, states[:,i], fname)
print('all good', good_cnt)


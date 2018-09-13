# Thanks to @https://github.com/weidi-chang for creating the core of this code
import numpy as np
from glob import glob
import os, sys
import csv
from IPython import embed
import pandas as pd
from datetime import date

base_dir = '/mnt/mllsynology/icra2019bags/'
files = glob(os.path.join(base_dir, 'sept7', 
                          '*_ref', '*.csv'))
print('found %s files' %len(files))
# ts, dso_x, dso_y, sdso_z,  dso_qx, dso_qy, sdso_qz, sds_qw
# define bags by hand so we know when they are messed up
csv_columns = ['t', 'dso_x', 'dso_y', 'sdso_z', 
               'dso_qx', 'dso_qy', 'dso_qz', 'dso_qw', 'dso_yaw', 'dso_stat', 
               'ml_x', 'ml_y', 'ml_z', 
               'ml_qx', 'ml_qy', 'ml_qz', 'ml_qw', 'ml_yaw', 
               'steering', 'throttle', 'score', 'img_name'
               ]
np_columns = ['dso_x', 'dso_y', 'sdso_z', 
              'dso_qx', 'dso_qy', 'dso_qz', 'dso_qw', 'dso_yaw', 'dso_stat', 
              'ml_x', 'ml_y', 'ml_z', 
              'ml_qx', 'ml_qy', 'ml_qz', 'ml_qw', 'ml_yaw', 
              'steering', 'throttle', 'score'
               ]

np_keys = {}
for (c,k) in enumerate(np_columns): np_keys[k]=c 

data = []
for cnt, f in enumerate(sorted(files)):
    readfile = pd.read_csv(f, names=csv_columns, skiprows=1)
    fname = os.path.split(f)[1]
    readfile['bag_name'] = fname
    data.append(np.array(readfile[np_columns]))
    if not cnt:
        out = readfile
    else:
        out = out.append(readfile)

bname = 'icra2019_vehicle_data_%04d_files_%s' %(len(files), str(date.today()))
data = np.swapaxes(np.array(data),0,1)
out.to_csv(bname+'.csv', header=True, index=True, index_label='N')
np.savez(bname, states=data, keys=np_keys)
print('finished loading all files, data is shape', out.shape)

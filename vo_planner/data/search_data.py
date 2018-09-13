# Thanks to @https://github.com/weidi-chang for creating the core of this code
import numpy as np
from glob import glob
import os, sys
import csv
from copy import deepcopy
from IPython import embed
import pandas as pd
from datetime import date

# jhansen@erehwon:/mnt/mllsynology2/icra2019bags/sept12$ rsync -avhp  /mnt/mllsynology_usb/icra2019bags/sept12/*  --include *.csv --exclude *.pickle  --exclude *.jpg --exclude *.bag --exclude useless --exclude *img  .
# super hack because the way these files are organized is pretty whack
# TODO enforce some structure

base_dir = '/mnt/mllsynology2/icra2019bags/'
s7 = os.path.join(base_dir, 'sept7', 'icra2019_sep7_csv_data_dso_pose_corrected', '*.csv')
files7 = glob(s7)

s12 = os.path.join(base_dir, 'sept12', '*_ref', '*.csv')
files12 = glob(s12)

files = files7+files12
print('found %s files - sept 7 %s sept 12 %s' %(len(files), len(files7), len(files12)))
if not len(files7):
    print(s7)
    sys.exit()
if not len(files12):
    print(s12)
    sys.exit()
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
names = []
for cnt, f in enumerate(sorted(files)):
    readfile = pd.read_csv(f, names=csv_columns, skiprows=1)
    fname = os.path.split(f)[1]
    print('working', fname)
    readfile['bag_name'] = fname
    data.append(np.array(readfile[np_columns]))
    names.append(fname)
    if not cnt:
        out = readfile
    else:
        out = out.append(deepcopy(readfile))

bname = 'icra2019_vehicle_data_%s_%04d_files' %(str(date.today()), len(files))
data = np.swapaxes(np.array(data),0,1)
out.to_csv(bname+'.csv', header=True, index=True, index_label='N')
print('finished loading all files, data is shape', data.shape)
print('finished loading all files, big csv is shape', out.shape)
np.savez(bname, states=data, keys=np_keys, bagnames=names)

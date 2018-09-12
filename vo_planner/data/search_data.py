import numpy as np
from glob import glob
import os, sys
import csv
from IPython import embed

base_dir = '/mnt/mllsynology/icra2019bags/'
files = glob(os.path.join(base_dir, '*', '*cooked_data', '*',  '*.csv'))
print('found %s files' %len(files))

data = []
for f in files:
    with open(f, 'r') as fh:
        episode = []
        csv_reader = csv.reader(fh, delimiter=',')
        n_lines = 0
        next(csv_reader, None)
        for row in csv_reader:
            episode.append(row[1:-1])
            n_lines +=1
    data.append(episode)

states = np.array(data, dtype=np.float32)
states = np.swapaxes(states, 0, 1)
np.savez('icra2019_vehicle_data', states=states)
embed()

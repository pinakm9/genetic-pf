"""
Finds a trajectory on the attractor of L96_10
"""
# add modules folder to Python's search path
import sys
from pathlib import Path
from os.path import dirname, realpath
script_dir = Path(dirname(realpath(__file__)))
module_dir = str(script_dir.parent)
sys.path.insert(0, module_dir + '/modules')

# import remaining modules
import numpy as np
import Lorenz96_alt
import pandas as pd

# set random initial point, load the L96_10 model
np.random.seed(42)
x0 = np.random.uniform(size=10)
model, gen_path = Lorenz96_alt.get_model(x0=x0, size=10, obs_gap=0.1)
length = 10000

# find a trajectory on the attractor
total_iters = int(1e5)
batch_size = int(1e4)
for i in range(int(total_iters/batch_size)):
    print('Working on batch #{}'.format(i), end='\r')
    hidden_path = gen_path(x0, batch_size)
    x0 = hidden_path[-1]

pd.DataFrame(gen_path(x0, length)).to_csv('attractor_{}.csv'.format(length), header=None, index=None)

    
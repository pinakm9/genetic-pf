# add models folder to Python's search path
import sys
from pathlib import Path
from os.path import dirname, abspath
script_dir = Path(abspath(''))
module_dir = str(script_dir.parent)
sys.path.insert(0, module_dir + '/models')
sys.path.insert(0, module_dir + '/modules')

import numpy as np
import Lorenz96_alt as l96
import config as cf
import filter as fl
import swarm_pf as sfl
import copy, os

# set parameters 
dims = range(3, 33)
max_seed = 2021
config = {}
config['prior_cov'] = 1.0
config['shift'] = 2.0
config['obs_gap'] = 0.1
config['obs_cov'] = 0.1
config['asml_steps'] = 50
batch_id = 1
results_folder = '../data/batch_{}'.format(batch_id) 
if not os.path.isdir(results_folder):
    os.mkdir(results_folder)

for d in dims:
    # set model
    seed = np.random.randint(max_seed)
    np.random.seed()
    config['seed'] = seed
    x0 = np.random.uniform(size=d)
    model, gen_path = l96.get_model(x0=x0, size=config['asml_steps'],\
                                     prior_cov=config['prior_cov'],\
                                     obs_cov=config['obs_cov'], shift=config['shift'],\
                                     obs_gap=config['obs_gap'])
    true_trajectory = gen_path(x0, config['asml_steps'])
    observed_path = model.observation.generate_path(true_trajectory)
    expr_name = 'Lorenz96_alt_{}'.format(d)
    # set filters
    spf_config = {}
    spf_config['max_iterations_per_step'] = 100
    spf_config['inertia'] = 1.0
    spf_config['cognitive_coeff'] = 2.0
    spf_config['social_coeff'] = 2.0
    spf_config['particle_count'] = 500
    spf_config['folder'] = results_folder + '/spf_{}'.format(d)
    spf = sfl.SwarmPF(model, **spf_config)
    
    bpf_config = {}
    bpf_config['particle_count'] = 500
    bpf_config['folder'] = results_folder + '/bpf_{}'.format(d)
    bpf = fl.ParticleFilter(model, **bpf_config)

    # run filters
    spf_config['regeneration_threshold'] = 0.1
    print('assimilating with genetic filter, dimension -> {}'.format(d), end='\r')
    spf.update(observed_path, threshold_factor=spf_config['regeneration_threshold'])
    bpf_config['resampling_method'] = 'systematic_noisy'
    bpf_config['resampling_threshold'] = 1.0
    bpf_config['resampling_noise'] = 1.0
    print('assimilating with particle filter, dimension -> {}'.format(d), end='\r')
    bpf.update(observed_path, threshold_factor=bpf_config['resampling_threshold'],\
               resampling_method = bpf_config['resampling_method'],\
               noise = bpf_config['resampling_noise'])
                
    # document results
    spf.plot_trajectories(true_trajectory, coords_to_plot=[0, 1, 2],\
                                    file_path=spf.folder + '/trajectories.png', measurements=False)
    spf.compute_error(true_trajectory)
    spf.plot_error(semilogy=True, resampling=False)
    cc = cf.ConfigCollector(expr_name = expr_name, folder = spf_config['folder'])
    config_all = {**config, **spf_config} 
    config_all['status'] = spf.status
    cc.add_params(config_all)
    cc.write(mode='json')
    print('spf-{} was a {}'.format(d, spf.status))
    
    bpf.plot_trajectories(true_trajectory, coords_to_plot=[0, 1, 2],\
                                    file_path=bpf.folder + '/trajectories.png', measurements=False)
    bpf.compute_error(true_trajectory)
    bpf.plot_error(semilogy=True, resampling=False)
    cc = cf.ConfigCollector(expr_name = expr_name, folder = bpf_config['folder'])
    config_all = {**config, **bpf_config} 
    config_all['status'] = bpf.status
    cc.add_params(config_all)
    cc.write(mode='json')
    print('bpf-{} was a {}'.format(d, bpf.status))
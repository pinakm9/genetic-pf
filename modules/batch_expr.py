# add models folder to Python's search path
import sys
from pathlib import Path
from os.path import dirname, realpath
script_dir = Path(dirname(realpath(__file__)))
model_dir = str(script_dir.parent)
sys.path.insert(0, model_dir + '/models')

import numpy as np 
import json
import filter as fl
import config as cf
import Lorenz63_xz as model
import copy
import os
import tables
import tensorflow as tf
import wasserstein as ws
import utility as ut
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
from matplotlib.ticker import MaxNLocator

class BPFBatchObs:
    """
    Runs a filtering experiment for a batch of observation realizations 
    """
    def __init__(self, config_file, true_trajectory, seeds, results_folder):
        self.true_trajectory = true_trajectory
        self.config_id = os.path.basename(config_file).split('.')[0]
        with open(config_file) as f:
            self.config = json.load(f)
        self.seeds = seeds 
        self.model = model.get_model(x0=self.true_trajectory[0], size=len(true_trajectory),\
                                     prior_cov=self.config['prior_cov'],\
                                     obs_cov=self.config['obs_cov'], shift=self.config['shift'],\
                                     obs_gap=self.config['obs_gap'])[0]
        self.results_folder = results_folder
        self.config['assimilation_steps'] = len(true_trajectory)

    def run_single_expr(self, seed):
        # set random seed
        np.random.seed(seed)
        # generate observation
        observed_path = self.model.observation.generate_path(self.true_trajectory)
        # set up logging
        config = copy.deepcopy(self.config)
        config['seed'] = seed
        expr_name = '{}_seed_{}'.format(self.config_id, seed)
        cc = cf.ConfigCollector(expr_name = expr_name, folder = self.results_folder)
        
        # assimilate
        #np.random.seed(0)
        self.bpf = fl.ParticleFilter(self.model, particle_count = self.config['particle_count'], folder = cc.res_path)
        print("starting assimilation ... ")
        self.bpf.update(observed_path, method = 'mean', resampling_method=self.config['resampling_method'],\
                        threshold_factor=self.config['resampling_threshold'], noise=self.config['resampling_cov'])
        # document results
        if self.bpf.status == 'success':
            self.bpf.plot_trajectories(self.true_trajectory, coords_to_plot=[0, 1, 2],\
                                       file_path=cc.res_path + '/trajectories.png', measurements=False)
            self.bpf.compute_error(self.true_trajectory)
            self.bpf.plot_error(semilogy=True, resampling=False)
            config['status'] = self.bpf.status
            cc.add_params(config)
            cc.write(mode='json')

    @ut.timer
    def run(self):
        for seed in self.seeds:
            self.run_single_expr(seed)


class BatchDist:
    """
    Computes distance for a batch of experiments
    """
    def __init__(self, config_folder, seeds, results_folder, dist_folder):
        self.configs = [f.split('.')[0] for f in os.listdir(config_folder)]
        self.seeds = seeds
        self.results_folder = results_folder
        self.dist_folder = dist_folder
    
    @ut.timer
    def run_for_pair(self, config_id_1, config_id_2, seed, gap=4, ev_time=400, epsilon=0.01, num_iters=200, p=2):
        # find the right folders
        for f in os.listdir(self.results_folder):
            if f.startswith(config_id_1) and f[:-2].endswith(str(seed)):
                folder_1 = self.results_folder + '/' + f
            if f.startswith(config_id_2) and f[:-2].endswith(str(seed)):
                folder_2 = self.results_folder + '/' + f
        # figure out number of assimilation steps
        #with open(folder_1 + '/config.json') as f:
        #    ev_time = json.load(f)['assimilation_steps']
        
        file_1 = tables.open_file(folder_1 + '/assimilation.h5')
        file_2 = tables.open_file(folder_2 + '/assimilation.h5')

        dist = np.zeros(int(ev_time / gap))
        for i, t in enumerate(range(0, ev_time, gap)):
            print('computing distance for step #{}'.format(t))
            ensemble_1 = np.array(getattr(file_1.root.particles, 'time_' + str(t)).read().tolist())
            ensemble_2 = np.array(getattr(file_2.root.particles, 'time_' + str(t)).read().tolist())
            #weights_1 = np.array(getattr(file_1.root.weights, 'time_' + str(t)).read().tolist())
            #weights_2 = np.array(getattr(file_2.root.weights, 'time_' + str(t)).read().tolist())
            ensemble_1 = tf.convert_to_tensor(ensemble_1, dtype=tf.float32)
            ensemble_2 = tf.convert_to_tensor(ensemble_2, dtype=tf.float32)
            #weights_1 = tf.convert_to_tensor(weights_1, dtype=tf.float32)
            #weights_2 = tf.convert_to_tensor(weights_2, dtype=tf.float32)
            dist[i] = (ws.sinkhorn_div_tf(ensemble_1, ensemble_2,\
                       epsilon=epsilon, num_iters=num_iters, p=p).numpy())**(1./p)

        id_1 = config_id_1.split('_')[-1]
        id_2 = config_id_2.split('_')[-1]
        file_path = '{}/{}_vs_{}_seed_{}.npy'.format(self.dist_folder, id_1, id_2, seed)
        #np.save(file_path, dist)
        file_1.close()
        file_2.close()
        return dist

    @ut.timer
    def run(self, gap=4, ev_time=400, epsilon=0.01, num_iters=200, p=2):
        for j, config_id_1 in enumerate(self.configs):
            for config_id_2 in self.configs[j+1:]:
                data = {'time': [], 'seed':[], 'sinkhorn_div': []}
                for i, seed in enumerate(self.seeds):
                    print('comparing {} and {} for seed: {}'.format(config_id_1, config_id_2, seed))
                    dist = self.run_for_pair(config_id_1, config_id_2, seed, gap, ev_time, epsilon, num_iters, p)
                    data['time'] += list(range(0, ev_time, gap))
                    data['seed'] += [seed] * len(dist) 
                    data['sinkhorn_div'] += list(dist)
                df = pd.DataFrame(data)
                id_1 = config_id_1.split('_')[-1]
                id_2 = config_id_2.split('_')[-1]
                df.to_csv('{}/{}_vs_{}.csv'.format(self.dist_folder, id_1, id_2), index=False)


class AvgDistPlotter:
    """
    Plots average distance for same number of particles
    """
    def __init__(self, dist_folder, inset_dist_folder=None):
        self.dist_folder = dist_folder
        self.inset_dist_folder = inset_dist_folder
        # sort folders according to particle counts in ascending order
        self.folders = sorted(os.listdir(dist_folder))
        self.particle_counts = [int(f.split('_')[-1]) for f in self.folders]
        self.particle_counts, self.folders = zip(*sorted(zip(self.particle_counts, self.folders)))
        # set colors and line styles
        self.colors = ['red', 'green', 'blue', 'orange', 'grey', 'purple']
        self.line_styles = [':', '-.', '--', '-']
        

    def plot(self, save_path, gap=4, ev_time=400, low_idx=0, high_idx=None, pc_idx=None, inset=False, ev_time2=20, y_lims=[None, None]):
        with plt.style.context('seaborn-paper'): # 'tableau-colorblind10'
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111)
            ax.tick_params(axis='both', which='major', labelsize=30)
            ax.tick_params(axis='both', which='minor', labelsize=30)
            ax.set_ylim([y_lims[0], y_lims[1]])
            if inset:
                ax_inset = ax.inset_axes([0.1, 0.5, 0.47, 0.47])
                ax_inset.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax_inset.tick_params(axis='both', which='major', labelsize=20)
                ax_inset.tick_params(axis='both', which='minor', labelsize=20)
            if pc_idx is None:
                pc_idx = list(range(len(self.particle_counts)))
            k = 0
            for j, folder in enumerate(self.folders):
                if j not in pc_idx:
                    continue
                dist_files = sorted(os.listdir(self.dist_folder + '/' + folder))
                if inset:
                    inset_dist_files = sorted(os.listdir(self.inset_dist_folder + '/' + folder))
                if high_idx is None:
                    high_idx = len(dist_files)
                for i, f in enumerate(dist_files[low_idx: high_idx]):
                    df = pd.read_csv(self.dist_folder + '/' + folder + '/' + f)
                    df = df.loc[df['time'].isin([k for k in range(0, ev_time, gap)])]
                    label = '#particles = {}'.format(self.particle_counts[j])
                    # set confidence interval
                    if j < len(self.folders) - 1:
                        ci = None
                    else:
                        ci = 'sd'
                    if high_idx - low_idx == 1:
                        color = 'black'
                    else:
                        color = self.colors[i]
                    sns.lineplot(data=df, x=df['time'], y=df['sinkhorn_div'], ci=ci, ax=ax,\
                                    label=label)
                    if inset:
                        df_inset = pd.read_csv(self.inset_dist_folder + '/' + folder + '/' + f)
                        df_inset = df_inset.loc[df_inset['time'].isin([k for k in range(ev_time2)])]
                        sns.lineplot(data=df_inset, x=df_inset['time'], y=df_inset['sinkhorn_div'], ci=ci, ax=ax_inset)
                        ax_inset.set_xlabel('', fontsize=30)
                        ax_inset.set_ylabel('', fontsize=30)
                        
                k += 1
            plt.xlabel('assimilation step (n)', fontsize=30)
            plt.ylabel('$D_{\epsilon}$', fontsize=30)
            id_1, _,  id_2 = f.split('.')[0].split('_')
            plt.title('L63,  $D_\epsilon(\pi_n^P(\mu_{}), \pi_n^P(\mu_{}))$'.format(id_1, id_2), fontsize=40)
            plt.legend(fontsize=30)
            plt.tight_layout()
            plt.savefig(save_path)


class BatchCov:
    """
    Computes highest eigenvalue for analysis covariance
    """
    def __init__(self, config_folder, results_folder, cov_folder):
        self.results_folder = results_folder
        self.asml_folders = sorted(os.listdir(results_folder))
        self.cov_folder = cov_folder
        self.configs = [f[:-5] for f in os.listdir(config_folder)]

    def compute_eigh(self, folder, gap=4, ev_time=400):
        asml = tables.open_file(self.results_folder + '/' + folder + '/assimilation.h5')
        eigh = np.zeros(int(ev_time / gap))
        for i, t in enumerate(range(0, ev_time, gap)):
            print('computing eigenvalue for assimilation step #{}'.format(t), end='\r')
            ensemble = np.array(getattr(asml.root.particles, 'time_' + str(t)).read().tolist()).T
            cov = np.cov(ensemble)
            eigh[i] = scipy.linalg.eigh(cov, subset_by_index=[cov.shape[0]-1, cov.shape[0]-1], eigvals_only=True)[0]
        asml.close()
        return eigh

    def run(self, gap=4, ev_time=400):
        for config in self.configs:
            data = {'time': [], 'seed':[], 'eigh': []}
            for folder in self.asml_folders:
                if folder.startswith(config):
                    seed = int(folder.split('#')[0].split('_')[-1])
                    print('working on {}_seed_{}'.format(config, seed))
                    eigh = self.compute_eigh(folder, gap, ev_time)
                    data['eigh'] += list(eigh)
                    data['time'] += list(range(0, ev_time, gap))
                    data['seed'] += [seed] * len(eigh)
            df = pd.DataFrame(data)
            df.to_csv(self.cov_folder + '/{}.csv'.format(config), index=False)

    def plot(self, save_path):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        colors = ['red', 'green', 'blue', 'orange', 'grey', 'purple']
        for i, f in enumerate(os.listdir(self.cov_folder)):
            df = pd.read_csv(self.cov_folder + '/' + f)
            sns.lineplot(data=df, x=df['time'], y=df['eigh'], color=colors[i], ci=None, ax=ax,\
                         label=f[:-4])
        plt.xlabel('assimilation step')
        plt.ylabel('largest eigenvalue of analysis covarience')
        plt.savefig(save_path)


import numpy as np
import tables
from tables import description
import ipywidgets as widgets
from IPython.display import display 
import matplotlib.pyplot as plt
import warnings

class L2Compare:

    def __init__(self, folder, filter_1, filter_2, dims, fig_size=(8, 8)):
        self.folder = folder
        self.filter_1 = filter_1 
        self.filter_2 = filter_2
        self.asml_file = lambda filter_id, dim: folder + '/{}_{}/assimilation.h5'.format(filter_1 if filter_id==1 else filter_2, dim)
        self.fig_size = fig_size
        hdf5 = tables.open_file(self.asml_file(1, dims[0]), 'r')
        self.observation = np.array(hdf5.root.observation.read().tolist())
        hdf5.close()
        self.num_steps = len(self.observation)
        self.dim_slider = widgets.IntSlider(value=dims[0], min=dims[0], max=dims[-1], step=1, description='dimension')
        widgets.interact(self.error_plot, dim=self.dim_slider)
    
    def error_plot(self, dim):
        fig = plt.figure(figsize=self.fig_size)
        ax = fig.add_subplot(111)
        err_1 = self.get_l2_error(1, dim)
        err_2 = self.get_l2_error(2, dim)
        ax.plot(range(self.num_steps), err_1, label=self.filter_1)
        ax.plot(range(self.num_steps), err_2, label=self.filter_2)
        ax.set_xlabel('assimilation step')
        ax.set_ylabel('l2 error')
        ax.legend()


    def get_l2_error(self, filter_id, dim):
        hdf5 = tables.open_file(self.asml_file(filter_id, dim), 'r')
        error = np.array(hdf5.root.l2_error.read().tolist())
        hdf5.close()
        return error


class GenVsDim:
    def __init__(self, folder, filter, dims, fig_size=(8, 8)):
        self.folder = folder
        self.filter = filter
        self.dims = dims
        self.asml_file = lambda dim: folder + '/{}_{}/assimilation.h5'.format(filter, dim)
        self.fig_size = fig_size
        hdf5 = tables.open_file(self.asml_file(dims[0]), 'r')
        self.observation = np.array(hdf5.root.observation.read().tolist())
        hdf5.close()
        self.num_steps = len(self.observation)
        self.dim_slider = widgets.IntSlider(value=dims[0], min=dims[0], max=dims[-1], step=1, description='dimension')
        widgets.interact(self.generation_plot, dim=self.dim_slider)
    
    def generation_plot(self, dim):
        fig = plt.figure(figsize=self.fig_size)
        ax = fig.add_subplot(111)
        gen = self.get_generation(dim)
        ax.plot(range(self.num_steps), gen, label='gen count')
        ax.plot(range(self.num_steps), np.mean(gen) * np.ones_like(gen), label='avg gen count')
        ax.set_xlabel('assimilation step')
        ax.set_ylabel('number of generations')
        ax.legend()


    def get_generation(self, dim):
        hdf5 = tables.open_file(self.asml_file(dim), 'r')
        error = np.array(hdf5.root.generation.read().tolist())
        hdf5.close()
        return error



class AvgGenVsDim:
    def __init__(self, folder, filter, dims, fig_size=(8, 8), nb_type='jupyter'):
        self.folder = folder
        self.filter = filter
        self.dims = dims
        self.nb_type = nb_type
        self.asml_file = lambda dim: folder + '/{}_{}/assimilation.h5'.format(filter, dim)
        self.avg_gen = np.zeros(len(self.dims))
        for i, dim in enumerate(self.dims):
            self.avg_gen[i] = np.mean(self.get_generation(dim))
        self.fig_size = fig_size
        hdf5 = tables.open_file(self.asml_file(dims[0]), 'r')
        self.observation = np.array(hdf5.root.observation.read().tolist())
        hdf5.close()
        self.num_steps = len(self.observation)
        self.degree_slider = widgets.IntSlider(value=1, min=1, max=10, step=1, description='degree')
        self.button = widgets.Button(description='compute and plot')
        self.button.on_click(self.avg_gen_vs_dim)
        if self.nb_type == 'colab':
            display(self.button, self.degree_slider)
        else:
            display(self.button, self.degree_slider)

    def avg_gen_vs_dim(self, button):
        degree = self.degree_slider.value
        fig = plt.figure(figsize=self.fig_size)
        ax = fig.add_subplot(111)
        ax.plot(self.dims, self.avg_gen, label='avg gen count')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', np.RankWarning)
            poly = np.poly1d(np.polyfit(self.dims, self.avg_gen, degree))
        ax.plot(self.dims, poly(self.dims), label='polyfit deg={}'.format(degree))
        ax.set_xlabel('dimension')
        ax.set_ylabel('avg number of generations')
        ax.legend()
        plt.show()

    def get_generation(self, dim):
        hdf5 = tables.open_file(self.asml_file(dim), 'r')
        error = np.array(hdf5.root.generation.read().tolist())
        hdf5.close()
        return error


class WeightHistogram:

    def __init__(self, folder, filter, dims, fig_size=(8, 8)):
        self.folder = folder
        self.filter = filter
        self.dims = dims
        self.asml_file = lambda dim: folder + '/{}_{}/assimilation.h5'.format(filter, dim)
        self.fig_size = fig_size
        hdf5 = tables.open_file(self.asml_file(dims[0]), 'r')
        self.observation = np.array(hdf5.root.observation.read().tolist())
        hdf5.close()
        self.num_steps = len(self.observation)
        self.dim_slider = widgets.IntSlider(value=dims[0], min=dims[0], max=dims[-1], step=1, description='dimension')
        self.step_slider = widgets.IntSlider(value=0, min=0, max=self.num_steps, step=1, description='asml step')
        widgets.interact(self.histogram, dim = self.dim_slider,step=self.step_slider)
        #widgets.interact(self.histogram, step=self.step_slider)


    def histogram(self, dim, step):
        fig = plt.figure(figsize=self.fig_size)
        ax = fig.add_subplot(111)
        weights = self.get_weights(dim, step)
        ax.hist(weights, bins=10, label='dim={}, step={}'.format(dim, step))
        ax.set_ylabel('weights')
        ax.legend()

    def get_weights(self, dim, step):
        hdf5 = tables.open_file(self.asml_file(dim), 'r')
        weights = np.array(getattr(hdf5.root.weights, 'time_' + str(step)).read().tolist())
        hdf5.close()
        return weights
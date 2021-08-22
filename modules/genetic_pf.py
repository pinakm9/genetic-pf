import filter as fl 
import  numpy as np
import copy
import tables 
import plot 

class GeneticPF(fl.ParticleFilter):

    def __init__(self, model, particle_count, folder = None, particles = None, max_generations_per_step=3,\
                 mutation_prob=0.2, ellitism_factor=0.5, max_population=2000, mutation_size=0.01):
        super().__init__(model=model, particle_count=particle_count, folder=folder, particles=particles)
        self.max_generations_per_step = max_generations_per_step
        self.mutation_prob = mutation_prob 
        self.ellitism_factor = ellitism_factor
        self.max_population = max_population
        self.mutation_size = mutation_size
        self.cov = 0.1 * np.identity(self.dimension)
        tbl = self.hdf5.create_table(self.hdf5.root, 'generation', {'count': tables.Int32Col(pos = 0)})
        tbl.flush()
        self.find_max_weight()

    def find_max_weight(self):
        self.max_log_weight = - 0.5 * np.log(2.0 * np.pi * self.model.observation.sigma[0][0]) * self.model.observation.dimension
        self.max_weight = np.exp(self.max_log_weight)

    def crossover_regular(self, particle_m, particle_d):
        # find a cross-over point
        alpha = np.random.randint(self.dimension)
        beta_1, beta_2 = np.random.uniform(size=2)
        offsprings = [np.zeros(self.dimension) for i in range(4)]
        for j in range(self.dimension):
            if j < alpha:
                offsprings[0][j] = particle_m[j]
                offsprings[1][j] = particle_d[j]
                offsprings[2][j] = particle_m[j]
                offsprings[3][j] = particle_d[j]

            elif  j >= alpha:
                offsprings[0][j] = particle_m[j] - beta_1 * (particle_m[j] - particle_d[j])
                offsprings[1][j] = particle_d[j] + beta_1 * (particle_m[j] - particle_d[j])
                offsprings[2][j] = particle_m[j] - beta_2 * (particle_m[j] - particle_d[j])
                offsprings[3][j] = particle_d[j] + beta_2 * (particle_m[j] - particle_d[j])
        return offsprings + [particle_m, particle_d]

    def crossover(self, particle_m, particle_d):
        # find a cross-over point
        beta = np.random.uniform(size=4)
        offsprings = []
        for i in range(4):
             mu = beta[i] * particle_m + (1.0 - beta[i]) * particle_d
             offsprings.append(np.random.multivariate_normal(mu, self.cov))
        return offsprings + [particle_m, particle_d]

    def mutate(self):
        # find particles to be mutated
        idx = np.random.choice(2, p=[1.0-self.mutation_prob, self.mutation_prob], size=len(self.current_population), replace=True)
        for i, j in enumerate(idx):
            if j == 1:
                alpha = np.random.randint(self.dimension)
                beta = np.random.choice(self.dimension, size=alpha, replace=False)
                self.current_population[i, beta] += np.random.normal(scale=self.mutation_size, size=alpha)

    def breed(self):
        # create a mating pool
        num_mating_pairs = int(self.max_population / 6)
        idx = len(self.current_population)
        idx_m = np.random.choice(idx, size=num_mating_pairs, replace=True)# p=probs)
        idx_d = np.random.choice(idx, size=num_mating_pairs, replace=True)# p=probs)
        # breed
        new_population = []
        for p in range(num_mating_pairs):
            new_population += self.crossover(self.current_population[idx_m[p]], self.current_population[idx_d[p]])
        self.current_population = np.array(new_population) 
        self.mutate()

    def select(self, observation, threshold_factor=0.5):
        self.pop_weights = np.array([self.model.observation.conditional_pdf(self.current_time, observation, particle)\
                        for particle in self.current_population])
        idx = np.where(self.pop_weights >= self.max_weight * threshold_factor)[0]
        if len(idx) > 0:
            a = self.current_selection_size
            b = min(self.current_selection_size + len(idx), self.particle_count)
            #print(self.particles[a: b].shape, self.current_population[idx].shape)
            #print(idx, self.current_population[idx])
            self.particles[a: b] = self.current_population[idx[: b-a]]
            self.weights[a: b] = self.pop_weights[idx[: b-a]]
            self.current_selection_size = b
        #self.weights = np.array(self.weights)[idx][::-1][:self.particle_count]
        #self.weights /= self.weights.sum()

    def one_step_update(self, observation, threshold_factor=0.5):
        """
        Description:
            Updates weights according to the last observation
        Args:
            observation: an observation of dimension = self.dimension
        Returns:
            self.weights
        """
        # predict the new particles
        if self.current_time == 0:
            self.current_population = self.model.hidden_state.sims[0].generate(self.max_population) 
            if len(self.particles) != self.particle_count:
                self.particles = self.model.hidden_state.sims[0].generate(self.particle_count)
        elif self.current_time > 0:
            self.current_population[self.particle_count :] = np.array([self.model.hidden_state.sims[self.current_time].algorithm(self.current_time, particle) for particle in self.current_population[self.particle_count :]])
            self.current_population[: self.particle_count] = np.array([self.model.hidden_state.sims[self.current_time].algorithm(self.current_time, particle) for particle in self.particles])

        gen = 0
        self.current_selection_size = 0
        self.pop_weights = np.array([self.model.observation.conditional_pdf(self.current_time, observation, particle)\
                        for particle in self.current_population])
        while self.current_selection_size < self.particle_count and gen < self.max_generations_per_step: 
            self.breed()
            self.select(observation, threshold_factor)
            gen += 1

        # normalize weights
        self.weights /= self.weights.sum()
        print('step: {}, sum of weights: {}'.format(self.current_time, self.weights.sum()), end='\r')
        if np.isnan(self.weights[0]) or np.isinf(self.weights[0]):
            self.status = 'faliure'
        self.last_gen = gen + 1
     
    def record(self, observation):
        """
        Description:
            Records assimilation steps
        """
        if self.recording:
            #hdf5 = tables.open_file(self.record_path, 'a')
            # record weights
            weights = self.hdf5.create_table(self.hdf5.root.weights, 'time_' + str(self.current_time), self.weight_description)
            weights.append(self.weights)
            weights.flush()
            # record particles
            particles = self.hdf5.create_table(self.hdf5.root.particles, 'time_' + str(self.current_time), self.particle_description)
            particles.append(self.particles)
            particles.flush()
            # record observation
            self.hdf5.root.observation.append(np.array(observation, dtype = np.float64))
            self.hdf5.root.observation.flush()
            # record generation
            self.hdf5.root.generation.append(np.array([self.last_gen], dtype = np.int32))
            self.hdf5.root.generation.flush()

    def update(self, observations, method = 'mean', threshold_factor=0.9):
        """
        Description:
            Updates using all the obeservations using self.one_step_update and self.resample
        Args:
            observations: list/np.array of observations to pass to self.one_step_update
            method: method for computing trajectory, default = 'mean'
        Returns:
            self.weights
        """
        self.observed_path = observations
        for observation in self.observed_path:
            self.one_step_update(observation = observation, threshold_factor=threshold_factor)
            if method is not None:
                self.compute_trajectory(method = method)
            self.record(observation)
            if self.status == 'failure':
                break
            self.current_time += 1
        if self.status != 'failure':
            self.status = 'success'
        self.hdf5.close()
        return self.status
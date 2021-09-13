import filter as fl 
import  numpy as np
import copy
import tables 
import plot 

class SwarmPF(fl.ParticleFilter):

    def __init__(self, model, particle_count, folder = None, particles = None, max_iterations_per_step=3,\
                 inertia=0.2, cognitive_coeff=2000, social_coeff=0.01):
        super().__init__(model=model, particle_count=particle_count, folder=folder, particles=particles)
        self.max_iterations_per_step = max_iterations_per_step
        self.cognitive_coeff = cognitive_coeff
        self.social_coeff = social_coeff
        self.inertia = inertia
        self.cov = 1.0 * np.identity(self.dimension)
        tbl = self.hdf5.create_table(self.hdf5.root, 'generation', {'count': tables.Int32Col(pos = 0)})
        tbl.flush()
        self.find_max_weight()
        self.velocities = np.random.normal(size=self.particles.shape)

    def find_max_weight(self):
        self.max_log_weight = - 0.5 * np.log(2.0 * np.pi * self.model.observation.sigma[0][0]) * self.model.observation.dimension
        self.max_weight = np.exp(self.max_log_weight)

    def find_best(self):
        """
        for i, particle in enumerate(self.particles):
            if self.weights[i] > self.best_weights[i]:
                self.personal_best[i] = particle
                self.best_weights[i] = self.weights[i]
        """
        idx = np.argmax(self.best_weights)
        if self.best_weight < self.best_weights[idx]:
            self.best_weight = self.best_weights[idx]
            self.best_particle = self.personal_best[idx]

        #self.particles = copy.deepcopy(self.personal_best)
        #self.weights = copy.deepcopy(self.best_weights)
    

    def move(self, observation):
        r_1 = np.random.uniform(low=0., high=2., size=(self.particle_count, 1))
        r_1 = np.hstack([r_1 for _ in range(self.model.hidden_state.dimension)])
        r_2 = np.random.uniform(low=0., high=2., size=(self.particle_count, 1))
        r_2 = np.hstack([r_2 for _ in range(self.model.hidden_state.dimension)])

        self.velocities = self.inertia * self.velocities \
                        + r_2 * self.social_coeff * (self.best_particle - self.particles)\
                        #+ r_1 * self.cognitive_coeff * (self.personal_best - self.particles)\
                            
        new_particles = np.concatenate([self.particles, self.particles + self.velocities], axis=0)
        self.best_weights = np.array([self.model.observation.conditional_pdf(self.current_time, observation, member) for member in new_particles])
        idx = np.argsort(self.best_weights)[::-1]
        self.best_weights = self.best_weights[idx]
        new_particles = new_particles[idx]
        self.particles = new_particles[self.particle_count:]
        self.personal_best = new_particles[:self.particle_count]
        self.weights = self.best_weights[:self.particle_count]
 
    
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
            self.current_population = self.model.hidden_state.sims[0].generate(self.particle_count) 
            if len(self.particles) != self.particle_count:
                self.particles = self.model.hidden_state.sims[0].generate(self.particle_count)
        elif self.current_time > 0:
            self.current_population = np.array([self.model.hidden_state.sims[self.current_time].algorithm(self.current_time, particle) for particle in self.particles])
            

        gen = 0
        self.velocities = np.random.normal(size=self.particles.shape)
        self.best_weights = copy.deepcopy(self.weights)
        self.personal_best = copy.deepcopy(self.particles)
        idx = np.argmax(self.best_weights)
        self.best_weight = self.best_weights[idx]
        self.best_particle = self.particles[idx]
        while gen < self.max_iterations_per_step: 
            self.move(observation)
            self.find_best()
            gen += 1
            print('gen = ' + str(gen), end='\r')
     
        self.particles = copy.deepcopy(self.personal_best)
        #self.weights = new_weights[:self.particle_count]
        # normalize weights
        print('step: {}, sum of weights: {}'.format(self.current_time, self.weights.sum()), end='\n')
        self.weights /= self.weights.sum()
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

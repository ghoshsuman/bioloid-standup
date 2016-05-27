import logging
import os
from threading import Barrier, BrokenBarrierError

import pickle

from Simulation import Simulation
from StateMapper import StateMapper
from pybrain.rl.agents import LearningAgent
from pybrain.rl.explorers import EpsilonGreedyExplorer
from pybrain.rl.learners import ActionValueTable, Q
from utils import Utils


class SimulationMaster:

    def __init__(self, n_threads=4, initial_port=19997, q_table_version=0, batch_size = None):
        self.barrier = Barrier(n_threads + 1, timeout=720)
        self.n_threads = n_threads
        self.initial_port = initial_port
        self.q_table_version = q_table_version
        self.batch_size = batch_size
        state_mapper = StateMapper()
        self.controller = ActionValueTable(state_mapper.get_state_space_size(), Utils.N_ACTIONS)
        self.learner = Q(0.5, 0.9)
        self.agent = LearningAgent(self.controller, self.learner)
        self.simulations = None
        self.explorer = self.learner.explorer = EpsilonGreedyExplorer(0.2, 0.998)
        self.logger = logging.getLogger('master_logger')
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(logging.FileHandler('data/learning-tables/master.log'))
        self.failed_simulations = []
        self.n_episodes = 0
        self.initialize_simulations()

    def initialize_simulations(self):
        self.simulations = []
        for i in range(self.n_threads):
            if self.batch_size is not None:
                self.simulations.append(Simulation(self, self.initial_port + i, self.batch_size))
            else:
                self.simulations.append(Simulation(self, self.initial_port + i))

    def initialize_q_table(self):
        self.controller.initialize(10.)
        self.load_q_table()

        for i in range(5):
            self.learn_trajectory()

    def learn_trajectory(self):
        with open('data/trajectory.pkl', 'rb') as file:
            trajectory_data = pickle.load(file)

        for trajectory in trajectory_data:
                for t in trajectory:
                    if t['action'] == -1:
                        continue
                    self.add_observation([t['state'], Utils.vecToInt(t['action']), t['reward']])
                self.agent.learn()
                self.agent.reset()

    def get_action(self, observation):
        action = self.controller.activate(observation)
        action = self.explorer.activate(observation, action)
        return action

    def add_observation(self, obs):
        """
            Adds observation in the agent memory
            :param obs: 3 dimensional vector containing [observation, action, reward]
        """
        self.agent.integrateObservation(obs[0])
        self.agent.lastaction = obs[1]
        self.agent.giveReward(obs[2])

    def update_q_table(self):
        """
            Updates the q table with the new simulators observations
        """
        for sim in self.simulations:
            for trace in sim.traces:
                for obs in trace:
                    self.add_observation(obs)
                self.agent.learn()
                self.agent.reset()
                self.n_episodes +=1

            sim.traces.clear()
        if self.explorer.epsilon > 0.1:
            self.explorer.decrement_epsilon()
        self.logger.info('new epsilon: {}'.format(self.explorer.epsilon))
        self.logger.info('n episodes: {}'.format(self.n_episodes))

    def load_q_table(self):
        """
            Load the q table from file if it exists
        :param q_table_version:
        """
        file_path = 'data/learning-tables/q-table-{}.pkl'.format(self.q_table_version)
        if os.path.exists(file_path):
            with open(file_path, 'rb') as file:
                self.controller._params = pickle.load(file)
            self.q_table_version += 1

    def save_q_table(self):
        """
            Saves the q table on file in the data/learning-tables folder
        """
        with open('data/learning-tables/q-table-{}.pkl'.format(self.q_table_version), 'wb') as file:
            pickle.dump(self.controller.params, file)
        self.q_table_version += 1

    def save_t_table(self):
        """
            Saves t tables, one for each thread
        """
        for sim in self.simulations:
            sim.save_t_table()

    def run(self):

        self.initialize_q_table()
        for sim in self.simulations:
            sim.start()

        while True:
            try:
                self.barrier.wait()  # wait until all simulations are done
                self.update_q_table()
                self.save_t_table()
                self.barrier.wait()  # Free simulations threads and start a new cycle
                self.save_q_table()
                while self.failed_simulations:
                    sim = self.failed_simulations.pop()
                    self.restart_simulation(sim)
            except BrokenBarrierError as e:
                self.logger.error('Broken Barrier Error Occurred')
                for sim in self.simulations:
                    sim.stop()
                for sim in self.simulations:
                    sim.join()
                del self.simulations
                self.initialize_simulations()
                self.barrier.reset()
                self.failed_simulations.clear()
                for sim in self.simulations:
                    sim.start()

    def restart_simulation(self, simulation):
        self.logger.info('Restarting simulation with port {}'.format(simulation.port))
        self.simulations.remove(simulation)
        new_simulation = Simulation(self, simulation.port)
        self.simulations.append(new_simulation)
        new_simulation.start()
        del simulation

import logging
from threading import Barrier

import pickle

import time

import numpy

from Simulation import Simulation
from StateMapper import StateMapper
from pybrain.rl.agents import LearningAgent
from pybrain.rl.explorers import EpsilonGreedyExplorer
from pybrain.rl.learners import ActionValueTable, Q
from utils import Utils


class SimulationMaster:

    def __init__(self, n_threads=4, initial_port=19997, q_table_version=0):
        self.barrier = Barrier(n_threads + 1)
        self.q_table_version = q_table_version
        state_mapper = StateMapper()
        self.controller = ActionValueTable(state_mapper.get_state_space_size(), Utils.N_ACTIONS)
        self.learner = Q(0.5, 0.9)
        self.agent = LearningAgent(self.controller, self.learner)
        self.simulations = []
        self.explorer = self.learner.explorer = EpsilonGreedyExplorer(0.2, 0.998)
        self.logger = logging.getLogger('master_logger')
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(logging.FileHandler('data/learning-tables/master.log'))
        self.failed_simulations = []
        self.n_episodes = 0

        for i in range(n_threads):
            self.simulations.append(Simulation(self, initial_port + i))

    def initialize_q_table(self):
        self.controller.initialize(10.)

        with open('data/trajectory.pkl', 'rb') as file:
            trajectory_data = pickle.load(file)

        for i in range(5):
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
        self.explorer.decrement_epsilon()
        self.logger.info('new epsilon: {}'.format(self.explorer.epsilon))
        self.logger.info('n episodes: {}'.format(self.n_episodes))

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
            self.barrier.wait()  # wait until all simulations are done
            self.update_q_table()
            self.save_t_table()
            self.barrier.wait()  # Free simulations threads and start a new cycle
            while self.failed_simulations:
                sim = self.failed_simulations.pop()
                self.restart_simulation(sim)
            self.save_q_table()

    def restart_simulation(self, simulation):
        self.logger.info('Restarting simulation with port {}'.format(simulation.port))
        self.simulations.remove(simulation)
        new_simulation = Simulation(self, simulation.port)
        self.simulations.append(new_simulation)
        new_simulation.start()
        del simulation

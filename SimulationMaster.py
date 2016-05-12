from threading import Barrier

import pickle

import time

from Simulation import Simulation
from pybrain.rl.agents import LearningAgent
from pybrain.rl.explorers import EpsilonGreedyExplorer
from pybrain.rl.learners import ActionValueTable, Q
from utils import Utils


class SimulationMaster:

    def __init__(self, n_threads=4, initial_port = 19997, q_table_version=0):
        self.barrier = Barrier(n_threads + 1)
        self.q_table_version = q_table_version
        self.controller = ActionValueTable(Utils.getNStates(), Utils.getNActions())
        self.learner = Q()
        self.agent = LearningAgent(self.controller, self.learner)
        self.simulators = []
        self.explorer = self.learner.explorer = EpsilonGreedyExplorer(0.2)

        for i in range(n_threads):
            self.simulators.append(Simulation(self, initial_port + i))

    def initialize_q_table(self):
        self.controller.initialize(10.)

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
        for sim in self.simulators:
            for trace in sim.traces:
                for obs in trace:
                    self.add_observation(obs)
                self.agent.learn()
                self.agent.reset()
            sim.traces.clear()

    def save_q_table(self, qtable_version = 0):
        """
            Saves the q table on file in the data/learning-tables folder
        """
        with open('data/learning-tables/q-table-{}.pkl'.format(qtable_version), 'wb') as file:
            pickle.dump(self.controller.params, file)

    def run(self):

        self.initialize_q_table()
        for sim in self.simulators:
            sim.start()

        while True:
            self.barrier.wait()  # wait until all simulations are done
            self.update_q_table()
            self.barrier.wait()  # Free simulations threads and start a new cycle
            self.save_q_table()


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

    def inizializeQTable(self):
        self.controller.initialize(10.)

        with open('data/trajectory.pkl', 'rb') as file:
            trajectory_data = pickle.load(file)

        for trajectory in trajectory_data:
            for t in trajectory:
                if t['action'] == -1:
                    continue
                self.addObservation([t['state'], Utils.vecToInt(t['action']), t['reward']])
            self.agent.learn()
            self.agent.reset()

    def getAction(self, observation):
        action = self.controller.activate(observation)
        action = self.explorer.activate(observation, action)
        return action

    def addObservation(self, obs):
        """
            Adds observation in the agent memory
        :param obs: 3 dimensional vector containing [observation, action, reward]
        """
        self.agent.integrateObservation(obs[0])
        self.agent.lastaction = obs[1]
        self.agent.giveReward(obs[2])

    def updateQTable(self):
        """
            Updates the q table with the new simulators observations
        """
        for sim in self.simulators:
            for obs in sim.trace:
                self.addObservation(obs)
            self.agent.learn()
            self.agent.reset()
            sim.trace.clear()

    def run(self):

        self.inizializeQTable()
        for sim in self.simulators:
            sim.start()
            time.sleep(5)

        while True:
            self.barrier.wait()  # wait until all simulations are done
            self.updateQTable()
            self.barrier.wait()  # Free simulations threads and start a new cycle


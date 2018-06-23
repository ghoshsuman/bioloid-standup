import logging
from threading import Barrier, BrokenBarrierError

from runners.Simulation import Simulation
from pybrain.rl.agents import LearningAgent
from pybrain.rl.explorers import EpsilonGreedyExplorer
from pybrain.rl.learners import Q
from models import MyActionValueTable
from utils import Utils


class SimulationMaster:

    def __init__(self, n_threads=4, initial_port=19997, q_table_version=0,
                 batch_size=None, learner=None, explorer=None):
        self.barrier = Barrier(n_threads + 1, timeout=720)
        self.n_threads = n_threads
        self.initial_port = initial_port
        self.batch_size = batch_size

        self.controller = MyActionValueTable(q_table_version)
        if learner is None:
            self.learner = Q(0.5, 0.9)
        else:
            self.learner = learner

        if explorer is None:
            self.explorer = self.learner.explorer = EpsilonGreedyExplorer(0.2, 0.998)
        else:
            self.explorer = self.learner.explorer = explorer
        self.agent = LearningAgent(self.controller, self.learner)
        # Logger initialization
        self.logger = logging.getLogger('master_logger')
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(logging.FileHandler(Utils.DATA_PATH + 'learning-tables/master.log'))
        self.failed_simulations = []
        self.n_episodes = 0
        self.simulations = []
        self.initialize_simulations()

    def initialize_simulations(self):
        self.simulations = []
        for i in range(self.n_threads):
            if self.batch_size is not None:
                self.simulations.append(Simulation(self, self.initial_port + i, self.batch_size))
            else:
                self.simulations.append(Simulation(self, self.initial_port + i))

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
                self.n_episodes += 1

            sim.traces.clear()
        if self.explorer.epsilon > 0.1:
            self.explorer.apply_decay()
        if self.learner.alpha > 0.1:
            self.learner.alpha *= 0.999
        self.logger.info('new epsilon: {}'.format(self.explorer.epsilon))
        self.logger.info('new alpha: {}'.format(self.learner.alpha))
        self.logger.info('n episodes: {}'.format(self.n_episodes))

    def save_t_table(self):
        """
            Saves t tables, one for each thread
        """
        for sim in self.simulations:
            sim.save_t_table()

    def run(self):

        self.controller.initialize(self.agent)
        for sim in self.simulations:
            sim.start()
        counter = 0
        while True:
            try:
                self.barrier.wait()  # wait until all simulations are done
                self.update_q_table()
                self.save_t_table()
                self.barrier.wait()  # Free simulations threads and start a new cycle
                # Counter to avoid to save q-table too often
                if counter == 5:
                    self.controller.save()
                    counter = 0
                else:
                    counter += 1
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

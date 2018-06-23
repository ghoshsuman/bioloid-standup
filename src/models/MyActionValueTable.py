import os
import pickle

from pybrain.rl.learners import ActionValueTable

from utils import Utils


class MyActionValueTable(ActionValueTable):

    def __init__(self, q_table_version=1):
        super().__init__(Utils.state_mapper().get_state_space_size(), Utils.N_ACTIONS)
        self.q_table_version = q_table_version

    def initialize(self, agent):
        """
        Initialize the q-table with default values
        :param agent: the agent using the table
        :return:
        """
        super().initialize(10.)
        self.load()

        alpha = agent.learner.alpha
        agent.learner.alpha = 1
        for i in range(5):
            self.learn_trajectory(agent)
        agent.learner.alpha = alpha

    def learn_trajectory(self, agent):
        with open(Utils.DATA_PATH + 'trajectory.pkl', 'rb') as file:
            trajectory_data = pickle.load(file)

        for trajectory in trajectory_data:
            for t in trajectory:
                if t['action'] == -1:
                    continue
                agent.integrateObservation(t['state'])
                agent.lastaction = t['action']
                agent.giveReward(t['reward'])
            agent.learn()
            agent.reset()

    def load(self):
        """
            Load the q table from file if it exists
            :param q_table_version:
        """
        file_path = Utils.DATA_PATH + 'learning-tables/q-table-{}.pkl'.format(self.q_table_version)
        if os.path.exists(file_path):
            with open(file_path, 'rb') as file:
                self._params = pickle.load(file)
            self.q_table_version += 1

    def save(self):
        """
            Saves the q table on file in the data/learning-tables folder
        """
        with open(Utils.DATA_PATH + 'learning-tables/q-table-{}.pkl'.format(self.q_table_version), 'wb') as file:
            pickle.dump(self._params, file)
        self.q_table_version += 1

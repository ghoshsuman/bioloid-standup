import logging
import numpy
from pybrain.rl.environments import EpisodicTask

from algs.StateMapper import StateMapper
from models import NDSparseMatrix
from utils import Utils


class StandingUpTask(EpisodicTask):

    GOAL_REWARD = 1000
    ENERGY_CONSUMPTION_REWARD = -1
    FALLEN_REWARD = -100
    SELF_COLLISION_REWARD = -100
    TOO_FAR_REWARD = -100

    def __init__(self, environment, log_file_path= Utils.DATA_PATH + 'learning.log', multiple_init_state=False):
        super(StandingUpTask, self).__init__(environment)
        self.finished = False
        self.t_table = NDSparseMatrix()
        self.t_table.load()
        self.state_mapper = StateMapper(self.env.bioloid)
        self.current_sensors = self.current_state = None
        self.update_current_state()
        self.logger = logging.getLogger(log_file_path)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(logging.FileHandler(log_file_path))
        self.init_state_prob_distribution = []
        self.multiple_init_state = multiple_init_state
        if multiple_init_state:
            n = len(Utils.standingUpActions)
            den = n * (n + 1) / 2
            for i in range(n):
                self.init_state_prob_distribution.append((n - i) / den)

    def getReward(self):
        reward = self.ENERGY_CONSUMPTION_REWARD
        if self.current_state == self.state_mapper.fallen_state:
            self.logger.info('Fallen!')
            reward = self.FALLEN_REWARD
            self.finish()
        elif self.current_state == self.state_mapper.self_collided_state:
            self.logger.info('Collision!')
            reward = self.SELF_COLLISION_REWARD
            self.finish()
        elif self.current_state == self.state_mapper.too_far_state:
            self.logger.info('Too Far!')
            reward = self.TOO_FAR_REWARD
            self.finish()
        elif self.current_state == self.state_mapper.goal_state:
            self.logger.info('Goal!')
            reward = self.GOAL_REWARD
            self.finish()
        return reward

    def performAction(self, action):
        if isinstance(action, numpy.ndarray):
            action = action[0]
        print('Action: '+str(action))
        self.env.performAction(Utils.intToVec(action))
        self.update_current_state(action)

    def update_current_state(self, action=None):
        previous_state = self.current_state
        sensors = self.env.getSensors()
        self.current_sensors = sensors
        self.current_state = self.state_mapper.map(sensors)

        # Store in the transition table the current transition
        if action is not None:
            self.t_table.incrementValue((previous_state, action, self.current_state))
        return self.current_state

    def getObservation(self):
        return [self.current_state]

    def finish(self):
        self.finished = True

    def reset(self):
        self.finished = False
        self.env.reset()
        if self.multiple_init_state:
            step = numpy.random.choice(len(self.init_state_prob_distribution), p=self.init_state_prob_distribution)
            for i in range(step):
                self.env.performAction(Utils.standingUpActions[i])
            self.update_current_state()
            self.logger.info('Init state {} step {}'.format(self.getObservation()[0], step))

    def isFinished(self):
        return self.finished

    def get_state_space_size(self):
        return self.state_mapper.get_state_space_size()

    def get_action_space_size(self):
        return Utils.N_ACTIONS
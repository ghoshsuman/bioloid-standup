import logging
import os
import pickle

import numpy
from scipy.spatial import KDTree
from scipy.spatial.distance import euclidean

import vrep
from NDSparseMatrix import NDSparseMatrix
from StateMapper import StateMapper
from StateNormalizer import StateNormalizer
from bioloid import Bioloid
from pybrain.rl.environments import Environment, Task, EpisodicTask
from utils import Utils

class StandingUpEnvironment(Environment):
    opmode = vrep.simx_opmode_blocking
    STATIONARY_THRESHOLD = 0.01
    MAX_ITERATIONS_PER_ACTION = 20

    def __init__(self, client_id, model_path='data/models/bioloid.ttt'):
        super(StandingUpEnvironment, self).__init__()
        self.model_path = os.path.abspath(model_path)
        self.discreteActions = True
        self.discreteStates = True
        self.client_id = client_id
        self.bioloid = None
        self.reset()
        self.state_normalizer = StateNormalizer()
        self.state_normalizer.extend_bounds()

    def reset(self):
        super(StandingUpEnvironment, self).reset()
        # print('*** Reset ***')
        vrep.simxStopSimulation(self.client_id, self.opmode)
        vrep.simxCloseScene(self.client_id, self.opmode)
        vrep.simxLoadScene(self.client_id, self.model_path, 0, self.opmode)
        vrep.simxSynchronous(self.client_id, True)
        vrep.simxStartSimulation(self.client_id, self.opmode)
        self.bioloid = Bioloid(self.client_id)

    def getSensors(self):
        state_vector = self.bioloid.read_state()
        # print('not norm: '+ str(state_vector))
        # state_vector = self.state_normalizer.normalize(state_vector)
        # print('norm: '+ str(state_vector))
        return state_vector

    def performAction(self, action):
        self.bioloid.move_arms(action[0:3])
        self.bioloid.move_legs(action[3:])
        old_state = self.bioloid.read_state()
        dist = 1
        count = 0
        while dist > self.STATIONARY_THRESHOLD and count < self.MAX_ITERATIONS_PER_ACTION:
            vrep.simxSynchronousTrigger(self.client_id)
            new_state = self.bioloid.read_state()
            # dist = euclidean(old_state, new_state)
            dist = numpy.max(numpy.absolute(old_state - new_state))
            # print('dist: ' + str(dist))
            old_state = new_state
            count += 1
        print('Count: ' + str(count))


class StandingUpTask(EpisodicTask):

    GOAL_REWARD = 100
    ENERGY_CONSUMPTION_REWARD = -0.5
    FALLEN_REWARD = -10
    SELF_COLLISION_REWARD = -10
    TOO_FAR_REWARD = -100

    def __init__(self, environment, log_file_path='data/learning.log'):
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
        print('Reward: '+str(reward))
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
        self.update_current_state()

    def isFinished(self):
        return self.finished

    def get_state_space_size(self):
        return self.state_mapper.get_state_space_size()

    def get_action_space_size(self):
        return 729
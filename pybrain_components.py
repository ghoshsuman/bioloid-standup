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
        # print('Count: ' + str(count))


class StandingUpTask(EpisodicTask):

    GOAL_REWARD = 1000
    ENERGY_CONSUMPTION_REWARD = -1
    FALLEN_REWARD = -100
    SELF_COLLISION_REWARD = -100
    TOO_FAR_REWARD = -100

    def __init__(self, environment, log_file_path='data/learning.log', multiple_init_state=False):
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
        # self.init_states = []
        self.multiple_init_state = multiple_init_state
        if multiple_init_state:
            n = len(Utils.standingUpActions)
            den = n * (n + 1) / 2
            for i in range(n):
                self.init_state_prob_distribution.append((n - i) / den)
            #     self.init_states.append([])
            # with open('data/trajectory.pkl', 'rb') as handle:
            #     trajectories = pickle.load(handle)
            #     for trajectory in trajectories:
            #         for i, t in enumerate(trajectory):
            #             if t['action'] != -1:
            #                 self.init_states[i].append(t['full_state'])



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
        # print('Reward: '+str(reward))
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
            # index = numpy.random.choice(len(self.init_states[step]))
            # state = self.init_states[step][index]
            # self.env.bioloid.set_full_state(state)
            # self.env.performAction([0, 0, 0, 0, 0, 0])
            # self.logger.info('Init at step {} {} {}'.format(step, len(self.init_states[step]), index))
            # self.update_current_state()
            # with open('data/trajectory.pkl', 'rb') as handle:
            #     trajectories = pickle.load(handle)
            #     self.logger.info('comp states {} {} '.format(self.current_state, trajectories[index][step]['state']))
            #     if self.current_state != trajectories[index][step]['state']:
            #         print(self.env.bioloid.read_state())
            #         print(trajectories[index][step]['state_vector'])
        self.update_current_state()
        self.logger.info('Init state {} step {}'.format(self.getObservation()[0], step))



    def isFinished(self):
        return self.finished

    def get_state_space_size(self):
        return self.state_mapper.get_state_space_size()

    def get_action_space_size(self):
        return Utils.N_ACTIONS
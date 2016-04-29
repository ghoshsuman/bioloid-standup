import logging
import os
import pickle

import numpy
from scipy.spatial import KDTree
from scipy.spatial.distance import euclidean

import vrep
from NDSparseMatrix import NDSparseMatrix
from StateNormalizer import StateNormalizer
from bioloid import Bioloid
from pybrain.rl.environments import Environment, Task, EpisodicTask
from utils import Utils

class StandingUpSimulator(Environment):
    opmode = vrep.simx_opmode_blocking
    STATIONARY_THRESHOLD = 0.01
    MAX_ITERATIONS_PER_ACTION = 20

    def __init__(self, client_id, model_path='data/models/bioloid.ttt'):
        super(StandingUpSimulator, self).__init__()
        self.model_path = os.path.abspath(model_path)
        self.discreteActions = True
        self.discreteStates = True
        self.client_id = client_id
        self.bioloid = None
        self.reset()
        self.state_normalizer = StateNormalizer()
        self.state_normalizer.extend_bounds()

    def reset(self):
        super(StandingUpSimulator, self).reset()
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
        state_vector = self.state_normalizer.normalize(state_vector)
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
    GOAL_STATE = 5852
    TOO_FAR_THRESHOLD = 0.4  # Mean/2 distance of points
    GOAL_THRESHOLD = 0.1

    def __init__(self, environment):
        super(StandingUpTask, self).__init__(environment)
        self.kdtree = None
        self.finished = False
        self.t_table = NDSparseMatrix()
        self.t_table.load()
        self.load_state_space()
        self.current_sensors = self.kdtree.data[0]
        n = len(self.kdtree.data)
        self.fallen_state = n
        self.self_collided_state = n + 1
        self.too_far_state = n + 2
        self.goal_state = n + 3
        self.end_state = n + 4
        self.current_state = 0
        self.update_current_state()
        self.logger = logging.getLogger('learning_logger')
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(logging.FileHandler('data/learning.log'))

    def load_state_space(self, filepath='data/state-space/state-space-all-0.pkl'):
        with open(filepath, 'rb') as handle:
            data = pickle.load(handle)
        for i in range(len(data)):
            data[i] = self.env.state_normalizer.normalize(data[i])
        self.kdtree = KDTree(data)

    def get_goal_state_vector(self):
        return self.kdtree.data[self.GOAL_STATE]

    def getReward(self):
        goal = self.get_goal_state_vector()
        if self.current_sensors is not None:
            goal_distance = euclidean(goal, self.current_sensors)
            print('Goal distance: ' + str(goal_distance))
        else:
            goal_distance = numpy.inf
        reward = self.ENERGY_CONSUMPTION_REWARD
        if self.current_state == self.fallen_state:
            self.logger.info('Fallen!')
            reward = self.FALLEN_REWARD
            self.finish()
        elif self.current_state == self.self_collided_state:
            self.logger.info('Collision!')
            reward += self.SELF_COLLISION_REWARD
            self.finish()
        elif self.current_state == self.too_far_state:
            self.logger.info('Too Far!')
            reward = self.TOO_FAR_REWARD
            self.finish()
        elif goal_distance < self.GOAL_THRESHOLD:
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
        dist = None
        try:
            sensors = self.env.getSensors()
            dist, index = self.kdtree.query(sensors)
            self.current_sensors = sensors
            self.current_state = index
            if dist > self.TOO_FAR_THRESHOLD:  # Check if the actual state is too far from the one it was mapped
                self.current_state = self.too_far_state
        except AssertionError:
            self.current_state = self.too_far_state
            self.current_sensors = None
            self.logger.debug('Normalization bounds excedeed: ' + str(self.env.bioloid.read_state()))

        if self.env.bioloid.isFallen():  # Check if the bioloid is fallen
            self.current_state = self.fallen_state
        elif self.env.bioloid.isSelfCollided():  # Check if the bioloid is self-collided
            self.current_state = self.self_collided_state

        # Store in the transition table the current transition
        if action is not None:
            self.t_table.incrementValue((previous_state, action, self.current_state))
        print('{} {}'.format(self.current_state, dist))
        return self.current_state

    def getObservation(self):
        return [self.current_state]

    def finish(self):
        self.finished = True
        self.env.reset()

    def reset(self):
        self.current_state = 0
        self.finished = False

    def isFinished(self):
        return self.finished

    def getStateNumber(self):
        return len(self.kdtree.data) + 5

    def getActionNumber(self):
        return 729


class RaiseArmSimulator(Environment):
    SCENE_PATH = '/home/simone/Dropbox/Vuotto Thesis/bioloid.ttt'
    opmode = vrep.simx_opmode_blocking

    def __init__(self, client_id):
        super(RaiseArmSimulator, self).__init__()
        self.discreteActions = True
        self.discreteStates = True
        self.client_id = client_id
        self.bioloid = None
        self.hand_handle = -1
        self.reset()

    def reset(self):
        super(RaiseArmSimulator, self).reset()
        print('*** Reset ***')
        vrep.simxStopSimulation(self.client_id, self.opmode)
        vrep.simxCloseScene(self.client_id, self.opmode)
        vrep.simxLoadScene(self.client_id, RaiseArmSimulator.SCENE_PATH, 0, self.opmode)
        vrep.simxSynchronous(self.client_id, True)
        vrep.simxStartSimulation(self.client_id, self.opmode)
        return_code, self.hand_handle = vrep.simxGetObjectHandle(self.client_id, 'left_hand_link_visual', self.opmode)
        self.bioloid = Bioloid(self.client_id)

    def getSensors(self):
        return_code, positions = vrep.simxGetObjectPosition(self.client_id, self.hand_handle, -1, self.opmode)
        return positions

    def performAction(self, action):
        self.bioloid.move_arms(action)
        for i in range(5):
            vrep.simxSynchronousTrigger(self.client_id)


class RaiseArmsTask(Task):

    arm_index = [0, 0, 0]

    def getReward(self):
        if self.arm_index[2] == 6:
            self.env.reset()
            return 10
        elif self.arm_index[2] == 5:
            return 1
        elif self.arm_index[2] == 0:
            self.env.reset()
            return -10
        else:
            return 0

    def performAction(self, action):
        action = int(action[0])
        a = []
        for i in range(3):
            v = action % 3
            action //= 3
            a.append(v-1)
        # a = [0, 0, 0]
        self.env.performAction(a)

    def getObservation(self):
        pos = self.env.getSensors()
        x = int(pos[0] * 100 + 13)
        y = int(pos[1] * 100 + 8)
        z = int(pos[2] * 100 - 13)

        self.arm_index[0] = max(0, x // 5)
        self.arm_index[1] = max(0, y // 5)
        self.arm_index[2] = min(6, max(0, z // 5))

        print(self.arm_index)

        return [self.arm_index[0] + self.arm_index[1] * 6 + self.arm_index[2] * 36]

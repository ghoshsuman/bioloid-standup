import os
import pickle

import numpy
from scipy.spatial import KDTree
from scipy.spatial.distance import euclidean

import vrep
from NDSparseMatrix import NDSparseMatrix
from StateNormalizer import StateNormalizer
from bioloid import Bioloid
from pybrain.rl.environments import Environment, Task
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


class StandingUpTask(Task):

    GOAL_REWARD = 100
    ENERGY_CONSUMPTION_REWARD = -0.5
    FALLEN_REWARD = -100
    SELF_COLLISION_REWARD = -10
    GOAL_DISTANCE_REWARD = 5
    GOAL_STATE = [0.99249412,  0.01152511,  0.9999996 ,  0.26637849,  0.7365726 ,
                  0.72537018,  0.26239846,  0.93522064,  0.61111627,  0.49702422,
                  0.79671855,  0.42516445,  0.49819804,  0.49492229,  0.94970996,
                  0.24974432,  0.49877833,  0.94491837,  0.25551248]
    TOO_FAR_THRESHOLD = 5  # 0.15  # 0.24 Mean distance of points

    def __init__(self, environment):
        super(StandingUpTask, self).__init__(environment)
        self.kdtree = None
        self.current_state = 0
        self.current_action = -1
        self.finished = False
        self.t_table = NDSparseMatrix()
        self.t_table.load()
        self.load_state_space()
        self.current_sensors = self.kdtree.data[0]
        self.too_far_state = self.getStateNumber() - 1

    def load_state_space(self, filepath = 'data/state-space/state-space-all-0.pkl'):
        with open(filepath, 'rb') as handle:
            data = pickle.load(handle)
        for i in range(len(data)):
            data[i] = self.env.state_normalizer.normalize(data[i])
        self.kdtree = KDTree(data)

    def calcDistance(self, v1, v2):
        d1 = euclidean(v1[0:3], v2[0:3])
        d2 = euclidean(v1[3:6], v2[3:6])
        d3 = euclidean(v1[6:], v2[6:])
        return d1 * 10 + d2 * 10 + d3

    def getReward(self):
        goal = self.env.norm.normalize(self.GOAL_STATE)
        # distance = self.calcDistance(goal, self.current_sensors)
        distance = euclidean(goal, self.current_sensors)
        print('Goal distance: ' + str(distance))
        reward = self.ENERGY_CONSUMPTION_REWARD
        if self.env.bioloid.isFallen():
            print('Fallen!')
            reward = self.FALLEN_REWARD
            self.finish()
        elif self.env.bioloid.isSelfCollided():
            print('Collision!')
            reward += self.SELF_COLLISION_REWARD
            self.finish()
        elif self.current_state == self.too_far_state:
            print('Too Far!')
            reward = self.TOO_FAR_REWARD
            self.finish()
        elif distance < 0.1:
            print('Goal!')
            reward += self.GOAL_REWARD
            self.finish()
        print('Reward: '+str(reward))
        return reward

    def performAction(self, action):
        print('Action: '+str(action))
        self.env.performAction(Utils.intToVec(action))

    def getObservation(self):
        sensors = self.env.getSensors()
        dist, index = self.kdtree.query(sensors)

        self.current_sensors = sensors
        previous_state = self.current_state
        self.current_state = index

        # Check if the actual state is too far from the one it was mapped
        if dist > self.TOO_FAR_THRESHOLD:
            self.current_state = self.too_far_state

        # Store in the transition table the current transition
        if self.current_action >= 0:
            self.t_table.incrementValue((previous_state, self.current_action, self.current_state))

        print(str(self.current_state) + '  ' + str(dist))

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
        return len(self.kdtree.data) + 1

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

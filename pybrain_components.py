import random

import pickle

from scipy.spatial import KDTree
from scipy.spatial.distance import euclidean

import vrep
from bioloid import Bioloid
from pybrain.rl.environments import Environment, Task
import scipy


class StandingUpSimulator(Environment):
    SCENE_PATH = '/home/simone/Dropbox/Vuotto Thesis/bioloid.ttt'
    opmode = vrep.simx_opmode_blocking
    STATIONARY_THRESHOLD = 0.01
    MAX_ITERATIONS_PER_ACTION = 20

    def __init__(self, client_id):
        super(StandingUpSimulator, self).__init__()
        self.discreteActions = True
        self.discreteStates = True
        self.client_id = client_id
        self.bioloid = None
        self.reset()

    def reset(self):
        super(StandingUpSimulator, self).reset()
        # print('*** Reset ***')
        vrep.simxStopSimulation(self.client_id, self.opmode)
        vrep.simxCloseScene(self.client_id, self.opmode)
        vrep.simxLoadScene(self.client_id, RaiseArmSimulator.SCENE_PATH, 0, self.opmode)
        vrep.simxSynchronous(self.client_id, True)
        vrep.simxStartSimulation(self.client_id, self.opmode)
        self.bioloid = Bioloid(self.client_id)

    def getSensors(self):
        return self.bioloid.read_state()

    def performAction(self, action):
        self.bioloid.move_arms(action[0:3])
        self.bioloid.move_legs(action[3:])
        old_state = self.bioloid.read_state()
        dist = 1
        count = 0
        while dist > self.STATIONARY_THRESHOLD and count < self.MAX_ITERATIONS_PER_ACTION:
            vrep.simxSynchronousTrigger(self.client_id)
            new_state = self.bioloid.read_state()
            dist = euclidean(old_state, new_state)
            old_state = new_state
            count += 1
        print('Count: ' + str(count))
        '''
        for i in range(5):
            vrep.simxSynchronousTrigger(self.client_id)
            new_state = self.bioloid.read_state()
            dist = euclidean(old_state, new_state)
            print('Distance: '+str(dist))
            old_state = new_state
        '''

    def vecToInt(self, action):
        res = 0
        for a in reversed(action):
            res = res * 3 + a + 1
        return res

    def intToVec(self, action, vecLength=6):
        a = []
        for i in range(vecLength):
            v = action % 3
            action //= 3
            a.append(v-1)
        return a


class StandingUpTask(Task):

    GOAL_REWARD = 100
    ENERGY_CONSUMPTION_REWARD = -0.01
    FALLEN_REWARD = -100
    SELF_COLLISION_REWARD = -10
    GOAL_DISTANCE_REWARD = 5
    GOAL_STATE = 8947

    def __init__(self, environment):
        super(StandingUpTask, self).__init__(environment)
        with open('state-space-filtered.pkl', 'rb') as handle:
            data = pickle.load(handle)
        self.kdtree = KDTree(data)
        self.current_state = 0
        self.finished = False

    def getReward(self):
        distance = euclidean(self.kdtree.data[self.GOAL_STATE], self.current_state)
        _, index = self.kdtree.query(self.current_state)
        reward = self.ENERGY_CONSUMPTION_REWARD + self.GOAL_DISTANCE_REWARD / distance
        if self.env.bioloid.isFallen():
            print('Fallen!')
            reward = self.FALLEN_REWARD
            self.finish()
        elif self.env.bioloid.isSelfCollided():
            print('Collision!')
            reward += self.SELF_COLLISION_REWARD
            self.finish()
        elif index == self.GOAL_STATE:
            print('Goal!')
            reward += self.GOAL_REWARD
            self.finish()

        return reward

    def performAction(self, action):
        self.env.performAction(self.env.intToVec(action))

    def getObservation(self):
        state = self.env.getSensors()
        dist, index = self.kdtree.query(state)
        self.current_state = state
        print(str(index) + '  ' + str(dist))
        return [index]

    def finish(self):
        self.finished = True
        self.env.reset()

    def reset(self):
        self.current_state = 0
        self.finished = False

    def isFinished(self):
        return self.finished

    def getStateNumber(self):
        return len(self.kdtree.data)

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

import os
import pickle
import subprocess

from scipy.spatial.distance import euclidean

import vrep
from StateMapper import StateMapper


class Utils:
    N_ACTIONS = 729
    NULL_ACTION = 364  # Action in which no movement is done -> [0, 0, 0, 0, 0, 0]
    NULL_ACTION_VEC = [0, 0, 0, 0, 0, 0]  # Action in which no movement is done

    standingUpActions = [[0, 1, 0, 1, -1, 0],
                         [0, 1, 0, 1, -1, 0],
                         [0, 1, 0, 1, -1, 0],
                         [-1, 0, 0, -1, 0, 1],
                         [0, 0, 0, -1, 0, 0],
                         [0, 0, 0, -1, 0, 0],
                         [0, 0, 0, -1, 0, 0],
                         [1, 0, 0, -1, 0, 1],
                         [0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 1, 0, 0],
                         [0, -1, 0, 1, 1, -1],
                         [0, -1, 0, -1, 1, 0],
                         [0, -1, 0, 0, 1, -1],
                         ]

    client_id = -1
    __state_mapper = None
    vrep_path = os.path.abspath('V-REP_PRO_EDU_V3_3_0_64_Linux/')


    @classmethod
    def state_mapper(cls):
        if cls.__state_mapper is None:
            cls.__state_mapper = StateMapper()
        return cls.__state_mapper

    @classmethod
    def exec_vrep(cls, port):
        proc = subprocess.Popen(
            'cd {} &&  xvfb-run --auto-servernum --server-num=1 ./vrep.sh -h -gREMOTEAPISERVERSERVICE_{}_FALSE_TRUE'
            .format(cls.vrep_path, port), shell=True)
        return proc

    @classmethod
    def connectToVREP(cls, port=19997):
        # vrep.simxFinish(-1)  # just in case, close all opened connections
        cls.client_id = vrep.simxStart('127.0.0.1', port, True, True, 5000, 5)  # Connect to V-REP
        assert cls.client_id >= 0, 'Failed connecting to remote API server'
        return cls.client_id

    @classmethod
    def endVREP(cls, client_id=None):
        if client_id is None:
            vrep.simxFinish(cls.client_id)
        else:
            vrep.simxFinish(client_id)

    @classmethod
    def vecToInt(cls, action):
        res = 0
        for a in reversed(action):
            res = res * 3 + a + 1
        return res

    @classmethod
    def intToVec(cls, action, vecLength=6):
        a = []
        for i in range(vecLength):
            v = action % 3
            action //= 3
            a.append(v - 1)
        return a

    @classmethod
    def distance(cls, v1, v2):
        d1 = euclidean(v1[0:3], v2[0:3])
        d2 = euclidean(v1[3:6], v2[3:6])
        d3 = euclidean(v1[6:], v2[6:])
        return d1 * 5 + d2 * 5 + d3

    @classmethod
    def getNActions(cls):
        return cls.N_ACTIONS

    @classmethod
    def getNStates(cls, filepath='data/state-space/state-space-all-0.pkl'):
        with open(filepath, 'rb') as handle:
            data = pickle.load(handle)
        return len(data) + 5

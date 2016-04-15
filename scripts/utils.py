import vrep


class Utils:

    N_ACTIONS = 729

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

    @classmethod
    def connectToVREP(cls):
        vrep.simxFinish(-1)  # just in case, close all opened connections
        cls.client_id = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)  # Connect to V-REP
        return cls.client_id

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
            a.append(v-1)
        return a


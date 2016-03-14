import vrep
from bioloid import Bioloid
from pybrain.rl.environments import Environment, Task


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

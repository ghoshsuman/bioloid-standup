from limb import *
import vrep


class Bioloid:

    def __init__(self, client_id):
        self.client_id = client_id
        self.leftArm = Arm(SIDE.LEFT, client_id)
        self.rightArm = Arm(SIDE.RIGHT, client_id)
        self.leftLeg = Leg(SIDE.LEFT, client_id)
        self.rightLeg = Leg(SIDE.RIGHT, client_id)

    def move_arms(self, values):
        self.leftArm.move_joints(self.client_id, values)
        self.rightArm.move_joints(self.client_id, values)

    def move_legs(self, values):
        self.leftLeg.move_joints(self.client_id, values)
        self.rightLeg.move_joints(self.client_id, values)

    def read_state(self):
        return_code, com_x = vrep.simxGetFloatSignal(self.client_id, 'COM_x', vrep.simx_opmode_blocking)
        return_code, com_y = vrep.simxGetFloatSignal(self.client_id, 'COM_y', vrep.simx_opmode_blocking)
        return_code, com_z = vrep.simxGetFloatSignal(self.client_id, 'COM_z', vrep.simx_opmode_blocking)
        print(str(com_x) + '   ' + str(com_y) + '   ' + str(com_z))

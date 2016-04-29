import numpy

from limb import *
import vrep


class Bioloid:

    opmode = vrep.simx_opmode_blocking

    def __init__(self, client_id):
        self.client_id = client_id
        self.leftArm = Arm(SIDE.LEFT, client_id)
        self.rightArm = Arm(SIDE.RIGHT, client_id)
        self.leftLeg = Leg(SIDE.LEFT, client_id)
        self.rightLeg = Leg(SIDE.RIGHT, client_id)
        res_code, body_handle = vrep.simxGetObjectHandle(client_id, 'torso_link_respondable', self.opmode)
        self.body_handle = body_handle

    def move_arms(self, values):
        self.leftArm.move_joints(self.client_id, values)
        self.rightArm.move_joints(self.client_id, values)

    def move_legs(self, values):
        self.leftLeg.move_joints(self.client_id, values)
        self.rightLeg.move_joints(self.client_id, values)

    def read_state(self):
        # Retriece Center of Mass Values
        return_code, com_x = vrep.simxGetFloatSignal(self.client_id, 'COM_x', self.opmode)
        return_code, com_y = vrep.simxGetFloatSignal(self.client_id, 'COM_y', self.opmode)
        return_code, com_z = vrep.simxGetFloatSignal(self.client_id, 'COM_z', self.opmode)
        # Retrieve quaternion values
        return_code, q1 = vrep.simxGetFloatSignal(self.client_id, 'q1', self.opmode)
        return_code, q2 = vrep.simxGetFloatSignal(self.client_id, 'q2', self.opmode)
        return_code, q3 = vrep.simxGetFloatSignal(self.client_id, 'q3', self.opmode)
        return_code, q4 = vrep.simxGetFloatSignal(self.client_id, 'q4', self.opmode)
        state_vector = [com_x, com_y, com_z, q1, q2, q3, q4]
        state_vector += self.leftArm.get_joints_position(self.client_id)
        state_vector += self.rightArm.get_joints_position(self.client_id)
        state_vector += self.leftLeg.get_joints_position(self.client_id)
        state_vector += self.rightLeg.get_joints_position(self.client_id)
        return numpy.array(state_vector)

    def isSelfCollided(self):
        return_code, selfcollision = vrep.simxGetIntegerSignal(self.client_id, 'is-self-collided', self.opmode)
        return selfcollision == 1

    def isFallen(self):
        return_code, fallen = vrep.simxGetIntegerSignal(self.client_id, 'is-fallen', self.opmode)
        return fallen == 1

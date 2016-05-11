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
        self.handles = []
        self._read_handles()

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
        if q1 < 0:
            q1 = -q1
            q2 = -q2
            q3 = -q3
            q4 = -q4
        state_vector = [com_x, com_y, com_z, q1, q2, q3, q4]
        state_vector += self.leftArm.get_joints_position(self.client_id)
        state_vector += self.rightArm.get_joints_position(self.client_id)
        state_vector += self.leftLeg.get_joints_position(self.client_id)
        state_vector += self.rightLeg.get_joints_position(self.client_id)
        return numpy.array(state_vector)

    def is_self_collided(self):
        return_code, self_collision = vrep.simxGetIntegerSignal(self.client_id, 'is-self-collided', self.opmode)
        return self_collision == 1

    def is_fallen(self):
        return_code, fallen = vrep.simxGetIntegerSignal(self.client_id, 'is-fallen', self.opmode)
        return fallen == 1

    def _read_handles(self):
        self.handles = []
        return_code, base_handler = vrep.simxGetObjectHandle(self.client_id, 'Bioloid', self.opmode)
        t = [base_handler]
        while len(t) > 0:
            h = t.pop()
            index = 0
            return_code, child_handler = vrep.simxGetObjectChild(self.client_id, h, index, self.opmode)
            while child_handler != -1:
                self.handles.append(child_handler)
                t.append(child_handler)
                index += 1
                return_code, child_handler = vrep.simxGetObjectChild(self.client_id, h, index, self.opmode)

    def read_full_state(self):
        objects_dict = []
        for h in self.handles:
            return_code, position = vrep.simxGetObjectPosition(self.client_id, h, -1, self.opmode)
            return_code, orientation = vrep.simxGetObjectOrientation(self.client_id, h, -1, self.opmode)
            objects_dict.append((position, orientation))
        joints_vector = []
        joints_vector += self.leftArm.get_joints_position(self.client_id)
        joints_vector += self.rightArm.get_joints_position(self.client_id)
        joints_vector += self.leftLeg.get_joints_position(self.client_id)
        joints_vector += self.rightLeg.get_joints_position(self.client_id)

        return {'objects': objects_dict, 'joints': joints_vector}

    def set_full_state(self, state):
        vrep.simxSetIntegerSignal(self.client_id, 'reset-dynamics', 1, vrep.simx_opmode_blocking)
        vrep.simxSynchronousTrigger(self.client_id)
        joints_vector = state['joints']
        self.leftArm.set_joints_position(self.client_id, joints_vector[0:3])
        self.rightArm.set_joints_position(self.client_id, joints_vector[3:6])
        self.leftLeg.set_joints_position(self.client_id, joints_vector[6:9])
        self.rightLeg.set_joints_position(self.client_id, joints_vector[9:])

        for h, s in zip(self.handles, state['objects']):
            vrep.simxSetObjectPosition(self.client_id, h, -1, s[0], self.opmode)
            vrep.simxSetObjectOrientation(self.client_id, h, -1, s[1], self.opmode)
import vrep
import math


def enum(**enums):
    return type('Enum', (), enums)

SIDE = enum(LEFT='left', RIGHT='right')


class Limb:
    opmode = vrep.simx_opmode_blocking

    def __init__(self, side):
        super().__init__()
        self.handlesList = []
        self.handlesDict = {}
        self.side = side

    def add_joint(self, client_id, name, handles_key):
        res_code, handle = vrep.simxGetObjectHandle(client_id, name, self.opmode)
        if res_code != 0:
            raise RuntimeError('Handle ' + name + ' not found!')
        self.handlesList.append(handle)
        self.handlesDict[handles_key] = handle

    def move_joints(self, client_id, values):
        assert len(values) == len(self.handlesList)
        for i in range(len(values)):
            handle = self.handlesList[i]
            return_code, position = vrep.simxGetJointPosition(client_id, handle, self.opmode)
            new_position = position + 30 * math.pi / 180 * values[i]
            vrep.simxSetJointTargetPosition(client_id, handle, new_position, self.opmode)

    def get_joint_index(self, name):
        return self.handlesList.index(self.handlesDict[name])

    def get_joints_position(self, client_id):
        positions = []
        for handle in self.handlesList:
            return_code, position = vrep.simxGetJointPosition(client_id, handle, self.opmode)
            positions.append(position)
        return positions


class Arm(Limb):

    SHOULDER_SWING_HANDLE = 'shoulder_swing'
    SHOULDER_LATERAL_HANDLE = 'shoulder_lateral'
    ELBOW_HANDLE = 'elbow'

    def __init__(self, side, client_id):
        super().__init__(side)
        self.add_joint(client_id, side + '_shoulder_swing_joint', self.SHOULDER_SWING_HANDLE)
        self.add_joint(client_id, side + '_shoulder_lateral_joint', self.SHOULDER_LATERAL_HANDLE)
        self.add_joint(client_id, side + '_elbow_joint', self.ELBOW_HANDLE)


class Leg(Limb):

    HIP_TWIST_HANDLE = 'hip_twist'
    HIP_LATERAL_HANDLE = 'hip_lateral'
    HIP_SWING_HANDLE = 'hip_swing'
    KNEE_HANDLE = 'knee'
    ANKLE_SWING_HANDLE = 'ankle_swing'
    ANKLE_LATERAL_HANDLE = 'ankle_lateral'

    def __init__(self, side, client_id):
        super().__init__(side)
        self.add_joint(client_id, side + '_hip_twist_joint', self.HIP_TWIST_HANDLE)
        self.add_joint(client_id, side + '_hip_lateral_joint', self.HIP_LATERAL_HANDLE)
        self.add_joint(client_id, side + '_hip_swing_joint', self.HIP_SWING_HANDLE)
        self.add_joint(client_id, side + '_knee_joint', self.KNEE_HANDLE)
        self.add_joint(client_id, side + '_ankle_swing_joint', self.ANKLE_SWING_HANDLE)
        self.add_joint(client_id, side + '_ankle_lateral_joint', self.ANKLE_LATERAL_HANDLE)

    def move_joints(self, client_id, values):
        # We use only a subset of availble joints in the leg
        used_joints = [self.HIP_SWING_HANDLE, self.KNEE_HANDLE, self.ANKLE_SWING_HANDLE]
        assert len(values) == len(used_joints)
        v = [0] * len(self.handlesList)
        for j_name, value in zip(used_joints, values):
            v[self.get_joint_index(j_name)] = value
        super(Leg, self).move_joints(client_id, v)

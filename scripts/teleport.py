import vrep
from utils import Utils
from pybrain_components import StandingUpSimulator


def get_handles(client_id):
    handles = []
    return_code, base_handler = vrep.simxGetObjectHandle(client_id, 'Bioloid', vrep.simx_opmode_blocking)
    t = [base_handler]
    while len(t) > 0:
        h = t.pop()
        index = 0
        return_code, child_handler = vrep.simxGetObjectChild(client_id, h, index, vrep.simx_opmode_blocking)
        while child_handler != -1:
            handles.append(child_handler)
            t.append(child_handler)
            index += 1
            return_code, child_handler = vrep.simxGetObjectChild(client_id, h, index, vrep.simx_opmode_blocking)
    return handles


def read_state(client_id, handles):
    state = []
    for h in handles:
        return_code, position = vrep.simxGetObjectPosition(client_id, h, -1, vrep.simx_opmode_blocking)
        return_code, orientation = vrep.simxGetObjectOrientation(client_id, h, -1, vrep.simx_opmode_blocking)
        state.append((position, orientation))
    return state


def write_state(client_id, handles, state):
    for h, s in zip(handles, state):
        vrep.simxSetObjectPosition(client_id, h, -1, s[0], vrep.simx_opmode_blocking)
        vrep.simxSetObjectOrientation(client_id, h, -1, s[1], vrep.simx_opmode_blocking)

def main():
    client_id = Utils.connectToVREP()

    env = StandingUpSimulator(client_id)


    handles = get_handles(client_id)
    state = read_state(client_id, handles)
    ang = env.bioloid.read_state()

    env.performAction(Utils.standingUpActions[0])
    env.performAction(Utils.standingUpActions[1])
    state = read_state(client_id, handles)
    ang = env.bioloid.read_state()
    env.performAction(Utils.standingUpActions[2])
    env.performAction(Utils.standingUpActions[3])
    env.performAction(Utils.standingUpActions[4])
    env.performAction(Utils.standingUpActions[5])
    env.performAction(Utils.standingUpActions[6])

    vrep.simxSetIntegerSignal(client_id, 'reset-dynamics', 1, vrep.simx_opmode_blocking)
    vrep.simxSynchronousTrigger(client_id)

    for i, h in enumerate(env.bioloid.leftArm.handlesList):
        vrep.simxSetJointPosition(client_id, h, ang[i + 7], vrep.simx_opmode_blocking)

    for i, h in enumerate(env.bioloid.rightArm.handlesList):
        vrep.simxSetJointPosition(client_id, h, ang[i + 10], vrep.simx_opmode_blocking)

    for j_name, value in zip(env.bioloid.leftLeg.used_joints, ang[13:16]):
        joint_handle = env.bioloid.leftLeg.handlesDict[j_name]
        vrep.simxSetJointPosition(client_id, joint_handle, value, vrep.simx_opmode_blocking)

    for j_name, value in zip(env.bioloid.rightLeg.used_joints, ang[16:]):
        joint_handle = env.bioloid.rightLeg.handlesDict[j_name]
        vrep.simxSetJointPosition(client_id, joint_handle, value, vrep.simx_opmode_blocking)


    write_state(client_id, handles, state)

    # vrep.simxSetIntegerSignal(client_id, 'reset-dynamics', 1, vrep.simx_opmode_blocking)
    # vrep.simxSynchronousTrigger(client_id)

    for i in range(2, len(Utils.standingUpActions)):
        env.performAction(Utils.standingUpActions[i])

if __name__ == '__main__':
    main()

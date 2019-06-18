import pickle

from models.pybrain import StandingUpEnvironment, StandingUpTask
from utils import Utils


def main():

    client_id = Utils.connectToVREP()
    environment = StandingUpEnvironment(client_id)
    task = StandingUpTask(environment)

    #print('Initial State: ')
    #print(environment.bioloid.read_state())

    trajectory_data = []
    for i in range(50):
        print('Iteration {}'.format(i))
        trajectory = []
        for action in Utils.standingUpActions:
            observation = task.getObservation()[0]
            state_vector = task.env.bioloid.read_state()
            action = Utils.vecToInt(action)
            task.performAction(action)
            reward = task.getReward()
            trajectory.append({'state': observation, 'state_vector': state_vector, 'action': action, 'reward': reward,
                               'full_state': task.env.bioloid.read_full_state()})
        trajectory_data.append(trajectory)
        observation = task.getObservation()[0]
        state_vector = task.env.bioloid.read_state()
        trajectory.append({'state': observation, 'state_vector': state_vector, 'action': -1, 'reward': 0,
                           'full_state': task.env.bioloid.read_full_state()})
        task.reset()

    with open('../data/trajectory.pkl', 'wb') as file:
        pickle.dump(trajectory_data, file)
    Utils.endVREP()

if __name__ == '__main__':
    main()

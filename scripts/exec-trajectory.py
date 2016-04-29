import pickle

from pybrain_components import StandingUpSimulator, StandingUpTask
from utils import Utils


def main():

    client_id = Utils.connectToVREP()
    environment = StandingUpSimulator(client_id)
    task = StandingUpTask(environment)

    #print('Initial State: ')
    #print(environment.bioloid.read_state())

    trajectory_data = []
    for i in range(25):
        print('Iteration {}'.format(i))
        trajectory = []
        for action in Utils.standingUpActions:
            observation = task.getObservation()[0]
            state_vector = task.env.bioloid.read_state()
            a = Utils.vecToInt(action)
            task.performAction(a)
            reward = task.getReward()
            # task.performAction(a)
            # task.getReward()
            # environment.performAction(action)
            # print(environment.bioloid.read_state())
            # print('self-collided: '+ str(environment.bioloid.isFallen()))
            # print('is-fallen: '+ str(environment.bioloid.isFallen()))
            # input("Press Enter to continue...")
            trajectory.append({'state': observation, 'state_vector': state_vector, 'action': action, 'reward': reward})
        trajectory_data.append(trajectory)
        observation = task.getObservation()[0]
        state_vector = task.env.bioloid.read_state()
        trajectory.append({'state': observation, 'state_vector': state_vector, 'action': -1, 'reward': 0})
        task.reset()

    with open('data/trajectory.pkl', 'wb') as file:
        pickle.dump(trajectory_data, file)
    Utils.endVREP()

if __name__ == '__main__':
    main()

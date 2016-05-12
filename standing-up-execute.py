import pickle

import vrep
from pybrain.rl.agents import LearningAgent
from pybrain.rl.experiments import Experiment, EpisodicExperiment
from pybrain.rl.learners import ActionValueTable, Q
from pybrain_components import StandingUpTask, StandingUpEnvironment


def main():
    vrep.simxFinish(-1)  # just in case, close all opened connections
    client_id = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)  # Connect to V-REP

    if client_id < 0:
        print('Failed connecting to remote API server')
        return -1

    print('Connected to remote API server')

    # Define RL elements
    environment = StandingUpEnvironment(client_id)

    task = StandingUpTask(environment)

    controller = ActionValueTable(task.get_state_space_size(), task.get_action_space_size())
    controller.initialize(1.)

    file = open('standing-up-q.pkl', 'rb')
    controller._params = pickle.load(file)
    file.close()

    # learner = Q()
    agent = LearningAgent(controller)

    experiment = EpisodicExperiment(task, agent)

    i = 0
    while True:
        i += 1
        print('Iteration n° ' + str(i))
        experiment.doEpisodes(1)

    vrep.simxFinish(client_id)


if __name__ == '__main__':
    main()

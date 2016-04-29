import pickle
import vrep
import pylab
from pybrain.rl.agents import LearningAgent
from pybrain.rl.experiments import Experiment
from pybrain.rl.learners import ActionValueTable, Q
from pybrain_components import RaiseArmSimulator, RaiseArmsTask


def main():
    vrep.simxFinish(-1)  # just in case, close all opened connections
    client_id = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)  # Connect to V-REP

    if client_id < 0:
        print('Failed connecting to remote API server')
        return -1

    print('Connected to remote API server')

    pylab.gray()
    pylab.ion()

    # Define RL elements
    environment = RaiseArmSimulator(client_id)

    controller = ActionValueTable(260, 27)
    controller.initialize(1.)

    learner = Q()
    agent = LearningAgent(controller, learner)

    task = RaiseArmsTask(environment)

    experiment = Experiment(task, agent)

    i = 0
    try:
        while True:
            i += 1
            print('Iteration n° ' + str(i))
            experiment.doInteractions(50)
            agent.learn()
            agent.reset()

            pylab.pcolor(controller.params.reshape(260, 27))
            pylab.draw()
            # pylab.pause(0.1)
    except (KeyboardInterrupt, SystemExit):
        with open('rl.pkl', 'wb') as handle:
            pickle.dump(controller.params, handle)

    vrep.simxFinish(client_id)


if __name__ == '__main__':
    main()

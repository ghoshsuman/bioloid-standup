import pickle

import numpy

import vrep
import pylab
import os
from pybrain.rl.agents import LearningAgent
from pybrain.rl.experiments import Experiment, EpisodicExperiment
from pybrain.rl.learners import ActionValueTable, Q
from pybrain_components import StandingUpSimulator, StandingUpTask

Q_TABLE_DIR = 'q-tables/'


def read_qtable(qtable_version = 1):
    filepath = os.path.join(Q_TABLE_DIR, 'q-table-' + str(qtable_version) + '.pkl')
    qtable = None
    if os.path.isfile(filepath):
        with open(filepath, 'rb') as file:
            qtable = pickle.load(file)
    return qtable


def write_qtable(qtable, qtable_version = 1):
    filepath = os.path.join(Q_TABLE_DIR, 'q-table-' + str(qtable_version) + '.pkl')
    with open(filepath, 'wb') as file:
        pickle.dump(qtable, file)


def main():
    vrep.simxFinish(-1)  # just in case, close all opened connections
    client_id = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)  # Connect to V-REP
    qtable_version = 1

    if client_id < 0:
        print('Failed connecting to remote API server')
        return -1

    print('Connected to remote API server')

    pylab.gray()
    pylab.ion()

    # Define RL elements
    environment = StandingUpSimulator(client_id)
    task = StandingUpTask(environment)

    controller = ActionValueTable(task.getStateNumber(), task.getActionNumber())
    controller.initialize(20.)

    # If the q-table file exist load it
    qtable = read_qtable(qtable_version)
    if qtable is not None:
        controller.params = qtable

    learner = Q(0.5, 0.8)
    agent = LearningAgent(controller, learner)

    experiment = EpisodicExperiment(task, agent)

    i = 0
    try:
        while True:
            i += 1
            print('Iteration n° ' + str(i))
            experiment.doEpisodes(10)
            agent.learn()
            agent.reset()
            print('mean: '+str(numpy.mean(controller.params)))
            print('max: '+str(numpy.max(controller.params)))
            print('min: '+str(numpy.min(controller.params)))

            if i % 10 == 0:  # Every 100 episodes
                print('Save q-table')
                qtable_version += 1
                write_qtable(controller.params, qtable_version)
                task.t_table.save(os.path.join(Q_TABLE_DIR, 't-table-'+str(qtable_version)+'.pkl'))

            # pylab.pcolor(controller.params.reshape(task.getStateNumber(), task.getActionNumber()))
            # pylab.draw()
    except (KeyboardInterrupt, SystemExit):
        with open('standing-up-q.pkl', 'wb') as handle:
            pickle.dump(controller.params, handle)
        task.t_table.save()

    vrep.simxFinish(client_id)


if __name__ == '__main__':
    main()

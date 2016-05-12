import pickle

import numpy
from scipy.spatial.distance import euclidean

import vrep
import os
from pybrain.rl.agents import LearningAgent
from pybrain.rl.experiments import Experiment, EpisodicExperiment
from pybrain.rl.explorers import BoltzmannExplorer, EpsilonGreedyExplorer
from pybrain.rl.learners import ActionValueTable, Q, QLambda
from pybrain_components import StandingUpEnvironment, StandingUpTask
from utils import Utils
from egreedy_boltzmann import EpsilonGreedyBoltzmannExplorer

Q_TABLE_DIR = 'data/learning-tables/'


def read_qtable(qtable_version = 0):
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
    client_id = Utils.connectToVREP()
    qtable_version = 14

    # Define RL elements
    environment = StandingUpEnvironment(client_id)
    task = StandingUpTask(environment)

    controller = ActionValueTable(task.get_state_space_size(), task.get_action_space_size())
    controller.initialize(10.)

    # If the q-table file exist load it
    qtable = read_qtable(qtable_version)
    if qtable is not None:
        print('qtable loaded')
        controller._params = qtable

    learner = Q(0.5, 0.9)
    agent = LearningAgent(controller, learner)

    learner.explorer = EpsilonGreedyExplorer(0.15, 1)  # EpsilonGreedyBoltzmannExplorer()
    experiment = EpisodicExperiment(task, agent)

    with open('data/trajectory.pkl', 'rb') as file:
        trajectory_data = pickle.load(file)

        for trajectory in trajectory_data:
            for t in trajectory:
                if t['action'] == -1:
                    continue
                agent.integrateObservation(t['state'])
                action = Utils.vecToInt(t['action'])
                agent.lastaction = action
                agent.giveReward(t['reward'])
            agent.learn()
            agent.reset()

    '''for _ in range(20):
        for action in Utils.standingUpActions:
            agent.integrateObservation(task.getObservation())
            a = Utils.vecToInt(action)
            agent.lastaction = a
            task.performAction(a)
            reward = task.getReward()
            agent.giveReward(reward)
        print('*****************')
        agent.learn()
        agent.reset()
        task.reset()
        print('mean: '+str(numpy.mean(controller.params)))
        print('max: '+str(numpy.max(controller.params)))
        print('min: '+str(numpy.min(controller.params)))

    for action in Utils.standingUpActions:
        state = task.getObservation()
        a = Utils.vecToInt(action)
        print('max action: {}'.format(controller.getMaxAction(state)))
        print(controller.getActionValues(state)[a])
        task.performAction(a)
    '''

    i = 0
    try:
        while True:
            i += 1
            print('Episode ' + str(i))
            experiment.doEpisodes()
            agent.learn()
            agent.reset()
            print('mean: '+str(numpy.mean(controller.params)))
            print('max: '+str(numpy.max(controller.params)))
            print('min: '+str(numpy.min(controller.params)))

            if i % 500 == 0:  # Every 500 episodes
                print('Save q-table')
                qtable_version += 1
                write_qtable(controller.params, qtable_version)
                task.t_table.save(os.path.join(Q_TABLE_DIR, 't-table-'+str(qtable_version)+'.pkl'))

    except (KeyboardInterrupt, SystemExit):
        with open('data/standing-up-q.pkl', 'wb') as handle:
            pickle.dump(controller.params, handle)
        task.t_table.save()

    vrep.simxFinish(client_id)


if __name__ == '__main__':
    main()

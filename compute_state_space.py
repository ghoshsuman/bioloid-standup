import pickle

from scipy.spatial import KDTree
import numpy
import vrep
import pylab
from pybrain.rl.agents import LearningAgent
from pybrain.rl.experiments import Experiment
from pybrain.rl.learners import ActionValueTable, Q
from pybrain_components import StandingUpSimulator, StandingUpTask


def main():

    N_ACTIONS = 729
    data = []

    staningUpActions = [[0, 1, 0, 1, -1, 0],
                        [0, 1, 0, 1, -1, 0],
                        [0, 1, 0, 1, -1, 0],
                        [-1, 0, 0, -1, 0, 1],
                        [0, 0, 0, -1, 0, 0],
                        [0, 0, 0, -1, 0, 0],
                        [0, 0, 0, -1, 0, 0],
                        [1, 0, 0, -1, 0, 1],
                        [0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 1, 0, 0],
                        [0, -1, 0, 1, 1, -1],
                        [0, -1, 0, -1, 1, 0],
                        [0, -1, 0, 0, 1, -1],
                        ]

    vrep.simxFinish(-1)  # just in case, close all opened connections
    client_id = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)  # Connect to V-REP

    if client_id < 0:
        print('Failed connecting to remote API server')
        return -1

    print('Connected to remote API server')

    # Define RL elements
    environment = StandingUpSimulator(client_id)

    for index, targetAction in enumerate(staningUpActions):
        for a in range(N_ACTIONS):
            environment.reset()
            for i in range(index):
                environment.performAction(staningUpActions[i])
            environment.performAction(environment.intToVec(a))
            data.append(environment.getSensors())
            print('N states: '+str(len(data)))

    with open('state-space.pkl', 'wb') as handle:
        pickle.dump(data, handle)

    kdtree = KDTree(data)
    epsilon = 0.01

    print(environment.vecToInt([0, -1, 0, 0, 0, 0]))
    print(environment.intToVec(445))
    print(len(data))
    numpy.set_printoptions(threshold=numpy.nan)
    n_similar = [0] * len(data)
    for i in range(len(data)):
        _, indexes = kdtree.query(data[i], len(data), distance_upper_bound=epsilon)
        n_similar[i] = sum(1 for x in indexes if i < x < len(data))
        print(str(i)+': '+str(n_similar[i]))

    filtered_data = [data[i] for i in range(len(data)) if n_similar[i] == 0]
    print('filtered_data')
    print(len(filtered_data))
    data = filtered_data

    with open('state-space-filtered.pkl', 'wb') as handle:
        pickle.dump(data, handle)

    kdtree = KDTree(data)

    '''
    s = environment.getSensors()
    print('Initial State: ')
    print(s)
    print(kdtree.query(s))


    for a in staningUpActions:
        environment.performAction(a)
        s = environment.getSensors()
        print(s)
        # print(kdtree.query(s))
    '''

    vrep.simxFinish(client_id)


if __name__ == '__main__':
    main()

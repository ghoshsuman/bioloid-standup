import pickle

import numpy
from scipy.spatial import KDTree
from scipy.spatial.distance import euclidean

from pybrain_components import StandingUpSimulator

N_ACTIONS = 729

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

with open('state-space.pkl', 'rb') as handle:
    data = pickle.load(handle)

    kdtree = KDTree(data)

    distances = []

    for index, targetState in enumerate(staningUpActions):
        max_dist = 0
        j = StandingUpSimulator.vecToInt(targetState)
        t = data[j + index * N_ACTIONS]
        for i in range(N_ACTIONS):

            s = data[i + index * N_ACTIONS]
            d = euclidean(s, t)
            if d > max_dist:
                max_dist = d
        distances.append(max_dist)

    numpy.set_printoptions(threshold=numpy.nan)
    print(distances)
    max_dist = numpy.max(distances)
    print('delta: ' + str(max_dist / 2))

    '''
    min_dists = []
    for i in range(len(data)):
        dists, indexes = kdtree.query(data[i], len(data))
        min_dists.append(dists[1])

    numpy.set_printoptions(threshold=numpy.nan)
    print(min_dists)
    print('min')
    print(numpy.min(min_dists))
    print('max')
    print(numpy.max(min_dists))
    '''

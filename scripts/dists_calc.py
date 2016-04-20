import pickle

import numpy
import pylab
from scipy.spatial import KDTree
from scipy.spatial.distance import euclidean
from scipy.stats import probplot
import statsmodels.api as sm

from StateNormalizer import StateNormalizer
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
    data = numpy.matrix(data)

    norm = StateNormalizer()

    for i in range(len(data)):
        data[i, :] = norm.normalize(data[i, :])

    with open('state-space-normalized2.pkl', 'wb') as file:
        pickle.dump(data, file)

    kdtree = KDTree(data)

    epsilon = 0.001

    print('len data: ')
    print(len(data))
    numpy.set_printoptions(threshold=numpy.nan)
    n_similar = numpy.zeros(len(data), dtype=int)
    for i in range(len(data)):
        _, indexes = kdtree.query(data[i], len(data), distance_upper_bound=epsilon)
        indexes = indexes.transpose()
        n_similar[i] = sum(1 for x in indexes if i < x < len(data))
        print(str(i)+': '+str(n_similar[i]))

    filtered_index = []
    filtered_data = []
    for i in range(len(data)):
        if n_similar[i] == 0:
            filtered_data.append(data[i])
        else:
            filtered_index.append(i)
    filtered_data = [data[i] for i in range(len(data)) if n_similar[i] == 0]
    print('filtered_data')
    print(len(filtered_data))
    # data = filtered_data

    with open('state-space-filtered-normalized2.pkl', 'wb') as handle:
        pickle.dump(data, handle)
    '''

    distances = []

    for index, targetState in enumerate(staningUpActions):
        max_dist = 100
        j = StandingUpSimulator.vecToInt(targetState)
        t = data[j + index * N_ACTIONS]
        for i in range(N_ACTIONS):

            s = data[i + index * N_ACTIONS]
            d = euclidean(s, t)
            if d != 0:  # and (i + index * N_ACTIONS) not in filtered_index:
                distances.append(d)

    print('mean: {}'.format(numpy.mean(distances)))
    print('var: {}'.format(numpy.var(distances)))

    numpy.set_printoptions(threshold=numpy.nan)
    print(numpy.min(distances))
    print(numpy.max(distances))

    # measurements = numpy.random.normal(loc = 20, scale = 5, size=100)
    # probplot(measurements, dist="norm", plot=pylab)
    # pylab.show()

    probplot(distances, dist="norm", plot=pylab)
    pylab.show()

    # sm.qqplot(numpy.array(distances), line='45')
    # pylab.show()

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


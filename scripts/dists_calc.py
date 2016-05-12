import pickle

import numpy
import pylab
import xlsxwriter
from scipy.spatial import KDTree
from scipy.spatial.distance import euclidean
from scipy.stats import probplot
import statsmodels.api as sm

from StateNormalizer import StateNormalizer
from pybrain_components import StandingUpEnvironment
from utils import Utils


def getState(data, traj_step, action):
    for k in range(len(data)):
        if data[k]['is-fallen'] or data[k]['self-collided']:
            continue
        elif data[k]['trajectory-step'] == traj_step and data[k]['action'] == action:
            return data[k]['state_vector']
    return None


def main():

    data = []

    workbook = xlsxwriter.Workbook('data/distances.xls')
    all_distances_sheet = workbook.add_worksheet('All Distances')
    min_distances_sheet = workbook.add_worksheet('Min Distances')

    for t in range(len(Utils.standingUpActions)):
        with open('data/state-space/state-space-t{}-0.pkl'.format(t), 'rb') as handle:
            data += pickle.load(handle)

    sn = StateNormalizer()
    sn.extend_bounds()

    distances = []

    for index, targetAction in enumerate(Utils.standingUpActions):
        action = Utils.vecToInt(targetAction)
        targetState = getState(data, index, action)
        targetState = sn.normalize(targetState)
        for i in range(Utils.N_ACTIONS):
            if i == Utils.NULL_ACTION or i == action:
                continue
            s = getState(data, index, i)

            if s is None or targetState is None:
                continue
            s = sn.normalize(s)
            d = euclidean(s, targetState)
            distances.append(d)

    print('mean: {}'.format(numpy.mean(distances)))
    print('var: {}'.format(numpy.var(distances)))
    print('median: '.format(numpy.median(distances)))
    print('max: {}'.format(numpy.max(distances)))
    print('min: {}'.format(numpy.min(distances)))

    all_distances_sheet.write(0, 0, 'Min')
    all_distances_sheet.write(0, 1, numpy.min(distances))
    all_distances_sheet.write(1, 0, 'Max')
    all_distances_sheet.write(1, 1, numpy.max(distances))
    all_distances_sheet.write(2, 0, 'Mean')
    all_distances_sheet.write(2, 1, numpy.mean(distances))
    all_distances_sheet.write(3, 0, 'Variance')
    all_distances_sheet.write(3, 1, numpy.var(distances))
    all_distances_sheet.write(4, 0, 'Median')
    all_distances_sheet.write(4, 1, numpy.median(distances))

    # measurements = numpy.random.normal(loc = 20, scale = 5, size=100)
    # probplot(measurements, dist="norm", plot=pylab)
    # pylab.show()

    probplot(distances, dist="norm", plot=pylab)
    pylab.show()

    # sm.qqplot(numpy.array(distances), line='45')
    # pylab.show()

    with open('data/state-space/state-space-all-0.pkl', 'rb') as handle:
        data = pickle.load(handle)

    for i in range(len(data)):
        data[i] = sn.normalize(data[i])

    kdtree = KDTree(data)

    min_dists = []
    for i in range(len(data)):
        dists, indexes = kdtree.query(data[i], 2)
        min_dists.append(dists[1])

    #numpy.set_printoptions(threshold=numpy.nan)
    # print(min_dists)
    print('-----------------')
    print('min: {}'.format(numpy.min(min_dists)))
    print('max: {}'.format(numpy.max(min_dists)))
    print('mean: {}'.format(numpy.mean(min_dists)))
    print('variance: {}'.format(numpy.var(min_dists)))
    print('median: {}'.format(numpy.median(min_dists)))

    min_distances_sheet.write(0, 0, 'Min')
    min_distances_sheet.write(0, 1, numpy.min(min_dists))
    min_distances_sheet.write(1, 0, 'Max')
    min_distances_sheet.write(1, 1, numpy.max(min_dists))
    min_distances_sheet.write(2, 0, 'Mean')
    min_distances_sheet.write(2, 1, numpy.mean(min_dists))
    min_distances_sheet.write(3, 0, 'Variance')
    min_distances_sheet.write(3, 1, numpy.var(min_dists))
    min_distances_sheet.write(4, 0, 'Median')
    min_distances_sheet.write(4, 1, numpy.median(min_dists))

    workbook.close()

if __name__ == '__main__':
    main()

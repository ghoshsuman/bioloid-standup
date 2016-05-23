import pickle

import collections
import numpy

from StateMapper import StateMapper
from utils import Utils

safe_shutdown_action = Utils.N_ACTIONS
state_mapper = StateMapper()


def softMax(items, T):
    values = []

    for v in items:
        values.append(numpy.exp(v[1] / T))
    den = numpy.sum(values)
    for i, value in enumerate(values):
        values[i] = (items[i][0], values[i] / den)
    '''
    max = 0
    for i in range(len(items)):
        if items[i][1] > items[max][1]:
            max = i
    for i in range(len(items)):
        if i == max:
            values.append((items[i][0], 1))
        else:
            values.append((items[i][0], 0))
    '''
    return values


def computePolicy(Q, T):
    n_states, n_actions = Q.shape
    policy = numpy.zeros((n_states, n_actions + 1), dtype=float)

    for state in range(n_states):
        good_actions = []
        for action in range(n_actions):
            if Q[state, action] != 10: # Q[state, action] >= 0 and
                good_actions.append((action, Q[state, action]))
        if len(good_actions) > 0:
            values = softMax(good_actions, T)
            for i in range(len(values)):
                policy[state, values[i][0]] = values[i][1]
        else:
            policy[state, safe_shutdown_action] = 1

    return policy


def getSuccessorStates(ttable, state, action):
    total = 0
    successors = []
    if state == state_mapper.goal_state:
        return [(state, 1)]
    if action == safe_shutdown_action or state == state_mapper.too_far_state or state == state_mapper.fallen_state or \
            state == state_mapper.self_collided_state:
        return [(state_mapper.INITIAL_STATE, 1)]

    for key, value in ttable.items():
        if key[0] != state or key[1] != action:
            continue
        else:
            total += value
            successors.append((key[2], value))
    for i, succ in enumerate(successors):
        successors[i] = (succ[0], succ[1] / total)
    if len(successors) == 0:
        successors = [(state_mapper.too_far_state, 1)]
    return successors


def getActionProbability(Q, state):
    actions = []


def main():
    with open('data/learning-tables/learning-20-may/t-table.pkl', 'rb') as file:
        ttable = pickle.load(file)

    with open('data/learning-tables/learning-20-may/q-table-227.pkl', 'rb') as file:
        qtable = pickle.load(file)

    Q = qtable.reshape(len(qtable) // Utils.N_ACTIONS, Utils.N_ACTIONS)
    n_states, n_actions = Q.shape

    T = 1
    policy = computePolicy(Q, T)

    print(n_states)

    dtmc = {}
    for state in range(n_states):
        print('State {}'.format(state))
        for action in range(n_actions + 1):
            if policy[state, action] == 0:
                continue
            # print('policy {} {} {}'.format(state, action, policy[state, action]))
            successors = getSuccessorStates(ttable, state, action)
            for succ in successors:
                succ_state = succ[0]
                prob = succ[1]
                # print('{} {}'.format(succ_state, prob))
                pr = policy[state, action] * prob
                v = dtmc.get((state, succ_state), 0)
                v += pr
                if v > 0:
                    dtmc[(state, succ_state)] = v

    file = open('dtmc.tra', 'w')
    file.write('dtmc\n')
    dtmc2 = collections.OrderedDict(dtmc)
    for trans, prob in sorted(dtmc.items()):
        file.write('{} {} {}\n'.format(trans[0], trans[1], prob))
    file.close()

    file = open('dtmc.lab', 'w')
    file.write('#DECLARATION\n')
    file.write('init fallen far collided goal\n')
    file.write('#END\n')
    file.write('{} init\n'.format(state_mapper.INITIAL_STATE))
    file.write('{} fallen\n'.format(state_mapper.fallen_state))
    file.write('{} collided\n'.format(state_mapper.self_collided_state))
    file.write('{} far\n'.format(state_mapper.too_far_state))
    file.write('{} goal\n'.format(state_mapper.goal_state))

    file.close()


if __name__ == '__main__':
    main()

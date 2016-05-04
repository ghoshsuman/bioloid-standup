import pickle

import collections
import numpy

from utils import Utils

initial_state = 0
goal_state = far_state = coll_state = fall_state = None
safe_shutdown_action = Utils.N_ACTIONS



def softMax(items, T):
    values = []
    '''for v in items:
        values.append(numpy.exp(v[1]/T))
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
    return values


def computePolicy(Q, T):
    n_states, n_actions = Q.shape
    policy = numpy.zeros((n_states, n_actions + 1), dtype=float)

    for state in range(n_states):
        good_actions = []
        for action in range(n_actions):
            if Q[state, action] >= 0 and Q[state, action] != 10:
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
    if state == goal_state :
        return [(state, 1)]
    if action == safe_shutdown_action or state == far_state or state == fall_state or state == coll_state:
        return [(initial_state, 1)]

    for key, value in ttable.items():
        if key[0] != state or key[1] != action:
            continue
        else:
            total += value
            successors.append((key[2], value))
    for i, succ in enumerate(successors):
        successors[i] = (succ[0], succ[1] / total)
    return successors


def getActionProbability(Q, state):
    actions = []


def main():

    with open('data/learning-tables/t-table-final.pkl', 'rb') as file:
        ttable = pickle.load(file)

    with open('data/learning-tables/q-table-19.pkl', 'rb') as file:
        qtable = pickle.load(file)

    Q = qtable.reshape(len(qtable) // Utils.N_ACTIONS, Utils.N_ACTIONS)
    n_states, n_actions = Q.shape
    global goal_state
    global far_state
    global coll_state
    global fall_state
    goal_state = 5675
    far_state  = n_states - 3
    coll_state = n_states - 4
    fall_state = n_states - 5

    T = 0.5  # 2.5
    policy = computePolicy(Q, T)

    dtmc = {}
    for state in range(n_states):
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
    file.write('{} init\n'.format(initial_state))
    file.write('{} goal\n'.format(goal_state))
    file.write('{} fallen\n'.format(fall_state))
    file.write('{} collided\n'.format(coll_state))
    file.write('{} far\n'.format(far_state))


    file.close()


if __name__ == '__main__':
    main()
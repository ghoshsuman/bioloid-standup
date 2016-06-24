import logging
import os
import pickle
import threading

import numpy
import time

from dtmc import DTMCGenerator
from pybrain.rl.learners import ActionValueTable
from pybrain_components import StandingUpEnvironment, StandingUpTask
from utils import Utils

BASE_DIR = 'data/learning-tables/learning-8-june-taclab/'
POLICY_PREFIX = 'sm2-rep'
Q_TABLE_VERSION = 881

def select_action(policy, state, method='argmax'):
    if method == 'argmax':
        return numpy.argmax(policy[state])
    elif method == 'prob':
        return numpy.random.choice(len(policy[state]), p=policy[state])
    else:
        raise ValueError('{} is not a supported method'.format(method))


def main():

    client_id = Utils.connectToVREP()

    # Define RL elements
    environment = StandingUpEnvironment(client_id)
    task = StandingUpTask(environment)

    temperature = 2
    ttable_path = os.path.join(BASE_DIR, 't-table.pkl')
    qtable_path = os.path.join(BASE_DIR, 'q-table-{}.pkl'.format(Q_TABLE_VERSION))
    dtmc_generator = DTMCGenerator(ttable_path, qtable_path, temperature)
    dtmc_generator.load_policy(POLICY_PREFIX + '-policy.pkl', BASE_DIR)

    with open(ttable_path, 'rb') as file:
        ttable = pickle.load(file)
        trans_prob_dict = {}

        for key, value in ttable.items():
            new_key = (key[0], key[1])
            v = trans_prob_dict.get(new_key, [])
            v.append((key[2], value))
            trans_prob_dict[new_key] = v

    while True:
        state = task.getObservation()[0]

        action = numpy.argmax(dtmc_generator.Q[state])
        print('trans probs {}'.format(trans_prob_dict.get((state, action))))

        action = select_action(dtmc_generator.policy, state, 'prob')
        print('State {} Action {} Prob {}'.format(state, action, dtmc_generator.policy[state][action]))
        print('pol trans probs {}'.format(trans_prob_dict.get((state, action))))

        if action == 729:
            task.reset()
        else:
            task.performAction(action)

        successors = dtmc_generator.get_successor_states(state, action)
        new_state = task.getObservation()[0]
        found = False
        for state, prob in successors:
            if new_state == state:
                print('{} state found, prob {}'.format(state, prob))
                found = True
        if not found:
            print('{} state not found! successors: {}'.format(new_state, successors))
    Utils.endVREP()

if __name__ == '__main__':
    main()
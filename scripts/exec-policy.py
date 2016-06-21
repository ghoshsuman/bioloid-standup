import logging
import os
import pickle
import threading

import numpy
import time

from pybrain.rl.learners import ActionValueTable
from pybrain_components import StandingUpEnvironment, StandingUpTask
from utils import Utils

BASE_DIR = 'data/learning-tables/learning-21-june-taclab/'
POLICY_PREFIX = 'sm2-rep'

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

    policy = None
    with open(os.path.join(BASE_DIR, POLICY_PREFIX + '-policy.pkl'), 'rb') as file:
        policy = pickle.load(file)

    while True:
        state = task.getObservation()[0]
        action = select_action(policy, state, 'prob')
        print('State {} Action {} Prob {}'.format(state, action, policy[state][action]))
        if action == 729:
            task.reset()
        else:
            task.performAction(action)
    Utils.endVREP()

if __name__ == '__main__':
    main()
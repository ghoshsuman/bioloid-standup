import logging
import os
import pickle
import threading

import numpy
import time

from pybrain.rl.learners import ActionValueTable
from pybrain_components import StandingUpEnvironment, StandingUpTask
from utils import Utils

BASE_DIR = 'data/learning-tables/learning-4-july-blade21/'
MAX_ITERATIONS = 300


class PolicyExecutor(threading.Thread):

    def __init__(self, port, policy_file_name='policy.pkl', base_dir='data/'):
        threading.Thread.__init__(self)
        self.port = port
        with open(os.path.join(base_dir, policy_file_name), 'rb') as file:
            self.policy = pickle.load(file)
        log_file_name = policy_file_name.replace('.pkl', '.log')
        self.logger = logging.getLogger(log_file_name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(logging.FileHandler(os.path.join(base_dir, log_file_name)))

    def run(self):
        proc = Utils.exec_vrep(self.port)
        time.sleep(10)
        # connect to V-REP server
        try:
            client_id = Utils.connectToVREP(self.port)
            env = StandingUpEnvironment(client_id)
            task = StandingUpTask(env)
            counter = 0
            while counter < MAX_ITERATIONS:

                state = task.getObservation()[0]
                action = select_action(self.policy, state, 'prob')
                # print('State {} Action {} Prob {}'.format(state, action, self.policy[state][action]))
                if action == 729:
                    if state == task.state_mapper.goal_state:
                        self.logger.info('Goal!')
                    elif state == task.state_mapper.fallen_state:
                        self.logger.info('Fallen!')
                    elif state == task.state_mapper.too_far_state:
                        self.logger.info('Far!')
                    elif state == task.state_mapper.self_collided_state:
                        self.logger.info('Collided!')
                    else:
                        self.logger.info(state)
                    counter += 1
                    print('Iteration {}'.format(counter))
                    task.reset()
                else:
                    task.performAction(action)
        finally:
            proc.kill()


def select_action(policy, state, method='argmax'):
    if method == 'argmax':
        return numpy.argmax(policy[state])
    elif method == 'prob':
        return numpy.random.choice(len(policy[state]), p=policy[state])
    else:
        raise ValueError('{} is not a supported method'.format(method))


def main():

    PolicyExecutor(8000, 'sm10-policy.pkl', BASE_DIR).start()
    PolicyExecutor(8001, 'sm10-rep-policy.pkl', BASE_DIR).start()


if __name__ == '__main__':
    main()
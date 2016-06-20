import os
import subprocess
import threading

import time

import pickle

import numpy

from pybrain_components import StandingUpEnvironment
from utils import Utils


class StateSpaceGenerationTask(threading.Thread):

    def __init__(self, port):
        threading.Thread.__init__(self)
        self.port = port

    def run(self):
        proc = Utils.exec_vrep(self.port)
        time.sleep(10)
        # connect to V-REP server
        try:
            client_id = Utils.connectToVREP(self.port)
            env = StandingUpEnvironment(client_id)

            trajectory_states = []

            print('Executing trajectory...')
            env.performAction(Utils.NULL_ACTION_VEC)
            for action in Utils.standingUpActions:
                print('Action {}'.format(action))
                for i in range(2):
                    trajectory_states.append(env.bioloid.read_full_state())
                    env.performAction(numpy.array(action) / 2)

            print('Calculating new states...')
            for t, traj_state in enumerate(trajectory_states):
                print('Step {}'.format(t))
                data = []
                for action in range(Utils.N_ACTIONS):
                    print('Action {}'.format(action))
                    env.bioloid.set_full_state(traj_state)
                    env.performAction(numpy.array(Utils.intToVec(action)) / 2)
                    bioloid = env.bioloid
                    row = {'action': action, 'state_vector': bioloid.read_state(), 'is-fallen': bioloid.is_fallen(),
                           'self-collided': bioloid.is_self_collided(), 'trajectory-step': t}
                    data.append(row)
                with open('data/state-space/state-space-t{}-{}.pkl'.format(t, self.port), 'wb') as file:
                    pickle.dump(data, file)
        finally:
            proc.kill()


def main():
    n_thread = 10
    for i in range(n_thread):
        StateSpaceGenerationTask(8000 + i).start()

if __name__ == '__main__':
    main()
from threading import Thread, Barrier

import time

import numpy

from NDSparseMatrix import NDSparseMatrix
from pybrain_components import StandingUpEnvironment, StandingUpTask
from utils import Utils


class PolicyExecutor(Thread):

    def __init__(self, port, n_episodes, barrier, policy, method='prob'):
        Thread.__init__(self)
        self.t_table = NDSparseMatrix()
        self.port = port
        self.n_episodes = n_episodes
        self.barrier = barrier
        self.policy = policy
        self.method = method
        self.counters = {'goal': 0, 'far': 0, 'fallen': 0, 'collided': 0, 'unknown': 0}

    def select_action(self, policy, state):
        if self.method == 'argmax':
            return numpy.argmax(policy[state])
        elif self.method == 'prob':
            return numpy.random.choice(len(policy[state]), p=policy[state])
        else:
            raise ValueError('{} is not a supported method'.format(self.method))

    def run(self):
        try:
            proc = Utils.exec_vrep(self.port)
            time.sleep(60)

            client_id = Utils.connectToVREP(self.port)
            environment = StandingUpEnvironment(client_id)
            task = StandingUpTask(environment)

            for episode in range(self.n_episodes):

                old_state = current_state = task.getObservation()[0]
                action = self.select_action(self.policy, current_state)
                print('State {} Action {} Prob {}'.format(current_state, action, self.policy[current_state][action]))
                task.performAction(action)
                while action != 729:
                    old_state = current_state
                    current_state = task.getObservation()[0]
                    self.t_table.incrementValue((old_state, action, current_state))
                    action = self.select_action(self.policy, current_state)
                    print('State {} Action {} Prob {}'.format(current_state, action, self.policy[current_state][action]))
                    task.performAction(action)
                task.reset()

                if current_state == task.state_mapper.goal_state:
                        self.counters['goal'] += 1
                elif current_state == task.state_mapper.fallen_state:
                        self.counters['fallen'] += 1
                elif current_state == task.state_mapper.too_far_state:
                        self.counters['far'] += 1
                elif current_state == task.state_mapper.self_collided_state:
                        self.counters['collided'] += 1
                else:
                        self.counters['unknown'] += 1
        finally:
            self.barrier.wait()
            # proc.kill()


class Monitor:

    def __init__(self, dtmc_generator, n_episodes=50, n_trheads=4, initial_port=8000):
        self.dtmc_geneartor = dtmc_generator
        self.n_episodes = n_episodes
        self.n_threads = n_trheads
        self.initial_port = initial_port

    def simulate_policy(self):

        simulators = []
        barrier = Barrier(self.n_threads + 1)
        for i in range(self.n_threads):
            executor = PolicyExecutor(self.initial_port + i, self.n_episodes, barrier, self.dtmc_geneartor.policy)
            simulators.append(executor)
            executor.start()
        barrier.wait()
        self.dtmc_geneartor
        for sim in simulators:
            self.dtmc_geneartor.t_table.add(sim.t_table)
            print(sim.counters)

        self.dtmc_geneartor.t_table.save('data/t-table-monitor.pkl')





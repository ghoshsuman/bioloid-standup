import logging
import os
import subprocess
from threading import Thread, Barrier

import time

import numpy

from NDSparseMatrix import NDSparseMatrix
from model_repair import DeltaRepairer
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
                    # Test to verify Monitor capability
                    if action == 579:
                        current_state = task.state_mapper.self_collided_state
                    else:
                        current_state = task.getObservation()[0]
                    self.t_table.incrementValue((old_state, action, current_state))
                    action = self.select_action(self.policy, current_state)
                    print(
                        'State {} Action {} Prob {}'.format(current_state, action, self.policy[current_state][action]))
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
            Utils.endVREP(client_id)
            proc.kill()


class Monitor:
    K = 1000

    def __init__(self, dtmc_generator, model_repairer, base_dir, n_episodes=50, n_trheads=4, initial_port=8000):
        self.dtmc_generator = dtmc_generator
        self.model_repairer = model_repairer
        self.n_episodes = n_episodes
        self.n_threads = n_trheads
        self.initial_port = initial_port
        self.base_dir = base_dir
        self.itr = 0

        # Logger initialization
        self.logger = logging.getLogger('monitor_logger')
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(
            logging.FileHandler(os.path.join(base_dir, 'monitor_{}.log'.format(dtmc_generator.temp))))

        self.t_table = NDSparseMatrix()
        self.t_table.add(self.dtmc_generator.t_table)

        self.transition_counters = NDSparseMatrix()
        for (s1, a, s2), value in self.t_table.items():
            self.transition_counters.incrementValue((s1, a), amount=value)

        self._normalize_t_table()

    def simulate_policy(self):

        simulators = []
        barrier = Barrier(self.n_threads + 1)
        for i in range(self.n_threads):
            executor = PolicyExecutor(self.initial_port + i, self.n_episodes, barrier, self.dtmc_generator.policy)
            simulators.append(executor)
            executor.start()
        barrier.wait()
        counters = {'goal': 0, 'far': 0, 'fallen': 0, 'collided': 0, 'unknown': 0}
        for sim in simulators:
            self._add_to_t_table(sim.t_table)
            self._add_to_t_table(sim.t_table)
            print(sim.counters)
            for key, value in sim.counters.items():
                counters[key] += value

        self.logger.info('Simulation values: {}'.format(counters))

        self.dtmc_generator.trans_prob_dict = self.dtmc_generator.compute_transition_probabilities_dict(
            self.transition_counters)
        self.t_table.save('data/repair/t-table-{}.pkl'.format(self.itr))
        subprocess.Popen('pkill vrep', shell=True)

        return counters

    def load(self, itr):
        self.itr = itr
        policy_file_name = 'policy-sm{}-{}.pkl'.format(self.dtmc_generator.temp, self.itr)
        # self.dtmc_generator.load_policy(policy_file_name, self.base_dir)
        self.t_table.load('data/repair/t-table-{}.pkl'.format(self.itr))
        self.transition_counters = NDSparseMatrix()
        for (s1, a, s2), value in self.t_table.items():
            self.transition_counters.incrementValue((s1, a), amount=value)

        self.dtmc_generator.t_table.load('data/repair/t-table-{}.pkl'.format(self.itr))
        self._normalize_t_table()

    def perform_repair(self):
        dtmc_file_name = 'dtmc-sm{}'.format(self.dtmc_generator.temp)
        policy_file_name = 'policy-sm{}-{}.pkl'.format(self.dtmc_generator.temp, self.itr)
        dtmc = self.dtmc_generator.compute_dtmc()
        self.logger.info('Prob before repair: {}'.format(dtmc.compute_probabilities()))

        dtmc = self.model_repairer.repair(DeltaRepairer(), dtmc_file_name, policy_file_name, 'data/repair')
        dtmc.save(dtmc_file_name + '-rep-{}'.format(self.itr), self.base_dir)
        self.dtmc_generator.save_policy(policy_file_name, self.base_dir)
        # dtmc_generator.load_policy(policy_file_name, BASE_DIR)
        self.logger.info('Prob after repair: {}'.format(dtmc.compute_probabilities()))

    def iteration(self):
        self.logger.info('Iteration {}'.format(self.itr))
        self.perform_repair()
        self.simulate_policy()
        self.itr += 1

    def _add_to_t_table(self, t_table):
        self.t_table.add(t_table)
        for (s1, a, s2), value in t_table.items():
            self.transition_counters.incrementValue((s1, a), value)
            if self.transition_counters.getValue((s1, a)) > self.K - value:
                self._update_t_table(s1, a, self.K - value)
            self.dtmc_generator.t_table.incrementValue((s1, a , s2), amount=value)

    def _update_t_table(self, s1, a, k=K):
        successors = self.dtmc_generator.get_successor_states(s1, a)
        new_successors = []
        total = 0
        for (s2, prob) in successors:
            val = self.dtmc_generator.t_table.getValue((s1, a, s2))
            new_successors.append((s2, val))
            total += val
        alpha = k / total
        for (s2, prob) in new_successors:
            val = self.dtmc_generator.t_table.getValue((s1, a, s2))
            self.dtmc_generator.t_table.setValue((s1, a, s2), val * alpha)

    def _normalize_t_table(self):
        for (s1, a), counter in self.transition_counters.items():
            if counter > self.K:
                self._update_t_table(s1, a)

        self.dtmc_generator.trans_prob_dict = self.dtmc_generator.compute_transition_probabilities_dict(
            self.transition_counters)

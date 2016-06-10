import os

import numpy
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import shortest_path

import stormpy
import stormpy.logic
from dtmc import DTMCGenerator, DTMC, state_mapper

BASE_DIR = 'data/learning-tables/learning-25-may-blade21/'
Q_TABLE_VERSION = 419
temperature = 2

class ModelRepairer:

    epsilon = 10 ** -8

    def __init__(self, dtmc_generator, _lambda=0.05):
        self.formula = stormpy.parse_formulas("P=? [ F \"collided\" ]")
        self.dtmc_generator = dtmc_generator
        self._lambda = _lambda

    def repair(self, file_name, base_dir):
        self.dtmc_generator.compute_policy()
        dtmc = self.dtmc_generator.compute_dtmc()
        dtmc.save(file_name, base_dir)
        model = stormpy.parse_explicit_model(os.path.join(base_dir, file_name + '.tra'),
                                             os.path.join(base_dir, file_name + '.lab'))
        unsafe_reachability_prob = stormpy.model_checking_all(model, self.formula[0])
        print('Initial Unsafe Prob {}'.format(unsafe_reachability_prob[state_mapper.INITIAL_STATE]))
        while unsafe_reachability_prob[state_mapper.INITIAL_STATE] > self._lambda and \
                self.local_repair(unsafe_reachability_prob):
            dtmc = self.dtmc_generator.compute_dtmc()
            dtmc.save(file_name, base_dir)
            model = stormpy.parse_explicit_model(os.path.join(base_dir, file_name + '.tra'),
                                                 os.path.join(base_dir, file_name + '.lab'))
            unsafe_reachability_prob = stormpy.model_checking_all(model, self.formula[0])
            print('Prob {}'.format(unsafe_reachability_prob[state_mapper.INITIAL_STATE]))
        print('Final Unsafe Prob {}'.format(unsafe_reachability_prob[state_mapper.INITIAL_STATE]))

        return dtmc

    def local_repair(self, probs, max_states_to_repair=10000):
        # Takes the first max_states_to_repair states with highest probabilities to reach the unsafe state
        # Skip the last state because it is the unsafe state itself
        sorted_indexes = numpy.argsort(probs)
        n = len(sorted_indexes)
        states = sorted_indexes[n - max_states_to_repair - 2: n - 2]

        return self.repair_states(states, probs)

    def local_repair2(self, probs):
        n = state_mapper.get_state_space_size()
        M = lil_matrix((n, n))
        dtmc = self.dtmc_generator.compute_dtmc()
        for (s1, s2), p in dtmc.dtmc.items():
            M[s1, s2] = 1 - p
            if M[s1, s2] < 0:
                M[s1, s2] = 0.0

        dist_matrix, predecessors = shortest_path(M, method='D', return_predecessors=True)
        path = []
        state = state_mapper.self_collided_state
        while state != state_mapper.INITIAL_STATE and state >= 0:
            state = predecessors[state_mapper.INITIAL_STATE, state]
            path.append(state)
        print(path)
        return self.repair_states(reversed(path), probs)

    def repair_states(self, states, probs):
        repaired = False
        for state in states:
            if self.repair_state(state, probs):
                repaired = True
        return repaired

    def repair_state(self, state, probs):
        possible_actions = self.dtmc_generator.get_possible_actions(state)
        if len(possible_actions) < 2:
            return False
        probs_to_reach_unsafe_state = []
        for action in possible_actions:
            successors = self.dtmc_generator.get_successor_states(state, action)
            p = 0.0
            for (succ_state, prob) in successors:
                p += prob * probs[succ_state]  # * self.dtmc_generator.policy[state, action]
            probs_to_reach_unsafe_state.append(p)
        probs_to_reach_unsafe_state = numpy.array(probs_to_reach_unsafe_state)
        # print(probs_to_reach_unsafe_state)
        index = numpy.argmax(probs_to_reach_unsafe_state)
        action_to_reduce = possible_actions[index]
        prob_to_reduce = self.dtmc_generator.policy[state, action_to_reduce]
        index = numpy.argmin(probs_to_reach_unsafe_state)
        action_to_increase = possible_actions[index]
        prob_to_increase = self.dtmc_generator.policy[state, action_to_increase]

        if action_to_increase == action_to_reduce or prob_to_increase == prob_to_reduce:
            return False

        if prob_to_reduce > self.epsilon:
            self.dtmc_generator.policy[state, action_to_reduce] = self.epsilon
            self.dtmc_generator.policy[state, action_to_increase] += prob_to_reduce - self.epsilon
            print('State {} repaired!'.format(state))

            print('Action {} from {} to {}, Action {} from {} to {}'.format(
                action_to_reduce, prob_to_reduce, self.dtmc_generator.policy[state, action_to_reduce],
                action_to_increase, prob_to_increase, self.dtmc_generator.policy[state, action_to_increase]))
            return True
        else:
            print('State {} cannot be repaired! {}'.format(state, prob_to_reduce))
        return False





def main():
    ttable_path = os.path.join(BASE_DIR, 't-table.pkl')
    qtable_path = os.path.join(BASE_DIR, 'q-table-{}.pkl'.format(Q_TABLE_VERSION))
    file_name = 'dtmc-sm{}'.format(temperature)
    dtmc_generator = DTMCGenerator(ttable_path, qtable_path, temperature)
    model_repairer = ModelRepairer(dtmc_generator)
    dtmc = model_repairer.repair(file_name, 'data/repair')
    dtmc.save(file_name + 'final', 'data/repair')


if __name__ == '__main__':
    main()
import os
from abc import ABC, abstractmethod

import numpy

import stormpy
from dtmc import state_mapper


class ModelRepairer:
    def __init__(self, dtmc_generator, _lambda=0.05):
        self.dtmc_generator = dtmc_generator
        self.formula = stormpy.parse_formulas("P=? [ F (\"far\"  | \"collided\" | \"fallen\")]")
        self._lambda = _lambda

    def repair(self, local_repairer, dtmc_name, policy_name, base_dir):
        dtmc = self.dtmc_generator.compute_dtmc()
        dtmc.save(dtmc_name, base_dir)
        dtmc_tra = os.path.join(base_dir, dtmc_name + '.tra')
        dtmc_lab = os.path.join(base_dir, dtmc_name + '.lab')
        model = stormpy.parse_explicit_model(dtmc_tra, dtmc_lab)
        unsafe_reachability_prob = stormpy.model_checking_all(model, self.formula[0])

        init_unsafe_prob = unsafe_reachability_prob[state_mapper.INITIAL_STATE]
        print('Init Unsafe Prob {}'.format(init_unsafe_prob))
        while init_unsafe_prob > self._lambda and local_repairer.repair(self.dtmc_generator, unsafe_reachability_prob):
            dtmc = self.dtmc_generator.compute_dtmc()
            dtmc.save(dtmc_name, base_dir)
            self.dtmc_generator.save_policy(policy_name, base_dir)
            model = stormpy.parse_explicit_model(dtmc_tra, dtmc_lab)
            unsafe_reachability_prob = stormpy.model_checking_all(model, self.formula[0])
            init_unsafe_prob = unsafe_reachability_prob[state_mapper.INITIAL_STATE]
            print('Prob {}'.format(init_unsafe_prob))
        print('Final Unsafe Prob {}'.format(init_unsafe_prob))

        return dtmc


class LocalRepairer(ABC):
    """
    :return True if the local repair succeded, False otherwise.
    """

    @abstractmethod
    def repair(self, dtmc_generator, probs):
        return False


class DeltaRepairer(LocalRepairer):
    def __init__(self, deltas=[0.2, 0.1, 0.05, 0.01, 0.001, 0.0001]):
        self.deltas = deltas
        self.delta_index = 0
        self.epsilon = 10 ** - 6
        self.unsafe_prob = 1 + self.epsilon

    def repair(self, dtmc_generator, probs):
        prob = probs[state_mapper.INITIAL_STATE]
        if self.unsafe_prob - prob < self.epsilon:
            if self.delta_index + 1 < len(self.deltas):
                self.delta_index += 1
            else:
                return False
        self.unsafe_prob = prob
        repair = False
        for state in range(len(probs)):
            repair |= self.repair_state(dtmc_generator, probs, state, self.deltas[self.delta_index])

        if repair is False and self.delta_index + 1 < len(self.deltas):
            self.delta_index += 1
            # Reset unsafe_prob to prevent that delta_index is incremented two times consecutively
            self.unsafe_prob = 1 + self.epsilon
            return self.repair(dtmc_generator, probs)
        return repair

    def repair_state(self, dtmc_generator, probs, state, delta):
        possible_actions = dtmc_generator.get_possible_actions(state)
        if len(possible_actions) < 2:
            return False
        probs_to_reach_unsafe_state = []
        for action in possible_actions:
            successors = dtmc_generator.get_successor_states(state, action)
            p = 0.0
            for (succ_state, prob) in successors:
                p += prob * probs[succ_state]
            probs_to_reach_unsafe_state.append(p)
        probs_to_reach_unsafe_state = numpy.array(probs_to_reach_unsafe_state)

        sorted_indexes = numpy.argsort(probs_to_reach_unsafe_state)
        action_to_increase = possible_actions[sorted_indexes[0]]
        prob_to_increase = dtmc_generator.policy[state, action_to_increase]
        repaired = False
        for i in range(1, len(sorted_indexes)):
            action_to_reduce = possible_actions[sorted_indexes[i]]
            prob_to_reduce = dtmc_generator.policy[state, action_to_reduce]

            if prob_to_reduce > delta:
                dtmc_generator.policy[state, action_to_reduce] -= delta
                dtmc_generator.policy[state, action_to_increase] += delta
                repaired = True

        return repaired


class TotalRepairer(LocalRepairer):

    def __init__(self):
        self.epsilon = 10 ** - 6
        self.delta = 10 ** - 6
        self.unsafe_prob = 1 + self.epsilon

    def repair(self, dtmc_generator, probs):
        prob = probs[state_mapper.INITIAL_STATE]
        if self.unsafe_prob - prob < self.epsilon:
            return False
        self.unsafe_prob = prob
        repair = False
        for state in range(len(probs)):
            repair |= self.repair_state(dtmc_generator, probs, state, self.delta)

        return repair

    def repair_state(self, dtmc_generator, probs, state, delta):
        possible_actions = dtmc_generator.get_possible_actions(state)
        if len(possible_actions) < 2:
            return False
        probs_to_reach_unsafe_state = []
        for action in possible_actions:
            successors = dtmc_generator.get_successor_states(state, action)
            p = 0.0
            for (succ_state, prob) in successors:
                p += prob * probs[succ_state]
            probs_to_reach_unsafe_state.append(p)
        probs_to_reach_unsafe_state = numpy.array(probs_to_reach_unsafe_state)

        sorted_indexes = numpy.argsort(probs_to_reach_unsafe_state)
        action_to_increase = possible_actions[sorted_indexes[0]]
        repaired = False
        for i in range(1, len(sorted_indexes)):
            action_to_reduce = possible_actions[sorted_indexes[i]]
            prob_to_reduce = dtmc_generator.policy[state, action_to_reduce]

            if prob_to_reduce > delta:
                p = dtmc_generator.policy[state, action_to_reduce] - delta
                dtmc_generator.policy[state, action_to_reduce] = delta
                dtmc_generator.policy[state, action_to_increase] += p
                repaired = True

        return repaired

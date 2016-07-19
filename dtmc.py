import os
import pickle
import numpy

from NDSparseMatrix import NDSparseMatrix
from StateMapper import StateMapper
from utils import Utils

state_mapper = StateMapper()


class DTMC:
    def __init__(self):
        self.dtmc = {}

    def get_probability(self, s_from, s_to):
        return self.dtmc.get((s_from, s_to), 0)

    def add_probability(self, s_from, s_to, value):
        value += self.get_probability(s_from, s_to)
        if value > 0:
            self.dtmc[(s_from, s_to)] = value

    def save(self, file_name='dtmc', base_dir='data/'):
        # Write tra file
        file = open(os.path.join(base_dir, file_name + '.tra'), 'w')
        file.write('dtmc\n')
        for (s_from, s_to), prob in sorted(self.dtmc.items()):
            file.write('{} {} {}\n'.format(s_from, s_to, prob))
        file.close()
        # Write lab file
        file = open(os.path.join(base_dir, file_name + '.lab'), 'w')
        file.write('#DECLARATION\n')
        file.write('init fallen far collided goal\n')
        file.write('#END\n')
        file.write('{} init\n'.format(state_mapper.INITIAL_STATE))
        file.write('{} fallen\n'.format(state_mapper.fallen_state))
        file.write('{} collided\n'.format(state_mapper.self_collided_state))
        file.write('{} far\n'.format(state_mapper.too_far_state))
        file.write('{} goal\n'.format(state_mapper.goal_state))
        file.close()

    def compute_probabilities(self, base_dir='data/repair'):
        import stormpy
        import stormpy.logic
        goal_formula = stormpy.parse_formulas("P=? [ F \"goal\" ]")
        fallen_formula = stormpy.parse_formulas("P=? [ F \"fallen\" ]")
        far_formula = stormpy.parse_formulas("P=? [ F \"far\" ]")
        collided_formula = stormpy.parse_formulas("P=? [ F \"collided\" ]")
        total_formula = stormpy.parse_formulas("P=? [ F (\"far\"  | \"collided\" | \"fallen\")]")
        self.save('temp', base_dir)
        model = stormpy.parse_explicit_model(os.path.join(base_dir, 'temp.tra'),
                                             os.path.join(base_dir, 'temp.lab'))
        goal_prob = stormpy.model_checking(model, goal_formula[0])
        fallen_prob = stormpy.model_checking(model, fallen_formula[0])
        far_prob = stormpy.model_checking(model, far_formula[0])
        collided_prob = stormpy.model_checking(model, collided_formula[0])
        total_prob = stormpy.model_checking(model, total_formula[0])

        return {'goal': goal_prob, 'fallen': fallen_prob, 'far': far_prob,
                'collided': collided_prob, 'total': total_prob}


class DTMCGenerator:
    safe_shutdown_action = Utils.N_ACTIONS

    def __init__(self, ttable_file_path, qtable_file_path, temp=1):
        self.t_table = NDSparseMatrix(ttable_file_path)
        self.trans_prob_dict = self.compute_transition_probabilities_dict()
        with open(qtable_file_path, 'rb') as file:
            qtable = pickle.load(file)
        self.Q = qtable.reshape(len(qtable) // Utils.N_ACTIONS, Utils.N_ACTIONS)
        self.temp = temp
        self.policy = None

    def compute_policy(self):
        n_states, n_actions = self.Q.shape
        policy = numpy.zeros((n_states, n_actions + 1), dtype=float)

        for state in range(n_states):
            if state == state_mapper.too_far_state or state == state_mapper.fallen_state or \
                            state == state_mapper.self_collided_state or state == state_mapper.goal_state:
                policy[state, self.safe_shutdown_action] = 1
                continue
            actions = []
            for action in range(n_actions):
                if self.Q[state, action] != 10 and action != Utils.NULL_ACTION:  # and Q[state, action] >= 0:
                    actions.append((action, self.Q[state, action]))
            if len(actions) > 0:
                if self.temp > 0:
                    values = self.softmax(actions, self.temp)
                else:
                    values = self.deterministic(actions)
                for i in range(len(values)):
                    policy[state, values[i][0]] = values[i][1]
            else:
                policy[state, self.safe_shutdown_action] = 1
        self.policy = policy
        return policy

    def save_policy(self, file_name='policy.pkl', base_dir='data/'):
        with open(os.path.join(base_dir, file_name), 'wb') as file:
            pickle.dump(self.policy, file)

    def load_policy(self, file_name='policy.pkl', base_dir='data/'):
        with open(os.path.join(base_dir, file_name), 'rb') as file:
            self.policy = pickle.load(file)

    def get_possible_actions(self, state):
        n_states, n_actions = self.Q.shape
        possible_actions = []
        for action in range(n_actions + 1):
            if self.policy[state, action] > 0:
                possible_actions.append(action)
        return possible_actions

    def compute_dtmc(self):
        n_states, n_actions = self.Q.shape
        dtmc = DTMC()
        if self.policy is None:
            self.compute_policy()
        for state in range(n_states):
            for action in range(n_actions + 1):
                if self.policy[state, action] == 0:
                    continue
                successors = self.get_successor_states(state, action)
                for (succ_state, prob) in successors:
                    p = self.policy[state, action] * prob
                    dtmc.add_probability(state, succ_state, p)
        return dtmc

    def get_successor_states(self, state, action):
        total = 0
        successors = []
        if state == state_mapper.goal_state:
            return [(state, 1)]
        if action == self.safe_shutdown_action or state == state_mapper.too_far_state \
                or state == state_mapper.fallen_state or state == state_mapper.self_collided_state:
            return [(state_mapper.INITIAL_STATE, 1)]
        successors = self.trans_prob_dict.get((state, action), [])
        if len(successors) == 0:
            successors = [(state_mapper.too_far_state, 1)]
        return successors

    def compute_transition_probabilities_dict(self, transition_counters=None):
        trans_prob_dict = {}

        for key, value in self.t_table.elements.items():
            if value <= 0:
                continue
            new_key = (key[0], key[1])
            v = trans_prob_dict.get(new_key, [])
            v.append((key[2], value))
            trans_prob_dict[new_key] = v

        for (s1, a), successors in trans_prob_dict.items():
            total = 0
            for a, v in successors:
                total += v

            val = int(numpy.exp(- total / 50 * len(successors)) * 100)
            if val > 0:
                total += val
                index = -1
                # Check if the far state is already in the successors vector
                for i, succ in enumerate(successors):
                    if succ[0] == state_mapper.too_far_state:
                        index = i
                if index < 0:
                    successors.append((state_mapper.too_far_state, val))
                else:
                    successors[index] = (successors[index][0], successors[index][1] + val)

            for i, succ in enumerate(successors):
                successors[i] = (succ[0], succ[1] / total)

        return trans_prob_dict

    @staticmethod
    def softmax(items, temp):
        values = []
        for v in items:
            # if v[1] > 500:
            #     value = 500
            # else:
            #     value = v[1]
            value = v[1]
            values.append(numpy.exp(value / temp))
        den = numpy.sum(values)
        for i, value in enumerate(values):
            values[i] = (items[i][0], values[i] / den)
        return values

    @staticmethod
    def deterministic(items):
        values = []
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

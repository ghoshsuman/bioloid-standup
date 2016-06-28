import os
import numpy

from StateMapper import StateMapper
from dtmc import DTMCGenerator

BASE_DIR = 'data/learning-tables/learning-8-june-taclab/'
POLICY_PREFIX = 'sm5-rep'
Q_TABLE_VERSION = 881


def select_action(policy, state, method='argmax'):
    if method == 'argmax':
        return numpy.argmax(policy[state])
    elif method == 'prob':
        return numpy.random.choice(len(policy[state]), p=policy[state])
    else:
        raise ValueError('{} is not a supported method'.format(method))


def main():
    temperature = 2
    ttable_path = os.path.join(BASE_DIR, 't-table.pkl')
    qtable_path = os.path.join(BASE_DIR, 'q-table-{}.pkl'.format(Q_TABLE_VERSION))
    dtmc_generator = DTMCGenerator(ttable_path, qtable_path, temperature)
    dtmc_generator.load_policy(POLICY_PREFIX+'-policy.pkl', BASE_DIR)
    state_mapper = StateMapper()
    state = state_mapper.INITIAL_STATE

    counter = 0
    final_state_counter = [0, 0, 0, 0, 0, 0]
    MAX_ITR = 10000
    c = 0
    while counter < MAX_ITR:
        action = select_action(dtmc_generator.policy, state, 'prob')
        # print('State {} Action {} Prob {}'.format(state, action, dtmc_generator.policy[state][action]))

        if action == 729:
            if state == state_mapper.goal_state:
                print('Goal!')
                final_state_counter[0] += 1
            elif state == state_mapper.fallen_state:
                print('Fallen!')
                final_state_counter[1] += 1
            elif state == state_mapper.too_far_state:
                print('Far!')
                final_state_counter[2] += 1
            elif state == state_mapper.self_collided_state:
                print('Collided!')
                final_state_counter[3] += 1
            else:
                print(state)
                final_state_counter[4] += 1
            state = state_mapper.INITIAL_STATE
            counter += 1
            c = 0
            print('Counter {}'.format(counter))
        else:
            c += 1
            successors = dtmc_generator.get_successor_states(state, action)
            probs = []
            for state, prob in successors:
                probs.append(prob)
            index = numpy.random.choice(len(successors), p=probs)
            if (state == successors[index][0] and successors[index][1] == 1.0
                and dtmc_generator.policy[state, action] == 1.0) or c == 100:
                print('Loops: {} {} {}'.format(action, dtmc_generator.get_possible_actions(state), successors))
                final_state_counter[5] += 1
                state = state_mapper.INITIAL_STATE
            else:
                state = successors[index][0]

    print('--------------------------------------')
    print('Results: {}'.format(final_state_counter))
    print('Goal {}%'.format(final_state_counter[0] / MAX_ITR * 100))
    print('Fallen {}%'.format(final_state_counter[1] / MAX_ITR * 100))
    print('Far {}%'.format(final_state_counter[2] / MAX_ITR * 100))
    print('Collided {}%'.format(final_state_counter[3] / MAX_ITR * 100))
    print('Unknown {}%'.format(final_state_counter[4] / MAX_ITR * 100))
    print('Loops {}%'.format(final_state_counter[5] / MAX_ITR * 100))

if __name__ == '__main__':
    main()
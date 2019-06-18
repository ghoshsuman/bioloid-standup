import scripts
import numpy

from models import NDSparseMatrix
from algs import StateMapper
from utils import Utils


def main():

    t_table = NDSparseMatrix('data/learning-tables/learning-8-june-taclab/t-table.pkl')
    state_mapper = StateMapper()
    current_states = set([state_mapper.INITIAL_STATE])
    counter_main_trace = [0] * len(Utils.standingUpActions)
    counter_diff_actions = [0] * len(Utils.standingUpActions)

    for i, action in enumerate(Utils.standingUpActions):
        action = Utils.vecToInt(action)
        print('Action {} {}'.format(i, action))
        print('Current states {}'.format(current_states))
        successor_states = set()
        for state in current_states:
            for key, value in t_table.elements.items():
                if key[0] != state:
                    continue
                if key[1] == action:
                    successor_states.add(key[2])
                    counter_main_trace[i] += value
                else:
                    counter_diff_actions[i] += value
        current_states = successor_states

    print('values main trace: {}'.format(counter_main_trace))
    print('values different trace: {}'.format(counter_diff_actions))

    print('sum: {} '.format(numpy.array(counter_main_trace) + numpy.array(counter_diff_actions)))

if __name__ == '__main__':
    main()
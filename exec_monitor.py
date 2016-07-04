import os

import numpy
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import shortest_path

import stormpy
import stormpy.logic
from dtmc import DTMCGenerator, DTMC, state_mapper
from model_repair import ModelRepairer, DeltaRepairer
from monitor import Monitor

BASE_DIR = 'data/learning-tables/learning-8-june-taclab/'
Q_TABLE_VERSION = 881
temperature = 2.5


def main():
    print('Repairing {} with temperature {}'.format(BASE_DIR, temperature))
    ttable_path = os.path.join(BASE_DIR, 't-table.pkl')
    qtable_path = os.path.join(BASE_DIR, 'q-table-{}.pkl'.format(Q_TABLE_VERSION))
    dtmc_file_name = 'dtmc-sm{}'.format(temperature)

    policy_file_name = 'policy-sm{}.pkl'.format(temperature)
    dtmc_generator = DTMCGenerator(ttable_path, qtable_path, temperature)
    dtmc_generator.compute_policy()
    dtmc_generator.save_policy(policy_file_name, 'data/repair')
    model_repairer = ModelRepairer(dtmc_generator)

    monitor = Monitor(dtmc_generator, n_trheads=2)

    for i in range(10):
        print('Iteration {}'.format(i))
        policy_file_name = 'policy-sm{}-{}.pkl'.format(temperature, i)
        dtmc = model_repairer.repair(DeltaRepairer(), dtmc_file_name, policy_file_name, 'data/repair')
        dtmc.save(dtmc_file_name + '-rep-{}'.format(i), BASE_DIR)
        dtmc_generator.save_policy(policy_file_name, BASE_DIR)
        # dtmc_generator.load_policy(policy_file_name, BASE_DIR)
        print(dtmc.compute_probabilities())
        monitor.simulate_policy()
        dtmc_generator.trans_prob_dict = dtmc_generator.compute_transition_probabilities_dict()


if __name__ == '__main__':
    main()
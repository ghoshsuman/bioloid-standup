import scripts
import os

import numpy
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import shortest_path

import stormpy
import stormpy.logic
from models.DTMC import DTMCGenerator, DTMC, state_mapper
from algs.model_repair import ModelRepairer, DeltaRepairer, TotalRepairer

BASE_DIR = 'data/learned_tables'
Q_TABLE_VERSION = 66
temperature = 10


def main():
    print('Repairing {} with temperature {}'.format(BASE_DIR, temperature))
    ttable_path = os.path.join(BASE_DIR, 't-table.pkl')
    qtable_path = os.path.join(BASE_DIR, 'q-table-{}.pkl'.format(Q_TABLE_VERSION))
    dtmc_file_name = 'dtmc-sm{}'.format(temperature)
    policy_file_name = 'sm{}-rep-policy.pkl'.format(temperature)
    dtmc_generator = DTMCGenerator(ttable_path, qtable_path, temperature)
    dtmc_generator.compute_policy()
    dtmc_generator.save_policy('sm{}-policy.pkl'.format(temperature), BASE_DIR)
    dtmc = dtmc_generator.compute_dtmc()
    print(dtmc.compute_probabilities())
    # dtmc_generator.load_policy('sm5-policy.pkl', BASE_DIR)
    dtmc_generator.save_policy(policy_file_name, 'data/repair')
    model_repairer = ModelRepairer(dtmc_generator, _lambda=0.001)
    dtmc = model_repairer.repair(TotalRepairer(), dtmc_file_name, policy_file_name, 'data/repair')
    dtmc.save(dtmc_file_name + '-rep', BASE_DIR)
    dtmc_generator.save_policy(policy_file_name, BASE_DIR)

    print(dtmc.compute_probabilities())


if __name__ == '__main__':
    main()

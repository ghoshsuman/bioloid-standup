import pickle
import os
import collections
import numpy

from StateMapper import StateMapper
from utils import Utils
from dtmc import DTMC, DTMCGenerator

safe_shutdown_action = Utils.N_ACTIONS
state_mapper = StateMapper()
BASE_DIR = 'data/learning-tables/learning-8-june-taclab/'
Q_TABLE_VERSION = 881
temperature = 2

def main():
    print('Loading data...')
    ttable_path = os.path.join(BASE_DIR, 't-table.pkl')
    qtable_path = os.path.join(BASE_DIR, 'q-table-{}.pkl'.format(Q_TABLE_VERSION))
    dtmc_generator = DTMCGenerator(ttable_path, qtable_path, temperature)

    print('Computing policy...')

    dtmc_generator.compute_policy()

    print('Computing dtmc...')

    dtmc = dtmc_generator.compute_dtmc()

    print('Saving files...')

    dtmc.save('dtmc-sm{}'.format(temperature), BASE_DIR)

    dtmc_generator.save_policy('sm{}-policy.pkl'.format(temperature), base_dir=BASE_DIR)

    print('Computing deterministic policy...')

    dtmc_generator.temp = 0
    dtmc_generator.compute_policy()

    print('Computing deterministic dtmc...')

    dtmc = dtmc_generator.compute_dtmc()

    print('Saving files...')

    dtmc.save('dtmc-det', BASE_DIR)

    dtmc_generator.save_policy('det-policy.pkl', base_dir=BASE_DIR)




if __name__ == '__main__':
    main()

import os

from models.DTMC import DTMCGenerator
from algs.model_repair import ModelRepairer
from runners.monitor import Monitor

BASE_DIR = 'data/learned_tables'
Q_TABLE_VERSION = 66
temperature = 10

def main():
    print('Repairing {} with temperature {}'.format(BASE_DIR, temperature))
    ttable_path = os.path.join(BASE_DIR, 't-table.pkl')
    qtable_path = os.path.join(BASE_DIR, 'q-table-{}.pkl'.format(Q_TABLE_VERSION))
    dtmc_file_name = 'dtmc-sm{}'.format(temperature)

    policy_file_name = 'policy-sm{}.pkl'.format(temperature)
    dtmc_generator = DTMCGenerator(ttable_path, qtable_path, temperature)
    dtmc_generator.compute_policy()
    dtmc_generator.save_policy(policy_file_name, 'data/repair')
    model_repairer = ModelRepairer(dtmc_generator, _lambda=0.005)

#    monitor = Monitor(dtmc_generator, model_repairer, BASE_DIR, n_trheads=3, n_episodes=200)
    monitor = Monitor(dtmc_generator, model_repairer, BASE_DIR, n_trheads=3, n_episodes=10)
    # monitor.load(9)

    # dtmc_generator.load_policy('policy-sm5-9.pkl', 'data/repair')
    # dtmc_generator.t_table.load('data/repair/t-table-9.pkl')
    # dtmc_generator.trans_prob_dict = dtmc_generator.compute_transition_probabilities_dict()

    for i in range(0, 20):
        print('Iteration {}'.format(i))
        monitor.iteration()


if __name__ == '__main__':
    main()

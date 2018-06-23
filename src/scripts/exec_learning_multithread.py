import pickle
import vrep
from runners.SimulationMaster import SimulationMaster
from pybrain.rl.explorers import EpsilonGreedyExplorer


def main():
    try:
        n_threads = 40
        initial_port = 8000
        q_table_version = 0
        batch_size = 10
        # explorer = EpsilonGreedyBoltzmannExplorer(0.2, 5, 0.998)
        explorer = EpsilonGreedyExplorer(0.15, 0.999)
        master = SimulationMaster(n_threads, initial_port, q_table_version, batch_size, explorer=explorer)

        master.run()

    except (KeyboardInterrupt, SystemExit):
        with open('data/learning-tables/standing-up-q.pkl', 'wb') as handle:
            pickle.dump(master.save_q_table(), handle)
        master.save_t_table()
        master.save_t_table()
        vrep.simxFinish(-1)


if __name__ == '__main__':
    main()
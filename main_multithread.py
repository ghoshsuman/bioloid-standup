import pickle

import subprocess

from SimulationMaster import SimulationMaster


def main():
    try:
        n_threads = 8
        initial_port = 8000

        master = SimulationMaster(n_threads, initial_port)

        master.run()

    except (KeyboardInterrupt, SystemExit):
        with open('data/learning-tables/standing-up-q.pkl', 'wb') as handle:
            pickle.dump(master.save_q_table(), handle)
        master.save_t_table()
        master.save_t_table()


if __name__ == '__main__':
    main()
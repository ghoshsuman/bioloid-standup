from SimulationMaster import SimulationMaster


def main():
    n_threads = 2
    initial_port = 19998
    master = SimulationMaster(n_threads, initial_port)

    master.run()


if __name__ == '__main__':
    main()
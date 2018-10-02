import pickle

from algs import StateDiscretizer


def main():

    with open('data/state-space/state-space-all.pkl', 'rb') as file:
        state_space = pickle.load(file)
    sd = StateDiscretizer()
    discretized_state_space = set()
    for state in state_space:
        x = sd.discretize(state)
        print(state)
        print(x)
        print('------------------------')

        discretized_state_space.add(tuple(x))

    print(len(state_space))
    print(len(discretized_state_space))

    state_space = []

    for s in discretized_state_space:
        state_space.append(list(s))

    with open('data/state-space/discretized-state-space.pkl', 'wb') as file:
        pickle.dump(state_space, file)


if __name__ == '__main__':
    main()
import pickle

import utils
from StateNormalizer import StateNormalizer
from pybrain_components import StandingUpSimulator
from utils import Utils


def main():

    states_space = []

    client_id = Utils.connectToVREP()

    # First add the initial state
    env = StandingUpSimulator(client_id)
    states_space.append(env.bioloid.read_state())

    for i in range(len(Utils.standingUpActions)):
        for j in range(1):
            with open('data/state-space/state-space-t{}-{}.pkl'.format(i, j), 'rb') as file:
                data = pickle.load(file)
                for k in range(len(data)):
                    if not data[k]['is-fallen'] and not data[k]['self-collided']:
                        states_space.append(data[k]['state_vector'])

    print(len(data) * (len(Utils.standingUpActions) + 1))
    print(len(states_space))
    with open('data/state-space/state-space-all-0.pkl', 'wb') as file:
        pickle.dump(states_space, file)

    Utils.endVREP()


if __name__ == '__main__':
    main()
import pickle

import utils
from StateNormalizer import StateNormalizer
from pybrain_components import StandingUpEnvironment
from utils import Utils


def main():

    states_space = []



    # First add the initial state
    client_id = Utils.connectToVREP()
    env = StandingUpEnvironment(client_id)
    states_space.append(env.bioloid.read_state())
    Utils.endVREP()

    for i in range(len(Utils.standingUpActions)):
        for j in range(10):
            with open('data/state-space/state-space-t{}-{}.pkl'.format(i, j), 'rb') as file:
                data = pickle.load(file)
                for k in range(len(data)):
                    if not data[k]['is-fallen'] and not data[k]['self-collided']:
                        states_space.append(data[k]['state_vector'])

    print(len(data) * (len(Utils.standingUpActions)) * 10)
    print(len(states_space))
    with open('data/state-space/state-space-all-0.pkl', 'wb') as file:
        pickle.dump(states_space, file)

    sn = StateNormalizer()
    print(sn)
    for sv in states_space:
        sn.update_bounds(sv)
    print(sn)

if __name__ == '__main__':
    main()
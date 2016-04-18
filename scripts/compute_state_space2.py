import pickle

from pybrain_components import StandingUpSimulator
from scripts.utils import Utils


def main():
    client_id = Utils.connectToVREP()
    data = []

    environment = StandingUpSimulator(client_id, 'data/models/bioloid-t{}.ttt'.format(1))

    for t in range(14):
        print('data/models/bioloid-t{}.ttt'.format(t))
        step_data = []
        for action in range(Utils.N_ACTIONS):
            print('Action {}'.format(action))
            if action == Utils.NULL_ACTION:  # Avoid the null action
                continue

            for i in range(10):
                environment = StandingUpSimulator(client_id, 'data/models/bioloid-t{}.ttt'.format(t))
                environment.performAction(Utils.intToVec(action))
                bioloid = environment.bioloid
                row = {'action' : action, 'state_vector': bioloid.read_state(), 'is-fallen': bioloid.isFallen(),
                       'self-collided': bioloid.isSelfCollided(), 'trajectory-step': t}
                step_data.append(row)
        with open('data/state-space-t{}.pkl'.format(t), 'wb') as file:
            pickle.dump(step_data, file)
        data += step_data

    Utils.endVREP()
    with open('data/state-space-all.pkl', 'wb') as file:
        pickle.dump(data, file)

if __name__ == '__main__':
    main()
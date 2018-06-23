import pickle

from algs.learning.pybrain_components import StandingUpEnvironment
from utils import Utils


def main():
    client_id = Utils.connectToVREP()
    data = []

    environment = StandingUpEnvironment(client_id)

    for i in range(10):
        print('Iteration {}'.format(i))

        for t, targetAction in enumerate(Utils.standingUpActions):
            print('Starting generation step {}'.format(t))
            step_data = []
            for action in range(Utils.N_ACTIONS):
                print('Action {}'.format(action))
                if action == Utils.NULL_ACTION:  # Avoid the null action
                    continue
                environment.reset()

                for j in range(t):
                    environment.performAction(Utils.standingUpActions[j])

                # environment = StandingUpSimulator(client_id, 'data/models/bioloid-t{}.ttt'.format(t))
                environment.performAction(Utils.intToVec(action))
                bioloid = environment.bioloid
                row = {'action': action, 'state_vector': bioloid.read_state(), 'is-fallen': bioloid.is_fallen(),
                       'self-collided': bioloid.is_self_collided(), 'trajectory-step': t}
                step_data.append(row)
            with open('data/state-space-t{}-{}.pkl'.format(t, i), 'wb') as file:
                pickle.dump(step_data, file)
            data += step_data

    Utils.endVREP()
    with open('data/state-space-all.pkl', 'wb') as file:
        pickle.dump(data, file)

if __name__ == '__main__':
    main()
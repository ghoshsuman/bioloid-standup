import pickle

from StateNormalizer import StateNormalizer


def main():


    state_space = []
    with open('data/state-space/state-space-all-0.pkl', 'rb') as file:
        state_space = pickle.load(file)

    sn = StateNormalizer()
    for state in state_space:
        sn.update_bounds(state)
        print(sn)

    sn.save_bounds('data/norm_bounds.pkl')

if __name__ == '__main__':
    main()
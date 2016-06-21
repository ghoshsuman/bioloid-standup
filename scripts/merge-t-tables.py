import os
import re

import pickle


def main():

    base_dir = 'data/learning-tables/learning-21-june-taclab'

    ttable = {}

    if os.path.exists('data/learning-tables/t-table.pkl'):
        with open('data/learning-tables/t-table.pkl', 'rb') as file:
            ttable = pickle.load(file)

    for f in os.listdir(base_dir):
        if re.match('t-table-', f):
            with open(os.path.join(base_dir, f), 'rb') as handle:
                t = pickle.load(handle)
                print(len(t))
                for key, value in t.items():
                    v = ttable.get(key, 0)
                    ttable[key] = v + value

    with open(os.path.join(base_dir, 't-table.pkl'), 'wb') as file:
        pickle.dump(ttable, file)
    print('ttable len: {}'.format(len(ttable)))

    counter = 0
    for key, value in ttable.items():
        counter += value

    print('counter {}'.format(counter))

if __name__ == '__main__':
    main()
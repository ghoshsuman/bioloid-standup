import scripts
import os
import re

import pickle

from NDSparseMatrix import NDSparseMatrix


def main():

    base_dir = 'data/learning-tables/learning-4-july-blade21'

    ttable = NDSparseMatrix()

    # if os.path.exists('data/learning-tables/t-table.pkl'):
    #     ttable = NDSparseMatrix('data/learning-tables/t-table.pkl')

    for f in os.listdir(base_dir):
        if re.match('t-table-', f):
            t = NDSparseMatrix(os.path.join(base_dir, f))
            print(len(t))
            ttable.add(t)

    ttable.save(os.path.join(base_dir, 't-table.pkl'))
    print('ttable len: {}'.format(len(ttable)))

    counter = 0
    for key, value in ttable.elements.items():
        counter += value

    print('counter {}'.format(counter))

if __name__ == '__main__':
    main()
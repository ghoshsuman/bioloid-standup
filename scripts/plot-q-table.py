import pickle

import pylab
import matplotlib.pyplot as plt

from utils import Utils


def main():

    pylab.gray()
    pylab.ion()

    with open('data/learning-tables/q-table-19.pkl', 'rb') as file:
        qtable = pickle.load(file)
        M = qtable.reshape(len(qtable) // Utils.N_ACTIONS, Utils.N_ACTIONS)
        print(M.shape)
        # pylab.imshow(M)



        x_pos = []
        y_pos = []

        x_ten = []
        y_ten = []

        x_10_0 = []
        y_10_0 = []

        x_neg = []
        y_neg = []

        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if M[i, j] > 10:
                    # print('{} {}'.format(i, j))
                    # print(M[i, j])
                    x_pos.append(j)
                    y_pos.append(i)
                elif M[i, j] == 10:
                    x_ten.append(j)
                    y_ten.append(i)
                elif M[i, j] >= 0:
                    x_10_0.append(j)
                    y_10_0.append(i)
                else:
                    x_neg.append(j)
                    y_neg.append(i)

        print(len(x_pos))
        print(len(x_ten))
        print(len(x_10_0))
        print(len(x_neg))
        plt.scatter(x_pos, y_pos, color='b', s=0.5)
        plt.title('Q-values > 10')
        plt.axis((0, M.shape[1], 0, M.shape[0]))
        plt.show()
        plt.figure()
        plt.scatter(x_neg, y_neg, color='b', s=0.5)
        plt.title('Q-values < 0')
        plt.axis((0, M.shape[1], 0, M.shape[0]))
        plt.show()
        # plt.figure()
        #plt.scatter(x_ten, y_ten, color='b')
        # plt.title('Q-values = 10 (not visited)')
        # plt.axis((0, M.shape[1], 0, M.shape[0]))
        # plt.show()
        plt.figure()
        plt.scatter(x_10_0, y_10_0, color='b', s=0.5)
        plt.title('0 <= Q-values < 10')
        plt.axis((0, M.shape[1], 0, M.shape[0]))
        plt.show()

        # pylab.pcolor(M)
        # pylab.draw()

        plt.pause(100)


if __name__ == '__main__':
    main()
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.decomposition import PCA

from pybrain_components import StandingUpSimulator

N_ACTIONS = 729

def main():
    X = []
    with open('state-space-filtered.pkl', 'rb') as file:
        X = pickle.load(file)
    y = np.zeros(len(X), dtype=int)

    trajectories = [
        [0, 1093, 1825, 2581, 4036, 4765, 4738, 5979, 6196, 7627, 7654, 8383, 8437, 8956],
        [0, 1093, 1042, 1771, 2767, 4768, 5494, 7383, 5684, 6925, 7657, 7516, 8434, 8947],
        [0, 1093, 3116, 3065, 4039, 3253, 5494, 4711, 5684, 6925, 7657, 7516, 8434, 8947],
        [0, 1093, 1822, 2581, 3284, 3257, 3982, 7383, 5684, 6925, 7657, 7516, 9112, 8947]
    ]

    colors = ['b', 'r', 'g', 'y', 'm', 'c', 'k', '#A13535', '#A19D24', '#328931', '#7B2B7B', '#2B787B', '#96FFED']

    for i, t in enumerate(trajectories):
        for p in t:
            y[p] = i + 1
    pca = PCA(2)
    X_pca = pca.fit_transform(X)

    # Plot and save results

    for i in range(len(trajectories)):
        plt.figure()

        plt.title("Projection by PCA")
        plt.xlabel("1st principal component")
        plt.ylabel("2nd component")

        plt.plot(X_pca[:, 0], X_pca[:, 1], "bo")

        indexes = np.zeros(len(X), dtype=int) == 1
        for p in trajectories[i]:
            indexes[p] = True
        plt.plot(X_pca[indexes, 0], X_pca[indexes, 1], colors[i + 1]+'o')

        plt.subplots_adjust(0.02, 0.10, 0.98, 0.94, 0.04, 0.35)

        # plt.show()
        plt.savefig('pca{}.png'.format(i), bbox_inches='tight')
        plt.close()

    for p in trajectories[0]:
        indexes[p] = True

    # Draw and store all the point divided according to the points that generated them

    with open('state-space-normalized.pkl', 'rb') as file:
        X = pickle.load(file)
    pca = PCA()
    X_pca = pca.fit_transform(X)
    plt.figure()
    for i in range(0, len(X), 729):
        index = i // N_ACTIONS

        if index > 0 and index % 8 == 0:
            plt.savefig('state_space_colour_1.png', bbox_inches='tight')
            plt.show()
            plt.figure()
        plt.subplot(2, 4, index % 8 + 1, aspect='equal')
        plt.axis([-0.4, 0.4, -0.3, 0.3])

        plt.scatter(X_pca[i:i + N_ACTIONS, 0], X_pca[i: i + N_ACTIONS, 1],  color=colors[index])

    plt.savefig('state_space_colour_2.png', bbox_inches='tight')
    plt.show()

    for i in range(100):
        print(StandingUpSimulator.intToVec(i))

if __name__ == '__main__':
    main()
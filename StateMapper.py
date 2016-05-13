import pickle

from scipy.spatial import KDTree
from scipy.spatial.distance import euclidean

from StateDiscretizer import StateDiscretizer


class StateMapper:

    GOAL_STATE = 757
    TOO_FAR_THRESHOLD = 0.2
    GOAL_THRESHOLD = 0.5

    def __init__(self, bioloid=None):
        self.sd = StateDiscretizer()
        self.state_space = self.kdtree = None
        self.fallen_state = self.self_collided_state = self.too_far_state = self.goal_state = -1
        self.load_state_space()
        self.bioloid = bioloid

    def map(self, raw_vector_state):
        discretized_state = self.sd.discretize(raw_vector_state)
        dist, index = self.kdtree.query(discretized_state)
        current_state = index
        goal_distance = self.get_goal_distance(discretized_state)
        if goal_distance < self.GOAL_THRESHOLD:
            current_state = self.goal_state

        if dist > self.TOO_FAR_THRESHOLD:  # Check if the actual state is too far from the one it was mapped
            current_state = self.too_far_state

        if self.bioloid is not None:
            if self.bioloid.is_fallen():  # Check if the bioloid is fallen
                current_state = self.fallen_state
            elif self.bioloid.is_self_collided():  # Check if the bioloid is self-collided
                current_state = self.self_collided_state
        print('state: {} dist: {}'.format(current_state, dist))
        return current_state

    def get_goal_distance(self, state):
        goal = self.get_goal_state_vector()
        goal_distance = euclidean(goal, state)
        # print('Goal distance: ' + str(goal_distance))
        return goal_distance

    def get_goal_state_vector(self):
        return self.state_space[self.GOAL_STATE]

    def load_state_space(self, filepath='data/state-space/discretized-state-space.pkl'):
        with open(filepath, 'rb') as handle:
            self.state_space = pickle.load(handle)
        n = len(self.state_space)
        self.fallen_state = n
        self.self_collided_state = n + 1
        self.too_far_state = n + 2
        self.goal_state = n + 3
        self.kdtree = KDTree(self.state_space)

    def get_state_space_size(self):
        return len(self.state_space) + 4

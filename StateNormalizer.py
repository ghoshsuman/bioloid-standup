import os
import numpy
import pickle


class StateNormalizer:

    BOUND_MARGIN = 0.1
    DEFAULT_PATH = 'data/norm_bounds.pkl'

    def __init__(self, filepath = DEFAULT_PATH):
        self.isInitialised = False

        if os.path.isfile(filepath):
            self.load_bound(filepath)


        self.isStable = True

    def normalise(self, state_vector):
        normalized_state = (state_vector - self.lowerbound) / (self.upperbound - self.lowerbound)
        # Check that all value are in [0, 1] range
        assert (normalized_state >= 0).all() and (normalized_state <= 1).all(), \
            'Current normalization bounded exceeded!'

        return normalized_state

    def update_bounds(self, state_vector):
        if not self.isInitialised:
            self.lowerbound = state_vector - self.BOUND_MARGIN
            self.upperbound = state_vector + self.BOUND_MARGIN
            self.isInitialised = True
        else:
            old_lowerbound = self.lowerbound
            old_upperbound = self.upperbound
            self.lowerbound = numpy.minimum(self.lowerbound, state_vector)
            self.upperbound = numpy.maximum(self.upperbound, state_vector)
            if not (old_lowerbound - self.lowerbound == 0).all() or not (old_upperbound - self.upperbound == 0).all():
                self.isStable = False
                self.lowerbound = numpy.minimum(self.lowerbound, state_vector - self.BOUND_MARGIN)
                self.upperbound = numpy.maximum(self.upperbound, state_vector + self.BOUND_MARGIN)
            return (state_vector - self.lowerbound) / (self.upperbound - self.lowerbound)

    def save_bounds(self, filepath = DEFAULT_PATH):
        data = { 'lowerbound': self.lowerbound, 'upperbound': self.upperbound}
        with open(filepath, 'wb') as file:
            pickle.dump(data, file)

    def load_bound(self, filepath = DEFAULT_PATH):
        with open(filepath, 'rb') as file:
            data = pickle.load(file)
            self.lowerbound = data['lowerbound']
            self.upperbound = data['upperbound']
            self.isInitialised = True

    def __str__(self):
        s = 'lowerbound:\n'+self.lowerbound.__str__() + '\n'
        s += 'upperbound:\n'+self.upperbound.__str__() + '\n'
        return s

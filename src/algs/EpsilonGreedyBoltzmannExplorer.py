from scipy import random, array

from pybrain.rl.explorers.discrete.discrete import DiscreteExplorer
from pybrain.utilities import drawGibbs


class EpsilonGreedyBoltzmannExplorer(DiscreteExplorer):

    def __init__(self, epsilon=0.2, tau=2.5, epsilon_decay = 0.9999, tau_decay = 1):
        DiscreteExplorer.__init__(self)
        self.epsilon = epsilon
        self.tau = tau
        self.epsilon_decay = epsilon_decay
        self.tau_decay = tau_decay
        self._state = None

    def activate(self, state, action):
        """ The super class ignores the state and simply passes the
            action through the module. implement _forwardImplementation()
            in subclasses.
        """
        self._state = state
        return DiscreteExplorer.activate(self, state, action)

    def _forwardImplementation(self, inbuf, outbuf):
        """ Draws a random number between 0 and 1. If the number is less
            than epsilon, the action is selected according to the boltzmann exploration strategy. If it is equal or
            larger than epsilon, the greedy action is returned.
        """
        assert self.module
        if random.random() < self.epsilon:
            values = self.module.getActionValues(self._state)
            action = drawGibbs(values, self.tau)
            outbuf[:] = array([action])
        else:
            outbuf[:] = inbuf

    def apply_decay(self):
        self.tau *= self.tau_decay
        self.epsilon *= self.epsilon_decay

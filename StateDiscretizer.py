import pickle


class StateDiscretizer:

    def __init__(self, n1=50, n2=20, n3=12):
        self.delta = None
        self.lowerbound = self.upperbound = None
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.load_bounds()
        self.compute_delta()

    def discretize(self, state):
        d = state - self.lowerbound
        x = (d / self.delta).astype(int)
        for i in range(len(x)):
            if d[i] - self.delta[i] * x[i] >= self.delta[i] / 2:
                x[i] += 1
        return self.lowerbound + x * self.delta

    def load_bounds(self, file_path='data/norm_bounds.pkl'):
         with open(file_path, 'rb') as file:
            data = pickle.load(file)
            self.lowerbound = data['lowerbound']
            self.upperbound = data['upperbound']

    def compute_delta(self):
        self.delta = (self.upperbound - self.lowerbound)
        self.delta[0:3] /= self.n1
        self.delta[3:7] /= self.n2
        self.delta[7:] /= self.n3

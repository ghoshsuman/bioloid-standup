import pickle


class NDSparseMatrix:

    DEFAULT_FILE_NAME = 't-table.pkl'

    def __init__(self):
        self.elements = {}

    def setValue(self, tuple, value):
        self.elements[tuple] = value

    def getValue(self, tuple):
        try:
          value = self.elements[tuple]
        except KeyError:
          value = 0
        return value

    def incrementValue(self, tuple):
        value = self.getValue(tuple)
        self.setValue(tuple, value + 1)

    def save(self, filename=DEFAULT_FILE_NAME):
        with open(filename, 'wb') as file:
            pickle.dump(self.elements, file)

    def load(self, filename=DEFAULT_FILE_NAME):
        with open(filename, 'rb') as file:
            self.elements = pickle.load(file)
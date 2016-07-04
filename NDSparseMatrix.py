import os
import pickle


class NDSparseMatrix:

    DEFAULT_FILE_NAME = 't-table.pkl'

    def __init__(self, file_path=None):
        self.elements = {}
        if file_path is not None:
            self.load(file_path)

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
        if os.path.isfile(filename):
            with open(filename, 'rb') as file:
                self.elements = pickle.load(file)
        else:
            self.elements = {}

    def add(self, sparse_matrix):
        for key, value in sparse_matrix.elements.items():
            self.setValue(key, self.getValue(key) + value)

    def reset(self):
        del self.elements
        self.elements = {}

    def __len__(self):
        return len(self.elements)

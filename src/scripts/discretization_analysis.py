import scripts
import pickle

import xlsxwriter

from StateDiscretizer import StateDiscretizer
from StateNormalizer import StateNormalizer


def main():

    with open('data/state-space/state-space-all.pkl', 'rb') as file:
        state_space = pickle.load(file)

    print(len(state_space))

    workbook = xlsxwriter.Workbook('data/reports/discr_size.xls')
    worksheet = workbook.add_worksheet('Sheet 1')
    row = 0
    for n1 in [10, 20, 50, 100, 500]:
        for n2 in [10, 12, 15, 20, 30, 50]:
            for n3 in [5, 10, 12, 13, 14, 15, 20]:
                sd = StateDiscretizer(n1, n2, n3)
                discretized_state_space = set()
                for state in state_space:
                    x = sd.discretize(state)
                    discretized_state_space.add(tuple(x))
                worksheet.write(row, 0, n1)
                worksheet.write(row, 1, n2)
                worksheet.write(row, 2, n3)
                worksheet.write(row, 3, len(discretized_state_space))
                counter_diff = []
                for i in range(len(state_space[0])):
                    counter_diff.append({})
                for state in discretized_state_space:
                    for i in range(len(state)):
                        count = counter_diff[i].get(state[i], 0)
                        counter_diff[i][state[i]] = count + 1

                for i in range(len(counter_diff)):
                    worksheet.write(row, i + 5, len(counter_diff[i]))
                row += 1
    workbook.close()

if __name__ == '__main__':
    main()
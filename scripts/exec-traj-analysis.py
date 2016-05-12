import pickle

import numpy
import xlsxwriter
from scipy.spatial.distance import euclidean

from StateNormalizer import StateNormalizer
from pybrain_components import StandingUpEnvironment, StandingUpTask
from utils import Utils


def main():

    client_id = Utils.connectToVREP()
    environment = StandingUpEnvironment(client_id)
    task = StandingUpTask(environment)
    state_vector_length = len(environment.bioloid.read_state())
    delta = task.state_mapper.sd.delta
    print(delta)

    print(numpy.sqrt(numpy.sum(delta ** 2)) / 2)

    n = int(input('Number of iterations: '))

    workbook = xlsxwriter.Workbook('data/reports/trajectory-trials.xls')
    worksheets = []
    for i in range(len(Utils.standingUpActions)):
        worksheets.append(workbook.add_worksheet('t'+str(i+1)))

    for i in range(n):
        print('Iteration ' + str(i + 1))
        print('Initial State: ')
        # task.getObservation()
        print(task.state_mapper.sd.discretize(environment.getSensors()))
        print(task.getObservation()[0])

        for j, action in enumerate(Utils.standingUpActions):
            environment.performAction(action)
            state_vector = environment.getSensors()
            discretized_state = task.state_mapper.sd.discretize(state_vector)
            for k, s in enumerate(discretized_state):
                worksheets[j].write(i, k, s)
            state_n = task.update_current_state()

            state_distance = euclidean(task.state_mapper.state_space[state_n], discretized_state)
            goal_distance = task.state_mapper.get_goal_distance(discretized_state)

            print(discretized_state)
            print(task.state_mapper.state_space[state_n])
            print('---------------------')

            worksheets[j].write(i, state_vector_length + 1, state_n)
            worksheets[j].write(i, state_vector_length + 2, state_distance)
            worksheets[j].write(i, state_vector_length + 3, goal_distance)

        environment.reset()

    res_worksheet = workbook.add_worksheet('Results')

    row = 0
    for i in range(len(Utils.standingUpActions)):
        sheet_name = 't'+str(i+1)
        res_worksheet.write(row, 0, sheet_name)
        res_worksheet.write(row, 1,   'mean')
        res_worksheet.write(row+1, 1, 'var')

        for j in range(state_vector_length):
            col_name = chr(ord('A') + j)
            data_range = sheet_name+'.'+col_name+'1:'+col_name+str(n)
            # TODO: check why range is made lowercase :/
            res_worksheet.write_formula(row, 2 + j, '=AVERAGE('+data_range+')')
            res_worksheet.write_formula(row + 1, 2 + j, '=VAR.P('+data_range+')')
        row += 2

    workbook.close()

if __name__ == '__main__':
    main()

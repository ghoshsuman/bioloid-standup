import xlsxwriter
from scipy.spatial.distance import euclidean

from StateNormalizer import StateNormalizer
from pybrain_components import StandingUpSimulator, StandingUpTask
from utils import Utils


def main():

    client_id = Utils.connectToVREP()
    environment = StandingUpSimulator(client_id)
    task = StandingUpTask(environment)
    state_vector_length = len(environment.bioloid.read_state())
    stateNormalizer = StateNormalizer()
    n = int(input('Number of iterations: '))

    workbook = xlsxwriter.Workbook('data/trajectory-trials.xls')
    worksheets = []
    for i in range(len(Utils.standingUpActions)):
        worksheets.append(workbook.add_worksheet('t'+str(i+1)))

    for i in range(n):
        print('Iteration ' + str(i + 1))
        for j, action in enumerate(Utils.standingUpActions):
            environment.performAction(action)
            state_vector = environment.getSensors()
            for k, s in enumerate(state_vector):
                worksheets[j].write(i, k, s)

            state_n = task.getObservation()
            goal_distance = euclidean(task.GOAL_STATE, state_vector)
            worksheets[j].write(i, state_vector_length + 3, state_n[0])
            worksheets[j].write(i, state_vector_length + 4, goal_distance)

        environment.reset()

    res_worksheet = workbook.add_worksheet('Results')
    stateNormalizer.save_bounds()

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
            res_worksheet.write(row, 2 + j, '=AVERAGE('+data_range+')')
            res_worksheet.write(row + 1, 2 + j, '=VAR.P('+data_range+')')
        row += 2
        '''
        for i in range(len(Utils.standingUpActions)):
        worksheets[i].write(n + 2, 1,   'mean')
        worksheets[i].write(n + 3, 1, 'var')
        for j in range(state_vector_length):
            col_name = chr(ord('A') + j)
            data_range = col_name+'1:'+col_name+str(n)
            # print(data_range)
            worksheets[i].write(n + 2, 2 + j, '=AVERAGE('+data_range+')')
            worksheets[i].write(n + 3, 2 + j, '=VAR.P('+data_range+')')
        '''

    workbook.close()

if __name__ == '__main__':
    main()

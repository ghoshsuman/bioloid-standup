import xlsxwriter

import vrep
from utils import Utils
from pybrain_components import StandingUpSimulator


def write_row(worksheet, row, values):
    for i, v in enumerate(values):
                worksheet.write(row, i, v)


def main():
    client_id = Utils.connectToVREP()

    env = StandingUpSimulator(client_id)
    workbook = xlsxwriter.Workbook('data/reports/teleport-test.xls')
    worksheet = workbook.add_worksheet('Sheet 1')

    env.performAction(Utils.NULL_ACTION_VEC)
    trajectory_states = [env.bioloid.read_full_state()]
    row = 0
    write_row(worksheet, row, env.bioloid.read_state())
    for action in Utils.standingUpActions:
        env.performAction(action)
        row += 1
        write_row(worksheet, row, env.bioloid.read_state())
        trajectory_states.append(env.bioloid.read_full_state())

    row += 2
    for state in trajectory_states:
        env.bioloid.set_full_state(state)
        env.performAction(Utils.NULL_ACTION_VEC)
        row += 1
        write_row(worksheet, row, env.bioloid.read_state())

    Utils.endVREP()
    workbook.close()

if __name__ == '__main__':
    main()

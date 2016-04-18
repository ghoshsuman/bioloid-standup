import pickle
import vrep
import pylab
from pybrain_components import StandingUpTask, StandingUpSimulator
from scripts.utils import Utils


def main():

    client_id = Utils.connectToVREP()
    environment = StandingUpSimulator(client_id)
    task = StandingUpTask(environment)
    counter = 0
    while counter < 5:
        for action in Utils.standingUpActions:
            observation = task.getObservation()
            print(task.current_sensors)
            a = Utils.vecToInt(action)
            task.performAction(a)
            task.getReward()
        environment.reset()
        if environment.norm.isStable:
            counter += 1
        else:
            counter = 0
            environment.norm.isStable = True
        print('stable: ' + str(counter))
        print('norm bounds: ')
        print(environment.norm.lowerbound)
        print(environment.norm.upperbound)

    Utils.endVREP()

if __name__ == '__main__':
    main()
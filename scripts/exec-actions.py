from pybrain_components import StandingUpSimulator, StandingUpTask
from utils import Utils


def main():

    client_id = Utils.connectToVREP()
    environment = StandingUpSimulator(client_id)
    task = StandingUpTask(environment)



    while True:
        action_str = input("Insert next action: ")
        action = [int(x) for x in action_str.split(' ')]

        observation = task.getObservation()
        print(task.current_sensors)
        a = Utils.vecToInt(action)
        task.performAction(a)
        task.getReward()
        print('self-collided: '+ str(environment.bioloid.isFallen()))
        print('is-fallen: '+ str(environment.bioloid.isFallen()))

    environment.reset()


if __name__ == '__main__':
    main()

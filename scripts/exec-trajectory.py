from pybrain_components import StandingUpSimulator, StandingUpTask
from scripts.utils import Utils


def main():

    client_id = Utils.connectToVREP()
    environment = StandingUpSimulator(client_id)
    task = StandingUpTask(environment)

    for action in Utils.standingUpActions:
        observation = task.getObservation()
        print(task.current_sensors)
        a = Utils.vecToInt(action)
        task.performAction(a)
        task.getReward()
        print('self-collided: '+ str(environment.bioloid.isFallen()))
        print('is-fallen: '+ str(environment.bioloid.isFallen()))
        input("Press Enter to continue...")
    environment.reset()


if __name__ == '__main__':
    main()

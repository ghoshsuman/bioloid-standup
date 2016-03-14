import pickle
import vrep
import pylab
from bioloid import Bioloid
from pybrain.rl.agents import LearningAgent
from pybrain.rl.experiments import Experiment
from pybrain.rl.learners import ActionValueTable, Q
from pybrain_components import Simulator, RaiseArmsTask


def main():
    vrep.simxFinish(-1)  # just in case, close all opened connections
    client_id = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)  # Connect to V-REP

    if client_id < 0:
        print('Failed connecting to remote API server')
        return -1

    print('Connected to remote API server')

    pylab.gray()
    pylab.ion()

    # Define RL elements
    environment = Simulator(client_id)

    controller = ActionValueTable(260, 27)
    controller.initialize(1.)

    learner = Q()
    agent = LearningAgent(controller, learner)

    task = RaiseArmsTask(environment)

    experiment = Experiment(task, agent)

    i = 0
    try:
        while True:
            i += 1
            print('Iteration n° ' + str(i))
            experiment.doInteractions(50)
            agent.learn()
            agent.reset()

            pylab.pcolor(controller.params.reshape(260, 27))
            pylab.draw()
            # pylab.pause(0.1)
    except (KeyboardInterrupt, SystemExit):
        with open('rl.pkl', 'wb') as handle:
            pickle.dump(controller.params, handle)


    '''
    vrep.simxLoadScene(client_id, '/home/simone/Dropbox/Vuotto Thesis/bioloid.ttt', 0, vrep.simx_opmode_blocking)

    # enable the synchronous mode on the client:
    vrep.simxSynchronous(client_id, True)

    # start the simulation:
    vrep.simxStartSimulation(client_id, vrep.simx_opmode_blocking)

    bioloid = Bioloid(client_id)

    # Now step a few times:
    startTime = time.time()
    while time.time()-startTime < 30:

        arm_values = [random.randint(-1, 1) for i in range(3)]
        leg_values = [random.randint(-1, 1) for i in range(3)]
        # bioloid.move_arms(arm_values)
        # bioloid.move_legs(leg_values)

        for i in range(5):
            vrep.simxSynchronousTrigger(client_id)
        time.sleep(0.01)
        # input('Press <enter> key to step the simulation!')

    # stop the simulation:
    vrep.simxStopSimulation(client_id,vrep.simx_opmode_blocking)

    vrep.simxCloseScene(client_id, vrep.simx_opmode_blocking)
    # Now close the connection to V-REP:
    '''
    vrep.simxFinish(client_id)


if __name__ == '__main__':
    main()

import threading
import traceback

import vrep
from pybrain_components import StandingUpSimulator, StandingUpTask
from utils import Utils


class Simulation(threading.Thread):

    BATCH_SIZE = 1

    def __init__(self, master, port):
        threading.Thread.__init__(self)
        self.daemon = True
        self.master = master
        self.client_id = -1
        self.environment = None
        self.task = None
        self.port = port
        self.trace = []

    def run(self):
        try:
            # connect to V-REP server
            self.client_id = Utils.connectToVREP(self.port)
            self.environment = StandingUpSimulator(self.client_id)
            self.task = StandingUpTask(self.environment)

            while True:
                # wait for barrier

                while not self.task.isFinished():
                    self.perform_step()

                self.task.reset()

                self.master.barrier.wait()  # wait for the end of other simulations
                self.master.barrier.wait()  # wait for q matrix update

        except Exception as e:
            print('[Simulation %s] %s' % (self.port, e.args[0]))
            traceback.print_exc()

        finally:
            # disconnect with V-REP server
            vrep.simxFinish(self.client_id)

    def perform_step(self):
        observation = self.task.getObservation()
        action = self.master.get_action(observation)
        self.task.performAction(action)
        reward = self.task.getReward()
        self.trace.append([observation, action, reward])


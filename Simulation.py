import os
import threading
import traceback

import subprocess

import time

import sys

import vrep
from pybrain_components import StandingUpEnvironment, StandingUpTask
from utils import Utils


class Simulation(threading.Thread):
    DEFAULT_BATCH_SIZE = 20

    def __init__(self, master, port, batch_size=DEFAULT_BATCH_SIZE):
        threading.Thread.__init__(self)
        self.daemon = True
        self.master = master
        self.client_id = -1
        self.environment = None
        self.task = None
        self.port = port
        self.batch_size = batch_size
        self.current_trace = []
        self.traces = []
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def stopped(self):
        return self._stop.isSet()

    def run(self):
        try:
            vrep_path = os.path.abspath('V-REP_PRO_EDU_V3_3_0_64_Linux/')
            proc = subprocess.Popen(
                'cd {} &&  xvfb-run --auto-servernum --server-num=1 ./vrep.sh -h -gREMOTEAPISERVERSERVICE_{}_FALSE_TRUE'.format(
                    vrep_path, self.port), shell=True)
            time.sleep(5)
            # connect to V-REP server
            self.client_id = Utils.connectToVREP(self.port)
            self.environment = StandingUpEnvironment(self.client_id)
            self.task = StandingUpTask(self.environment, 'data/learning-tables/log_{}.log'.format(self.port))
            self.load_t_table()

            while not self.stopped():
                # wait for barrier
                for _ in range(self.batch_size):
                    self.current_trace = []
                    while not self.task.isFinished():
                        self.perform_step()
                    self.traces.append(self.current_trace)
                    self.task.reset()

                self.master.barrier.wait()  # wait for the end of other simulations
                self.master.barrier.wait()  # wait for q matrix update

        except (RuntimeError, AssertionError) as e:
            self.master.logger.error('[Simulation %s] %s' % (self.port, e.args[0]))
            traceback.print_exc()
            vrep.simxFinish(self.client_id)  # disconnect with V-REP server
            self.current_trace = []
            self.master.failed_simulations.append(self)
            self.master.barrier.wait()
            self.master.barrier.wait()
        except:
            self.master.logger.error('[Simulation %s] Unexpected error: %s' % (self.port, sys.exc_info()[0]))
            traceback.print_exc()
            self.current_trace = []
            self.master.failed_simulations.append(self)
            self.master.barrier.wait()
            self.master.barrier.wait()
        finally:
            proc.kill()
            self.task.logger.info('[Simulation %s] Ended' % self.port)

    def perform_step(self):
        observation = self.task.getObservation()
        action = self.master.get_action(observation)
        self.task.performAction(action)
        reward = self.task.getReward()
        self.current_trace.append([observation, action, reward])

    def save_t_table(self):
        if self.task is not None:
            self.task.t_table.save('data/learning-tables/t-table-{}.pkl'.format(self.port))

    def load_t_table(self):
        path = 'data/learning-tables/t-table-{}.pkl'.format(self.port)
        if self.task is not None and os.path.exists(path):
            self.task.t_table.load(path)

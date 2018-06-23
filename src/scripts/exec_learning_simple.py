import pickle

import numpy
import vrep
from pybrain.rl.agents import LearningAgent
from pybrain.rl.experiments import EpisodicExperiment
from pybrain.rl.explorers import EpsilonGreedyExplorer
from pybrain.rl.learners import Q
from models import MyActionValueTable
from models.pybrain import StandingUpEnvironment, StandingUpTask
from utils import Utils


def main():
    client_id = Utils.connectToVREP()

    # Define RL elements
    environment = StandingUpEnvironment(client_id)
    task = StandingUpTask(environment)
    controller = MyActionValueTable()
    learner = Q(0.5, 0.9)
    learner.explorer = EpsilonGreedyExplorer(0.15, 1)  # EpsilonGreedyBoltzmannExplorer()
    agent = LearningAgent(controller, learner)
    experiment = EpisodicExperiment(task, agent)

    controller.initialize(agent)

    i = 0
    try:
        while True:
            i += 1
            print('Episode ' + str(i))
            experiment.doEpisodes()
            agent.learn()
            agent.reset()
            print('mean: '+str(numpy.mean(controller.params)))
            print('max: '+str(numpy.max(controller.params)))
            print('min: '+str(numpy.min(controller.params)))

            if i % 500 == 0:  # Save q-table every 500 episodes
                print('Save q-table')
                controller.save()
                task.t_table.save()

    except (KeyboardInterrupt, SystemExit):
        with open('../data/standing-up-q.pkl', 'wb') as handle:
            pickle.dump(controller.params, handle)
        task.t_table.save()
        controller.save()

    vrep.simxFinish(client_id)


if __name__ == '__main__':
    main()

# Bioloid Standup

This project was part of the Simone Vuotto's master thesis, and published in the International Symposium of Laveraging Applicatioons of Formal Methods.
If you use the code, please acknowledge it by citing:

    @inproceedings{leofante2016combining,
        title={Combining static and runtime methods to achieve safe standing-up for humanoid robots},
        author={Leofante, Francesco and Vuotto, Simone and {\'A}brah{\'a}m, Erika and Tacchella, Armando and Jansen, Nils},
        booktitle={International Symposium on Leveraging Applications of Formal Methods},
        pages={496--514},
        year={2016},
        organization={Springer}
    }

## Learning

There are two scripts to start the learning experiemnt: `scripts/exec_learning_simple.py` and `scripts/exec_learning_multithread.py`. 
The former shows the basic skeleton of the learning process and allows the user to
see the robot behaviour directly in the simulator, while the latter is a multithread 
optimization designed to work on a multicore server without user interface.
They also use a slightly different learning algorithm, with different parameters.

In order to run the scripts, you first have to activate the virtualenv and set 
the PYTHONPATH env variable:

    source venv/bin/activate
    cd src
    export PYTHONPATH=$(pwd) 
    
In the case of `scripts/exec_learning_simple.py`, you also need to run a vrep instance first:
   
    ../V-REP_PRO_EDU_V3_5_0_Linux/vrep.sh &
    
Finally you can simply execute the script:

    python scripts/exec_learning_simple.py
    
or 

    python scripts/exec_learning_multithread.py
    

The two scripts save log files and regularly dump the q-table and t-table learned during the process in the `data` directory.

### How does it learn?

As explained in the paper, the q-table is initialized with a scripted sequence of actions, and the state space is built around 
the states reached during this sequence. In this way the agent already know a good strategy and the state space is limited 
to the interested region only. The agent is then left to explore for new alternative strategies.

The episode terminates when the robot reached one of the following states:

 - **Fallen State**: a collision with the floor is detected
 - **Collided State**: a collision with itself is detected
 - **Too Far State**: the agent went exploring too far from the defined state space
 - **Goal Reached**: the agent reached the goal, *i.e.* it reached the standup position
 
 The `scripts/exec_learning_simple.py` script use the classic Q-learning algorithm with EpsilonGreedy exploration strategy.
 
 In order to improve performance and reduce instability, `scripts/exec_learning_multithread.py` adopts a batched version of the 
 Q-learning algorithm, that
 updates the q-table only after the `n_threads` parallel simulations have executed `batch_size` episodes.
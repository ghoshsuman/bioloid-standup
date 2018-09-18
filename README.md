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
 
 ## UPDATES IN THE ARCHITECTURE : FEB-JULY 2018 , EMARO+ program
 
 The follwing updates and further work was carried out as a Group projrct in the cousework of the EMARO+ masters program. 
 
 Apart from adjusting various scripts and path-variables, major update was done for adjusting the current scripts as per the latest STORM(A modern model checker for probabilistic systems) and its set of python bindings- Stormpy, used fundamentaaly in Probabilistic Model checking and Model repair/update.
 
 A separate folder `/data/src` was created to store logs,learned and repaired tables for run time monitoring and model-repair. Pre-learned tables (q-table and t-table) were taken for all further experiments performed.
 
  ## Model-Repair and Run-time Monitoring
  
  There are two scripts for this : `scripts/exec_model_repair.py` and `scripts/exec_monitor.py`. To run them, after initializing the virtual environment as explained earlier and execute the scripts:
  
        `python scripts/exec_model_repair.py`
 
 and 
 
        `python scripts/exec_monitor.py`
        
  First script does the probabilistic model checking, i.e. calculates the 'unsafe' probabilities to reach terminal states instead of 'goal'. It further tries to 'reapair' the model i.e. decresing the unsafe probabilities below a certain 'lambda' or stops when its not getting further reduced below specified thresholds. The path to the 'learned' q-table ,t-table and the temperature for repair is needed to be specified in this script. This saves the computed policy file( as .pkl) and repaired policy file. 
  
 The 'exec_monitor' script performs Run-time Montioring of the Standing-Up task. Before running this, it is required ot initialize V-rep as eariler The path to the 'learned' q-table ,t-table ,temperature for repair as well as the number of iteration, threads and episodes per iteration are needed to be specified in this script. After repairing the model intially, an iteraiton of the Standing up task is run with repaired policy. After every iteration the 'unsafe' probabilities are again computed and the model-repair is performed, saving the new policy, q-table and t-table corresponding to the repaired model.
 
 The repaired-poicies can be 'visualised' by executing the `exec-policy.py` script after running the V-rep environment.
  
## Faults Injection: 

 Three types of faults were injected as follows:

 - **Certain action(s) lead to 'collided state' **: 
    This can be set in `/src/runners/monitor.py`
 - **Actuator(s) fail to work**: These can be modelled in `/src/models/Bioloid.py`. A pair of joints can also be made a faulty in `/src/models/pybrain/StandingUpEnvironmnet`.
 - **Sensor(s) fail to work**:  These can be modelled in `/src/models/Bioloid.py`.
 
 

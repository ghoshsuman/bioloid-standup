# Scripts

- **combine-state-space-data.py** Takes the files generate by _compute_state_space2.py_ in `data/state-space/` and
    combine them in a single file, removing states in which the robot is fallen or collided;
- **compute-bounds.py** Takes in input the file generate by _combine-state-space-data.py_ and compute the lower and 
    upper bounds of each component with `StateNormalizer`, saving them in `data/norm_bounds.pkl`;
- **compute_probabilities.py** Computes probabilities to reach unsafe states of a given policy, used for debug only;
- ~~**compute_state_space.py**~~ First version of the script to compute the state space starting from the scripted action
    sequence, replaced by _compute_state_space2.py_;
- **compute_state_space2.py** Current version of state space generation script;
- **data_visualization.py** Perform PCA and component analysis on generated state space, used for debug only;
- **discretization_analysis.py**
- **discretize_state_space.py**
- **dist_calc.py**
- **exec_actions.py**
- **exec-policy.py**
- **exec-policy-analysis.py**
- **exec-policy-model.py**
- **exec-traj-analysis.py**
- **exec-trajectory.py**
- **exec-trajectory2.py**
- **exec_model_repair.py**
- **exec_monitor.py**
- **find-different-path.py**
- **generate-dtmc.py**
- **main.py**
- **main_multithread.py**
- **merge-t-tables.py**
- **plot-q-table.py**
- **t-table-analysis.py**
- **teleport.py**
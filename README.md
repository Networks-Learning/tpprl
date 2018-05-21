# Deep Reinforcement Learning of Marked Temporal Point Processes

This is the code produced as part of the paper "Deep Reinforcement Learning of Marked Temporal Point Processes".

## Packages needed

 - `tensorflow 1.8.0`
 - `numpy 1.14.3`
 - `pandas 0.22.0`
 - `pip install decorated_options`

Needed for Smart Broadcaster experiments, and the `RedQueen` baseline:

 - `pip install git+https://github.com/Networks-Learning/RedQueen.git@master#egg=redqueen`

Needed for the `Karimi` baseline:

 - `pip install git+https://github.com/Networks-Learning/broadcast_ref.git@master#egg=broadcast_ref`

We obtained the parameters from the item difficulties via personal correspondence with the authors of MEMORIZE.

## Scripts

Running experiments:

### `sbatch/exp_run.py`

This script is used for running Smart Broadcasting experiments on a SLURM cluster using the job scripts `sbatch/r_2_job.sh` and `sbatch/top_k_job.sh`.

The job scripts assumes that there is a conda environment with the name `tf-cpu` and that the code resides in `${HOME}/prog/work/broadcast-rl/` folder. These details can be edited easily to match with any host's configuration via the scripts themselves.

### `rq_compare-algo.py`

This script is used for running an experiment for one-user in the Smart Broadcasting setup.


### `compare_results-algo.py`

Reads output produced by `rq_compare-algo.py` and compares it against baselines and saves the results in a CSV file ready for analysis/plotting.

### `memorize_exp.py`

TODO: Scripts for running MEMORIZE training.

## Reproducing figures

TODO

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

 - `sbatch/exp_run.py`: Running Smart Broadcasting experiments on a SLURM cluster.
 - `rq_compare-algo.py`: Running one experiment for Smart Broadcastering.
 - `compare_results-algo.py`: Reads output produced by `rq_compare-algo.py` and compares it with baselines.
 - TODO: Scripts for running MEMORIZE training.

## Reproducing plots

TODO

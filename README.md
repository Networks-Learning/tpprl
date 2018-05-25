# Deep Reinforcement Learning of Marked Temporal Point Processes

This is the code produced as part of the paper _Deep Reinforcement Learning of Marked Temporal Point Processes_ 

> "Deep Reinforcement Learning of Marked Temporal Point Processes"
> Utkarsh Upadhyay, Abir De, Manuel Gomez-Rodriguez. [arXiv:1805.09360](https://arxiv.org/abs/1805.09360), 2018.

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

## Experiment execution

Running experiments:

### `sbatch/exp_run.py`

This script is used for running Smart Broadcasting experiments on a SLURM cluster using the job scripts `sbatch/r_2_job.sh` and `sbatch/top_k_job.sh`.

The job scripts assumes that there is a `conda` environment with the name `tf-cpu` and that the code resides in `${HOME}/prog/work/broadcast-rl/` folder. These details can be edited easily to match with any host's configuration via the scripts themselves.

### `train-broadcasting.py`

This script is used for running an experiment for one-user in the Smart Broadcasting setup.

It can be executed as:

    export USER_IDX=218   # Which user to train the model for.
    mkdir -p output-smart-broadcasting/
    python train-broadcastring ./data/twitter_data.dill ${USER_IDX} ./output-smart-broadcasting \
      --reward r_2_reward --q 100.0 --algo-feed --save-every 100 \
      --no-merge-sinks

If the `${USER_IDX}` variable is looped over the list of indexes in `./data/r_2.csv` (or `./data/top_k.csv`), then we can reproduce the trained network for all 100 users used for experiments in the paper.

The output will be saved in files `./output-dir/train-save-user_idx-${USER_IDX}`.

### `analyze-broadcasting.py`

Reads output produced by `train-broadcasting.py` and compares it against baselines and saves the results in a CSV file ready for analysis/plotting.

    python analyze-broadcasting.py ./data/twitter_data.dill ./output-smart-broadcasting/ broadcasting-analysis.csv  \
            --no-merge-sinks --algo-feed


### `train-teaching.py`

This script is used for running the Spaced Repetition experiment:

    mkdir -p output-spaced-repetition;
    python train-teaching.py ./data/initial-difficulty.csv 0.049 0.0052 ./output-spaced-repetition \
        --q 0.0001 --q-entropy 0.005


If the training gets stuck on 0 reward, then training an initial version `--with-recall-probs` as reward instead of binary reward, and/or increasing the `batch_size` may help.


## Reproducing figures

### Smart broadcasting

Running:

    python plot-smart-broadcasting.py broadcasting-analysis.csv

will reproduce the plots in the paper in the `./output-plots` folder.

### Spaced repetition

Running:

    python analyze-spaced-repetition.py ./data/initial_difficulty.csv 0.049 0.0052 ./output-spaced-repetition

will re-produce the plots in the paper in the `./output-plots` folder.

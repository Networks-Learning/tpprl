#!/bin/bash

set -eo pipefail

source conda.sh
source activate tf-cpu

cd ${HOME}/prog/work/tpprl/;

in_file=$1
user_idx=$2
out_dir=$3
N=$4
q=$5
until=$6
epochs=$7
K=$8
save_every=$9
algo_feed=${10}
algo_approx=${11}
with_zero_wt=${12}

python train-broadcasting.py "${in_file}" "${user_idx}" "${out_dir}" --N $N --q $q --reward top_k_reward --until ${until} --epochs ${epochs} --restore --only-cpu --reward-top-k $K --save-every $save_every $algo_feed $algo_approx $with_zero_wt --no-merge-sinks

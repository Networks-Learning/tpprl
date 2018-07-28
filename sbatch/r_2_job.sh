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
save_every=$8
algo_feed=$9
algo_approx=${10}
with_zero_wt=${11}

python train-broadcasting.py "${in_file}" "${user_idx}" "${out_dir}" --N $N --q $q --reward r_2_reward --until ${until} --epochs ${epochs} --restore --only-cpu --save-every $save_every $algo_feed $algo_approx ${with_zero_wt}

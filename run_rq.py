#!/usr/bin/env python
import click
import os
import pandas as pd
# from tpprl.utils import _now


@click.command()
@click.argument('hostname')
@click.argument('gpu_device_idx', type=int)
@click.argument('in_csv', type=click.Path(exists=True))
@click.option('--dry/--no-dry', 'dry_run', help='Dry run.', default=True)
@click.option('--reps', 'reps', help='How many times to iterate over the executions?', default=1)
# @click.option('--use-gpus/--no-user-gpus', 'use_gpus', help='Whether to use the GPUs during training.', default=True)
def run(hostname, gpu_device_idx, in_csv, dry_run, reps):
    """Execute the tasks for this HOSTNAME and GPU_DEVICE_IDX in the IN_CSV."""
    # os.putenv('CUDA_VISIBLE_DEVICES', str(gpu_device_idx))
    df = pd.read_csv(in_csv)

    for rep in range(reps):
        print('\n\n####################################################\n\n')
        print('Rep  {} is starting ...'.format(rep))
        print('\n\n####################################################\n\n')

        for _, row in df[(df['host'] == hostname) & (df['gpu'] == gpu_device_idx)].iterrows():
            print('Running for idx: {}'.format(row.idx))
            cmd = (# f'CUDA_VISIBLE_DEVICES={gpu_device_idx} ' +
                   f'time ./rq-compare.py "{row.inp_file}" {row.idx} "{row.out_dir}" --N {row.N} --q {row.q} ' +
                   f' --reward {row.reward} --reward-top-k {row.K} --until {row.until} ' +
                   f' --epochs {row.epochs} --allow-growth')

            print("Running:", cmd)
            if not dry_run:
                os.system(cmd)

            print('\n\n****************************************************\n\n')

    print('\n\n####################################################\n\n')


if __name__ == '__main__':
    run()

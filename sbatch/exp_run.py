#!/usr/bin/env python

import pandas as pd
import click
import os

OUTPUT_DIR = "/tmp"


@click.command()
@click.argument('in_csv')
@click.option('--user-data-file', help='The .dill file containing all users\' data.', default='../data/twitter_data.dill', show_default=True)
@click.option('--dry/--no-dry', help='Dry run.', default=True, show_default=True)
@click.option('--epochs', help='Epochs.', default=25, show_default=True)
@click.option('--reward', 'reward_kind', help='Which reward to use [r_2_reward, top_k_reward].', default='r_2_reward', show_default=True)
@click.option('--output-dir', 'output_dir', help='Where to save the output', default=OUTPUT_DIR, show_default=True)
@click.option('--K', 'k', help='K in top-k loss.', default=1, show_default=True)
@click.option('--mem', 'mem', help='How much memory will each job need (MB).', default=10000, show_default=True)
@click.option('--until', 'until', help='Until which step to run the experiments.', default=1000, show_default=True)
@click.option('--save-every', 'save_every', help='How many epochs to save output at.', default=5, show_default=True)
@click.option('--q', 'q', help='Which q value to use. Negative values imply using the value in the CSV file.', default=-1.0, show_default=True)
@click.option('--N', 'N', help='What should be the average number of posts in a window?', default=300, show_default=True)
@click.option('--algo-feed/--no-algo-feed', 'algo_feed', help='Use algorithmic feeds.', default=False, show_default=True)
@click.option('--algo-approx/--no-algo-approx', 'with_approx_rewards', help='Whether to use exact or approximate rewards for algorithmic feeds.', default=True, show_default=True)
@click.option('--with-zero-wt/--no-with-zero-wt', 'with_zero_wt', help='Force wt to be zero.', default=False, show_default=True)
def run(in_csv, user_data_file, dry, epochs, k, mem, reward_kind, output_dir, until, q, save_every,
        algo_feed, with_approx_rewards, N, with_zero_wt):
    """Read parameters from in_csv, ignore the host/gpu information, and execute them on using sbatch."""
    os.makedirs(os.path.join(output_dir, 'stdout'), exist_ok=True)
    df = pd.read_csv(in_csv)

    if algo_feed:
        algo_feed_str = '--algo-feed'
    else:
        algo_feed_str = '--no-algo-feed'

    if with_approx_rewards:
        algo_approx_str = '--algo-approx'
    else:
        algo_approx_str = '--no-algo-approx'

    if with_zero_wt:
        with_zero_wt_str = '--with-zero-wt'
    else:
        with_zero_wt_str = '--no-with-zero-wt'

    user_data_file_abs = os.path.abspath(user_data_file)

    for row_idx, row in df.iterrows():
        stdout_file = f'{output_dir}/stdout/user_idx-{row.idx}.%j.out'

        if q < 0:
            q = row.q

        if reward_kind == 'top_k_reward':
            cmd = (f'sbatch -c 2 --mem={mem} -o "{stdout_file}" ' +
                   f'./top_k_job.sh {user_data_file_abs} {row.idx} "{output_dir}" ' +
                   f'{N} {q} {until} {epochs} {k} {save_every} ' +
                   f'{algo_feed_str} {algo_approx_str} {with_zero_wt_str}')
        elif reward_kind == 'r_2_reward':
            cmd = (f'sbatch -c 2 --mem={mem} -o "{stdout_file}" ' +
                   f'./r_2_job.sh {user_data_file_abs} {row.idx} "{output_dir}" ' +
                   f'{N} {q} {until} {epochs} {save_every} ' +
                   f'{algo_feed_str} {algo_approx_str} {with_zero_wt_str}')
        else:
            raise ValueError("Unknown reward: {}".format(reward_kind))

        if dry:
            print(cmd)
        else:
            os.system(cmd)


if __name__ == '__main__':
    run()

#!/usr/bin/env python

import click
import pandas as pd
import os
import glob
import warnings
import re
import numpy as np
import sys
import dill
import multiprocessing as MP
import redqueen.utils as RU
import redqueen.opt_runs as OR
# import tpprl.read_data_utils as RDU
import tpprl.exp_broadcaster as EB
import natsort


u_data_dir_tmpl = 'u_data-user_idx-{}'
save_dir_tmpl = 'train-save-user_idx-{}'
save_dir_regex = re.compile(r'train-save-user_idx-(\d*)')
save_dir_glob = r'train-save-user_idx-*'
user_data = None


def read_user_data(user_data_file):
    global user_data
    with open(user_data_file, 'rb') as f:
        user_data = dill.load(f)


def worker_user(params):
    (user_idx, output_dir, test_batches, RQ_cap_adjust) = params

    save_dir = os.path.join(output_dir, save_dir_tmpl.format(user_idx))

    with open(os.path.join(save_dir, 'user_opt_dict.dill'), 'rb') as f:
        user_opt_dict = dill.load(f)

    one_user_data = user_data[user_idx]
    ret = {
        'user_idx': user_idx,
        'user_id': one_user_data['user_id'],
        'num_other_posts': one_user_data['num_other_posts'],
        'num_own_posts': one_user_data['num_user_events'],
        'num_followees': one_user_data['num_followees'],
        'duration': one_user_data['duration'],
        'num_followers': user_opt_dict['num_followers'],
        'N': user_opt_dict['N'],
        'reward_kind': user_opt_dict['trainer_opts_dict']['reward_kind'],
    }

    window_start, eval_sim_opts = EB.make_real_data_batch_sim_opts(
        one_user_data=one_user_data,
        N=user_opt_dict['N'],
        seed=-1,
        is_test=True
    )
    ret['window_start'] = window_start
    ret['window_end'] = eval_sim_opts.end_time

    # The file names are of the form `*/tpprl.ckpt-<num>.meta`.
    # Hence, the number is interpreted as negative.
    # So to extract the last checkpoint, we do a sort by real-values and
    # pick the most negative value.
    # Also, we drop the `.meta` suffix.
    all_chpt_file = glob.glob(os.path.join(save_dir, '*.meta'))
    last_chpt_file = natsort.realsorted(all_chpt_file)[0][:-5]

    ret['chpt_file'] = last_chpt_file

    rl_b_dict = EB.rl_b_dict_from_chpt(
        # '/NL/crowdjudged/work/rl-broadcast/r_2-sim-opt-fix/train-save-user_idx-218/tpprl.ckpt-898',
        last_chpt_file,
        one_user_data=one_user_data,
        window_start=window_start,
        user_opt_dict=user_opt_dict
    )

    if 'q' in user_opt_dict:
        q = user_opt_dict['q']
    else:
        warnings.warn('Setting q manually.')
        reward_kind = user_opt_dict['trainer_opts_dict']['reward_kind']
        if reward_kind == 'r_2_reward':
            q = 100.0
        elif reward_kind == 'top_k_reward':
            q = 1.0

    init_seed = 865
    rl_dfs = []
    for idx in range(test_batches):
        mgr = EB.get_real_data_mgr_chpt_np(
            rl_b_dict,
            t_min=window_start,
            batch_sim_opt=eval_sim_opts,
            seed=init_seed + idx
        )
        mgr.run_dynamic()
        rl_dfs.append(mgr.get_state().get_dataframe())

    num_tweets = [RU.num_tweets_of(df, broadcaster_id=eval_sim_opts.src_id)
                  for df in rl_dfs]
    capacity_cap, capacity_std = np.mean(num_tweets), np.std(num_tweets)

    ret['capacity'] = capacity_cap
    ret['capacity_std'] = capacity_std

    # Figure out what 'q' to use for RQ to get the same number of tweets.
    # Removing '1' because RQ systematically tweets more.
    q_RQ = RU.sweep_q(eval_sim_opts, capacity_cap=capacity_cap - RQ_cap_adjust,
                      verbose=False, q_init=q, parallel=False)
    ret['q_RQ'] = q_RQ

    # Run RedQueen.
    RQ_dfs = []
    for idx in range(test_batches):
        mgr = eval_sim_opts.update({'q': q_RQ}).create_manager_with_opt(seed=init_seed + idx)
        # mgr = eval_sim_opts.update({}).create_manager_with_opt(seed=init_seed + idx)
        mgr.state.time = window_start
        mgr.run_dynamic()
        RQ_dfs.append(mgr.get_state().get_dataframe())

    # Run Poisson.
    poisson_dfs = []
    for idx in range(test_batches):
        mgr = eval_sim_opts.create_manager_with_poisson(
            rate=capacity_cap / (eval_sim_opts.end_time - window_start),
            seed=init_seed + idx
        )
        mgr.state.time = window_start
        mgr.run_dynamic()
        poisson_dfs.append(mgr.get_state().get_dataframe())

    # Running Karimi
    K = 1

    T = eval_sim_opts.end_time - window_start
    num_segments = 10
    seg_len = T / num_segments
    wall_mgr = eval_sim_opts.create_manager_for_wall()
    wall_mgr.run_dynamic()
    wall_df = wall_mgr.state.get_dataframe()

    ret['num_segments'] = num_segments
    ret['num_wall_tweets'] = wall_df.event_id.nunique()

    seg_idx = ((wall_df.t.values - window_start) / T * num_segments).astype(int)
    intensity_df = (wall_df.groupby(['sink_id', pd.Series(seg_idx, name='segment')]).size() / (T / num_segments)).reset_index(name='intensity')
    wall_intensities_df = intensity_df.pivot_table(values='intensity', index='sink_id', columns='segment').fillna(0)
    for seg_idx in range(num_segments):
        if seg_idx not in wall_intensities_df.columns:
            wall_intensities_df[seg_idx] = 0.0
    wall_intensities = wall_intensities_df[list(range(num_segments))].values

    # This is the single-threaded version
    karimi_dfs = []
    params = (init_seed, capacity_cap, num_segments, eval_sim_opts, wall_intensities, None)
    op = OR.worker_kdd(params, verbose=False, Ks=[K], window_start=window_start)

    for idx in range(test_batches):
        piecewise_const_mgr = eval_sim_opts.create_manager_with_piecewise_const(
            seed=init_seed + idx,
            change_times=window_start + np.arange(num_segments) * seg_len,
            rates=op['kdd_opt_{}'.format(K)] / seg_len
        )
        piecewise_const_mgr.state.time = window_start
        piecewise_const_mgr.run_dynamic()
        df = piecewise_const_mgr.state.get_dataframe()
        karimi_dfs.append(df)

    # Calculating metrics
    metric_name = 'num_tweets'
    for type, dfs in [('RL', rl_dfs), ('RQ', RQ_dfs), ('poisson', poisson_dfs), ('karimi', karimi_dfs)]:
        metric = [RU.num_tweets_of(df, broadcaster_id=eval_sim_opts.src_id) for df in dfs]
        ret[type + '_' + metric_name + '_mean'], ret[type + '_' + metric_name + '_std'] = (np.mean(metric), np.std(metric))

    metric_name = 'top_k'
    for type, dfs in [('RL', rl_dfs), ('RQ', RQ_dfs), ('poisson', poisson_dfs), ('karimi', karimi_dfs)]:
        metric = [RU.time_in_top_k(df, K=K, sim_opts=eval_sim_opts) for df in dfs]
        ret[type + '_' + metric_name + '_mean'], ret[type + '_' + metric_name + '_std'] = (np.mean(metric), np.std(metric))

    metric_name = 'avg_rank'
    for type, dfs in [('RL', rl_dfs), ('RQ', RQ_dfs), ('poisson', poisson_dfs), ('karimi', karimi_dfs)]:
        metric = [RU.average_rank(df, sim_opts=eval_sim_opts) for df in dfs]
        ret[type + '_' + metric_name + '_mean'], ret[type + '_' + metric_name + '_std'] = (np.mean(metric), np.std(metric))

    metric_name = 'r_2'
    for type, dfs in [('RL', rl_dfs), ('RQ', RQ_dfs), ('poisson', poisson_dfs), ('karimi', karimi_dfs)]:
        metric = [RU.int_r_2(df, sim_opts=eval_sim_opts) for df in dfs]
        ret[type + '_' + metric_name + '_mean'], ret[type + '_' + metric_name + '_std'] = (np.mean(metric), np.std(metric))

    return ret


@click.command()
@click.argument('tweeters_data_file')
@click.argument('output_dir')
@click.argument('save_csv')
@click.option('--batches', 'batches', help='How large should the test batches be.', default=64)
@click.option('--force/--no-force', 'force', help='Whether to overwrite the output-csv file.', default=False)
@click.option('--RQ-cap-adjust', 'RQ_cap_adjust', help='How much to compensate the tweets of RedQueen by.', default=1.0)
def run(output_dir, save_csv, tweeters_data_file, batches, force, RQ_cap_adjust):
    """Read all OUTPUT_DIR and compile the results for all users and save them in SAVE_CSV.
    The user data is read from IN_DATA_FILE and `batches` number of batches are executed.
    """

    if os.path.exists(save_csv) and not force:
        print('File {} exists and --force was not supplied.'.format(save_csv))
        sys.exit(-1)

    read_user_data(tweeters_data_file)

    save_dirs = glob.glob(os.path.join(output_dir, save_dir_glob))
    user_idxes = [int(save_dir_regex.search(x)[1]) for x in save_dirs]

    all_params = [(x, output_dir, batches, RQ_cap_adjust) for x in user_idxes]
    with MP.Pool() as pool:
        save_dict = pool.map(worker_user, all_params)

    save_df = pd.DataFrame(save_dict)
    save_df.to_csv(save_csv, index=False)


if __name__ == '__main__':
    run()

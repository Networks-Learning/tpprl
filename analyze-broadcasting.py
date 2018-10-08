#!/usr/bin/env python

import warnings
# This removes the annoying warning from h5py
warnings.simplefilter(action='ignore', category=FutureWarning)

import click
import pandas as pd
import os
import glob
import numpy as np
import sys
import dill
import multiprocessing as MP
import redqueen.utils as RU
import redqueen.opt_runs as OR
import redqueen.opt_model as OM
import tpprl.read_data_utils as RDU
import tpprl.exp_broadcaster as EB
import tpprl.exp_sampler as ES
import natsort
from collections import defaultdict
# import decorated_options as Deco


# This ensures that cvxopt uses only one thread for computation.
# Otherwise, the computation leaks onto other cores and causes multiple
# context switches, slowing down the overall calculation.
os.environ['OMP_NUM_THREADS'] = "1"

user_data = None

MAX_EVENTS = 8000
MAX_ITERS = 500
REWARD_STEPS = 1000


def read_user_data(user_data_file):
    """Reading the user_data into a global file so that we can share it across
    processes, without having to keep multiple copies in memory."""
    global user_data
    with open(user_data_file, 'rb') as f:
        user_data = dill.load(f)


def worker_user(params):
    """
    Worker for calculating metrics for one user.
    Uses only one core.
    """
    (user_idx, output_dir, test_batches, RQ_cap_adjust,
     for_epoch, verbose, only_rl, algo_feed, algo_frac, merge_sinks,
     set_wt_zero) = params

    save_dir = os.path.join(output_dir, EB.SAVE_DIR_TMPL.format(user_idx))

    if verbose:
        print('Working on user_idx: {}'.format(user_idx))

    with open(os.path.join(save_dir, 'user_opt_dict.dill'), 'rb') as f:
        user_opt_dict = dill.load(f)

    one_user_data = user_data[user_idx]
    if merge_sinks:
        if verbose:
            print('Merged sinks!')
        one_user_data = RDU.merge_sinks(one_user_data)

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
        'for_epoch': for_epoch,
        'num_batches': test_batches,
        'algo_feed': algo_feed,
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

    if for_epoch < 0:
        for_epoch = EB.find_largest_chpt(save_dir, verbose=verbose)
        if for_epoch is None:
            return ret

    chosen_chpt_file = os.path.join(save_dir, EB.TPPRL_CHPT_TMPL.format(for_epoch))

    if verbose:
        print('chosen_chpt_file = ', chosen_chpt_file)

    ret['chpt_file'] = chosen_chpt_file
    if not os.path.exists(chosen_chpt_file + '.meta'):
        ret['error'] = 'File Not Found: {}.'.format(chosen_chpt_file)
        return ret

    rl_b_dict = EB.rl_b_dict_from_chpt(
        # '/NL/crowdjudged/work/rl-broadcast/r_2-sim-opt-fix/train-save-user_idx-218/tpprl.ckpt-898',
        chosen_chpt_file,
        one_user_data=one_user_data,
        window_start=window_start,
        user_opt_dict=user_opt_dict
    )

    if set_wt_zero:
        rl_b_dict['wt'] = 0

    sink_ids = one_user_data['sim_opts'].sink_ids
    if algo_feed:
        algo_c = user_opt_dict['algo_c']
        lifetimes = defaultdict(lambda: (eval_sim_opts.end_time - window_start) * algo_frac)
        # algo_feed_args = ES.make_prefs(sink_ids, src_ids, seed=algo_feed_seed,
        #                                src_lifetime_dict=lifetimes)
        algo_feed_args = ES.make_freq_prefs(
            one_user_data=one_user_data,
            sink_ids=sink_ids,
            src_lifetime_dict=lifetimes
        )

        rl_b_dict['algo_feed'] = algo_feed
        rl_b_dict['algo_feed_args'] = algo_feed_args
        rl_b_dict['algo_c'] = algo_c
        rl_b_dict['t_min'] = window_start

    # This is the "K" in top-K
    K = 1

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
    rl_events = []
    rl_u_2 = []

    for idx in range(test_batches):
        mgr, exp_b = EB.get_real_data_mgr_chpt_np(
            rl_b_dict,
            t_min=window_start,
            batch_sim_opt=eval_sim_opts,
            seed=init_seed + idx,
            with_broadcaster=True
        )
        mgr.run_dynamic(max_events=MAX_EVENTS)
        rl_dfs.append(mgr.get_state().get_dataframe())
        rl_events.append(mgr.state.events)

        # Calculating the u^2 loss
        c_is = exp_b.get_all_c_is()
        time_deltas = exp_b.get_all_time_deltas()
        rl_u_2.append(exp_b.exp_sampler.calc_quad_loss(time_deltas, c_is))

    num_tweets = [RU.num_tweets_of(df, broadcaster_id=eval_sim_opts.src_id)
                  for df in rl_dfs]
    capacity_cap, capacity_std = np.mean(num_tweets), np.std(num_tweets)

    ret['capacity'] = capacity_cap
    ret['capacity_std'] = capacity_std

    ret['RL_u_2_mean'] = np.mean(rl_u_2)
    ret['RL_u_2_std'] = np.std(rl_u_2)

    if not only_rl:
        # Figure out what 'q' to use for RQ to get the same number of tweets.
        # Removing 'RQ_cap_adjust' because RQ systematically tweets more.
        q_RQ = RU.sweep_q(eval_sim_opts, capacity_cap=capacity_cap - RQ_cap_adjust,
                          verbose=verbose, q_init=q, parallel=False, max_events=MAX_EVENTS,
                          max_iters=MAX_ITERS, only_tol=True, tol=0.1)
        ret['q_RQ'] = q_RQ

        # Run RedQueen.
        RQ_dfs = []
        RQ_events = []
        for idx in range(test_batches):
            # Deliberately using eval_sim_opts.s, as it was used to calculate q_RQ.
            # It seems to be initialized to constant (equal significance).
            opt = OM.Opt(src_id=eval_sim_opts.src_id,
                         s=eval_sim_opts.s, seed=init_seed + idx, q=q_RQ)
            mgr = eval_sim_opts.update({'q': q_RQ}).create_manager_with_broadcaster(opt)
            # mgr = eval_sim_opts.update({}).create_manager_with_opt(seed=init_seed + idx)
            mgr.state.time = window_start
            mgr.run_dynamic(max_events=MAX_EVENTS)
            RQ_dfs.append(mgr.get_state().get_dataframe())
            RQ_events.append(opt.state.events)

        if algo_feed:
            # Figure out what 'q' to use for RQ to get the same number of tweets.
            # Removing 'RQ_cap_adjust' because RQ systematically tweets more.
            q_RQ_algo = ES.sweep_q_algo(
                sim_opts=eval_sim_opts,
                capacity_cap=capacity_cap - RQ_cap_adjust,
                algo_feed_args=algo_feed_args,
                algo_c=algo_c,
                verbose=verbose,
                q_init=1000.0,
                max_events=MAX_EVENTS,
                max_iters=MAX_ITERS,
                tol=0.1,
                only_tol=True,
                t_min=window_start,
            )
            ret['q_RQ_algo'] = q_RQ_algo

            # Run RedQueen heuristic.
            RQ_algo_dfs = []
            RQ_algo_events = []
            for idx in range(test_batches):
                # Deliberately not using eval_sim_opts.s, it seems to be initialized to
                # something strange.
                opt = ES.OptAlgo(src_id=eval_sim_opts.src_id, seed=init_seed + idx, q=q_RQ_algo,
                                 algo_feed_args=algo_feed_args, algo_c=algo_c)
                mgr = eval_sim_opts.update({'q': q_RQ_algo}).create_manager_with_broadcaster(opt)
                # mgr = eval_sim_opts.update({}).create_manager_with_opt(seed=init_seed + idx)
                mgr.state.time = window_start
                mgr.run_dynamic(max_events=MAX_EVENTS)
                RQ_algo_dfs.append(mgr.get_state().get_dataframe())
                RQ_algo_events.append(opt.state.events)

        # Run Poisson.
        poisson_dfs = []
        poisson_events = []
        rate = capacity_cap / (eval_sim_opts.end_time - window_start)
        for idx in range(test_batches):
            poisson = OM.Poisson2(src_id=eval_sim_opts.src_id, seed=init_seed + idx, rate=rate)
            mgr = eval_sim_opts.create_manager_with_broadcaster(poisson)
            mgr.state.time = window_start
            mgr.run_dynamic(max_events=MAX_EVENTS)
            poisson_dfs.append(mgr.get_state().get_dataframe())
            poisson_events.append(mgr.get_state().events)

        # Running Karimi
        T = eval_sim_opts.end_time - window_start
        num_segments = 10
        seg_len = T / num_segments
        wall_mgr = eval_sim_opts.create_manager_for_wall()
        wall_mgr.run_dynamic(max_events=MAX_EVENTS)
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
        params = (init_seed, capacity_cap, num_segments, eval_sim_opts, wall_intensities, None)
        op = OR.worker_kdd(params, verbose=verbose, Ks=[K], window_start=window_start)

        karimi_dfs = []
        karimi_events = []
        for idx in range(test_batches):
            piecewise = OM.PiecewiseConst(
                src_id=eval_sim_opts.src_id,
                seed=init_seed * 2 + idx,
                change_times=window_start + np.arange(num_segments) * seg_len,
                rates=op['kdd_opt_{}'.format(K)] / seg_len
            )
            piecewise_const_mgr = eval_sim_opts.create_manager_with_broadcaster(piecewise)
            piecewise_const_mgr.state.time = window_start
            piecewise_const_mgr.run_dynamic(max_events=MAX_EVENTS)
            df = piecewise_const_mgr.state.get_dataframe()
            karimi_dfs.append(df)
            karimi_events.append(piecewise_const_mgr.get_state().events)

    # Calculating metrics
    if only_rl:
        all_settings = [('RL', rl_dfs)]
    else:
        all_settings = [('RL', rl_dfs), ('RQ', RQ_dfs), ('poisson', poisson_dfs),
                        ('karimi', karimi_dfs)]

        if algo_feed:
            all_settings += [('RQ_algo', RQ_algo_dfs)]

    metric_name = 'num_tweets'
    for type, dfs in all_settings:
        metric = [RU.num_tweets_of(df, broadcaster_id=eval_sim_opts.src_id) for df in dfs]
        ret[type + '_' + metric_name + '_mean'], ret[type + '_' + metric_name + '_std'] = (np.mean(metric), np.std(metric))

    metric_name = 'top_k'
    for type, dfs in all_settings:
        metric = [RU.time_in_top_k(df, K=K, sim_opts=eval_sim_opts) for df in dfs]
        ret[type + '_' + metric_name + '_mean'], ret[type + '_' + metric_name + '_std'] = (np.mean(metric), np.std(metric))

    metric_name = 'avg_rank'
    for type, dfs in all_settings:
        metric = [RU.average_rank(df, sim_opts=eval_sim_opts) for df in dfs]
        ret[type + '_' + metric_name + '_mean'], ret[type + '_' + metric_name + '_std'] = (np.mean(metric), np.std(metric))

    metric_name = 'r_2'
    for type, dfs in all_settings:
        metric = [RU.int_r_2(df, sim_opts=eval_sim_opts) for df in dfs]
        ret[type + '_' + metric_name + '_mean'], ret[type + '_' + metric_name + '_std'] = (np.mean(metric), np.std(metric))

    if algo_feed:
        if only_rl:
            all_settings = [('RL', rl_events)]
        else:
            all_settings = [('RL', rl_events), ('RQ', RQ_events),
                            ('poisson', poisson_events), ('karimi', karimi_events)]
            if algo_feed:
                all_settings += [('RQ_algo', RQ_algo_events)]

        for type, all_events in all_settings:
            r_2_algo = []
            r_algo = []
            top_k_algo = []

            for events in all_events:
                # Calculate some metrics here itself.
                times, r_2 = ES.algo_true_rank(
                    sink_ids=sink_ids,
                    src_id=eval_sim_opts.src_id,
                    events=events,
                    start_time=window_start,
                    end_time=eval_sim_opts.end_time,
                    steps=REWARD_STEPS,
                    all_prefs=algo_feed_args,
                    square=True,
                    c=algo_c
                )
                r_2_algo.append(np.sum(r_2) * (times[1] - times[0]))

                times, ranks = ES.algo_true_rank(
                    sink_ids=sink_ids,
                    src_id=eval_sim_opts.src_id,
                    events=events,
                    start_time=window_start,
                    end_time=eval_sim_opts.end_time,
                    steps=REWARD_STEPS,
                    all_prefs=algo_feed_args,
                    square=False,
                    c=algo_c
                )
                r_algo.append(np.sum(ranks) * (times[1] - times[0]))

                times, top_ks = ES.algo_top_k(
                    sink_ids=sink_ids,
                    src_id=eval_sim_opts.src_id,
                    events=events,
                    start_time=window_start,
                    end_time=eval_sim_opts.end_time,
                    K=K,
                    steps=REWARD_STEPS,
                    all_prefs=algo_feed_args,
                    c=algo_c
                )
                top_k_algo.append(np.sum(top_ks) * (times[1] - times[0]))

            ret[type + '_r_2_algo_mean'] = np.mean(r_2_algo)
            ret[type + '_r_2_algo_std'] = np.std(r_2_algo)

            ret[type + '_avg_rank_algo_mean'] = np.mean(r_algo)
            ret[type + '_avg_rank_algo_std'] = np.std(r_algo)

            ret[type + '_top_k_algo_mean'] = np.mean(top_k_algo)
            ret[type + '_top_k_algo_std'] = np.std(top_k_algo)

    return ret


@click.command()
@click.argument('tweeters_data_file')
@click.argument('output_dir')
@click.argument('save_csv')
@click.option('--batches', 'batches', help='How large should the test batches be.', default=64, show_default=True)
@click.option('--force/--no-force', 'force', help='Whether to overwrite the output-csv file.', default=False, show_default=True)
@click.option('--RQ-cap-adjust', 'RQ_cap_adjust', help='How much to compensate the tweets of RedQueen by.', default=1.0, show_default=True)
@click.option('--for-epoch', 'for_epoch', help='Whether to produce the CSV for a given epoch. Negative numbers of running it on the latest epoch. Can provide a comma separated list to run it on multiple epochs one after the other.', default='-1', show_default=True)
@click.option('--limit-num-users', 'limit_num_users', help='Select only these many users to run the experiments on. Selection is sorted by first the number of training checkpoints and then the idx. Set negative to run on all users.', default=-1, show_default=True)
@click.option('--parallel/--no-parallel', 'parallel', help='Whether to run on Multi-processing.', default=True, show_default=True)
@click.option('--verbose/--no-verbose', 'verbose', help='Verbose mode.', default=False, show_default=True)
@click.option('--only-rl/--no-only-rl', 'only_rl', help='Calculate only RL stats.', default=False, show_default=True)
@click.option('--algo-feed/--no-algo-feed', 'algo_feed', help='Consider algorithmic feeds?', default=False, show_default=True)
@click.option('--algo-frac', 'algo_frac', help='What fraction of the window is the lifetime of the priority queue?', default=0.1, show_default=True)
@click.option('--merge-sinks/--no-merge-sinks', 'merge_sinks', help='Whether to merge the sinks or not.', default=True, show_default=True)
@click.option('--set-wt-zero/--no-set-wt-zero', 'set_wt_zero', help='Force wt to be zero.', default=False, show_default=True)
def run(output_dir, save_csv, tweeters_data_file, batches, force, RQ_cap_adjust, for_epoch,
        parallel, verbose, only_rl, algo_feed, algo_frac, merge_sinks, set_wt_zero,
        limit_num_users):
    """Read all OUTPUT_DIR and compile the results for all users and save them in SAVE_CSV.
    The user data is read from IN_DATA_FILE and `batches` number of batches are executed.
    """
    cmd(output_dir, save_csv, tweeters_data_file, batches, force,
        RQ_cap_adjust, for_epoch, parallel, verbose, only_rl, algo_feed,
        algo_frac, merge_sinks, set_wt_zero, limit_num_users)


def cmd(output_dir, save_csv, tweeters_data_file, batches, force, RQ_cap_adjust,
        for_epoch, parallel, verbose, only_rl, algo_feed, algo_frac, merge_sinks,
        set_wt_zero, limit_num_users):

    if os.path.exists(save_csv) and not force:
        print('File {} exists and --force was not supplied.'.format(save_csv))
        sys.exit(-1)

    read_user_data(tweeters_data_file)

    save_dirs = glob.glob(os.path.join(output_dir, EB.SAVE_DIR_GLOB))
    user_idxes = [int(EB.SAVE_DIR_REGEX.search(x)[1]) for x in save_dirs]

    if limit_num_users > 0:
        raise NotImplementedError

    all_epochs = [int(x) for x in for_epoch.split(',')]

    save_dict = []
    all_params = [(x, output_dir, batches, RQ_cap_adjust,
                   for_epoch_, verbose, only_rl, algo_feed, algo_frac,
                   merge_sinks, set_wt_zero)
                  for x in user_idxes
                  for for_epoch_ in all_epochs]

    try:
        if parallel:
            with MP.Pool() as pool:
                for x in pool.imap_unordered(worker_user, all_params):
                    save_dict.append(x)
        else:
            for param in all_params:
                save_dict.append(worker_user(param))
    finally:
        # Save something even if we have to kill the simulation at some point.
        save_df = pd.DataFrame(save_dict)

        save_csv_dir = os.path.dirname(save_csv)
        if save_csv_dir != '':
            os.makedirs(save_csv_dir, exist_ok=True)

        save_df.to_csv(save_csv, index=False)
        print('Saved {} users.'.format(len(save_dict)))


if __name__ == '__main__':
    run()

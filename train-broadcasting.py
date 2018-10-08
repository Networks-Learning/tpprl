#!/usr/bin/env python
import warnings
# This removes the annoying warning from h5py
warnings.simplefilter(action='ignore', category=FutureWarning)

import click
import dill
import os
import tpprl.exp_sampler as ES
import tpprl.exp_broadcaster as EB
import tpprl.read_data_utils as RDU
from tpprl.utils import _now
import tensorflow as tf
import numpy as np
import warnings
import sys
from collections import defaultdict


def log_eval(u_data):
    mean_reward = np.mean(u_data['rewards'])
    mean_loss = np.mean(u_data['loss'])
    if 'RQ_perf' in u_data:
        mean_RQ = np.mean(u_data['RQ_perf'])
    else:
        mean_RQ = -1

    print('Mean reward = {}, Mean loss = {}, CTG = {}, RQ = {}'
          .format(mean_reward, mean_loss, mean_reward + mean_loss, mean_RQ))


@click.command()
@click.argument('all_user_data_file', type=click.Path(exists=True))
@click.argument('user_idx', type=int)
@click.argument('output_dir')
@click.option('--N', 'N', help='How many posts to consider in a window.', default=300, show_default=True)
@click.option('--q', help='Weight of the regularizer.', default=100.0, show_default=True)
@click.option('--gpu', help='Which GPU device to use.', default='/gpu:0', show_default=True)  # Is also effected by masking via CUDA_VISIBLE_DEVICES.
@click.option('--hidden-dims', 'hidden_dims', help='How many hidden dimensions to use.', default=8, show_default=True)
@click.option('--epochs', 'epochs', help='How many epochs to train for.', default=200, show_default=True)
@click.option('--num-iters', 'num_iters', help='How many iterations in each epoch.', default=20, show_default=True)
@click.option('--save-every', 'save_every', help='How many epochs to save a copy to disk.', default=5, show_default=True)
@click.option('--only-cpu/--no-only-cpu', 'only_cpu', help='Whether to use GPUs at all.', default=False, show_default=True)
@click.option('--with-summaries/--no-with-summaries', 'with_summaries', help='Whether to produce summaries in output_dir.', default=False, show_default=True)
@click.option('--reward', 'reward_kind', help='What kind of reward to use.', default='r_2_reward', show_default=True)
@click.option('--reward-top-k', 'K', help='The K in top-k reward.', default=1, show_default=True)
@click.option('--restore/--no-restore', 'should_restore', help='Whether to restore from a previous save, if present.', default=True, show_default=True)
@click.option('--until', 'until', help='How many steps of iterations to run.', default=10000, show_default=True)
@click.option('--log-device-placement/--no-log-device-placement', 'log_device_placement', help='Whether to list which GPU is being used.', default=False, show_default=True)
@click.option('--allow-growth/--no-allow-growth', 'allow_growth', help='Whether to grow GPU memory or allocate all together.', default=True, show_default=True)
@click.option('--algo-feed/--no-algo-feed', 'algo_feed', help='Use algorithmic feeds.', default=False, show_default=True)
@click.option('--algo-c', 'algo_c', help='DEPRECATED: The decay parameter for algorithmic feeds.', default=0.5, show_default=True)
@click.option('--algo-lifetime-frac', 'algo_lifetime_frac', help='The decay parameter for algorithmic feeds.', default=0.1, show_default=True)
@click.option('--algo-approx/--no-algo-approx', 'with_approx_rewards', help='Whether to use exact or approximate rewards for algorithmic feeds.', default=True, show_default=True)
@click.option('--merge-sinks/--no-merge-sinks', 'merge_sinks', help='Should all followers be merged into one giant wall.', default=False, show_default=True)
@click.option('--with-zero-wt/--no-with-zero-wt', 'with_zero_wt', help='Force wt to be zero.', default=False, show_default=True)
def run(all_user_data_file, user_idx, output_dir, q, N, gpu, reward_kind, K, should_restore, algo_lifetime_frac,
        hidden_dims, only_cpu, with_summaries, epochs, num_iters, save_every, until,
        log_device_placement, allow_growth, algo_feed, algo_c, with_approx_rewards,
        merge_sinks, with_zero_wt):
    """Read data from `all_user_data`, extract `user_idx` from the array and run code for it."""

    assert reward_kind in [EB.R_2_REWARD, EB.TOP_K_REWARD], '"{}" is not recognized as a reward_kind.'.format(reward_kind)

    save_dir = os.path.join(output_dir, EB.SAVE_DIR_TMPL.format(user_idx))
    if not os.path.exists(save_dir) and should_restore:
        warnings.warn('{} does not exist, will NOT RESTORE.'.format(save_dir))

    with open(all_user_data_file, 'rb') as f:
        all_user_data = dill.load(f)
        one_user_data = all_user_data[user_idx]

        if merge_sinks:
            print(_now(), 'Merging the sinks!')
            one_user_data = RDU.merge_sinks(one_user_data)

    print(_now(), 'Making the trainer ...')
    sim_opts = one_user_data['sim_opts'].update({'q': q})

    num_other_broadcasters = len(sim_opts.other_sources)
    num_followers = len(sim_opts.sink_ids)

    # These parameters can also be made arguments, if needed.
    max_events = 50000
    reward_time_steps = 1000
    decay_steps = 1
    with_baseline = True
    batch_size = 16

    trainer_opts_seed = 42
    trainer_opts = EB.mk_def_exp_recurrent_trainer_opts(
        seed=trainer_opts_seed,
        device_gpu=gpu,
        hidden_dims=hidden_dims,
        num_other_broadcasters=num_other_broadcasters,
        only_cpu=only_cpu,
        max_events=max_events,
        reward_top_k=K,
        reward_kind=reward_kind,
        batch_size=batch_size,
        decay_steps=decay_steps,
        num_followers=num_followers,
        with_baseline=with_baseline,
        summary_dir=os.path.join(output_dir, 'train-summary-user_idx-{}/train'.format(user_idx)),
        save_dir=save_dir,
        set_wt_zero=with_zero_wt,
    )

    config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=log_device_placement
    )
    config.gpu_options.allow_growth = allow_growth

    sess = tf.Session(config=config)
    trainer = EB.ExpRecurrentTrainer(
        sim_opts=sim_opts,
        _opts=trainer_opts,
        sess=sess
    )
    print(_now(), 'trainer made.')

    sink_ids = one_user_data['sim_opts'].sink_ids
    window_len = (one_user_data['duration'] / one_user_data['num_other_posts']) * N
    lifetimes = defaultdict(lambda: algo_lifetime_frac * window_len)

    algo_feed_seed = 42 + 1
    algo_feed_args = ES.make_freq_prefs(
        one_user_data=one_user_data,
        sink_ids=sink_ids,
        src_lifetime_dict=lifetimes
    )

    user_opt_dict = {}
    user_opt_dict['trainer_opts_dict'] = trainer_opts._get_dict()
    user_opt_dict['num_other_broadcasters'] = len(trainer.sim_opts.other_sources)
    user_opt_dict['hidden_dims'] = trainer.num_hidden_states
    user_opt_dict['num_followers'] = len(trainer.sim_opts.sink_ids)
    user_opt_dict['seed'] = trainer_opts_seed

    user_opt_dict['algo_feed'] = algo_feed
    user_opt_dict['algo_feed_seed'] = algo_feed_seed
    user_opt_dict['algo_feed_args'] = algo_feed_args
    user_opt_dict['algo_c'] = algo_c
    user_opt_dict['algo_with_approx_rewards'] = with_approx_rewards
    user_opt_dict['algo_reward_time_steps'] = reward_time_steps

    # Needed for experiments later
    user_opt_dict['N'] = N
    user_opt_dict['q'] = q

    os.makedirs(trainer.save_dir, exist_ok=True)
    with open(os.path.join(trainer.save_dir, 'user_opt_dict.dill'), 'wb') as f:
        dill.dump(user_opt_dict, f)

    trainer.initialize(finalize=True)

    if should_restore and os.path.exists(save_dir):
        try:
            trainer.restore()
        except (FileNotFoundError, AttributeError):
            warnings.warn('"{}" exists, but no save files were found. Not restoring.'
                          .format(save_dir))

    global_steps = trainer.sess.run(trainer.global_step)
    if global_steps > until:
        print(
            _now(),
            'Have already run {} > {} iterations, not going further.'
            .format(global_steps, until)
        )

    op_dir = os.path.join(output_dir, 'u_data-user_idx-{}/'.format(user_idx))
    os.makedirs(op_dir, exist_ok=True)

    # start_time, end_time = one_user_data['user_event_times'][0], one_user_data['user_event_times'][-1]
    if algo_feed:
        u_datas = [EB.get_real_data_eval_algo(
            trainer=trainer,
            one_user_data=one_user_data,
            N=N,
            batch_c=algo_c,
            algo_feed_args=algo_feed_args,
            reward_time_steps=reward_time_steps,
            with_approx_rewards=with_approx_rewards
        )]
    else:
        u_datas = [EB.get_real_data_eval(trainer, one_user_data, N=N, with_red_queen=True)]

    log_eval(u_datas[-1])

    for epoch in range(epochs):
        # Ensure that the output is pushed to the SLURM file.
        sys.stdout.flush()
        step = trainer.sess.run(trainer.global_step)
        with_df = (epoch == epochs - 1) or (step > until)

        if algo_feed:
            EB.train_real_data_algo(
                trainer=trainer,
                N=N,
                one_user_data=one_user_data,
                num_iters=num_iters,
                init_seed=42 + user_idx,
                algo_feed_args=algo_feed_args,
                with_summaries=with_summaries,
                with_approx_rewards=with_approx_rewards,
                batch_c=algo_c,
                reward_time_steps=reward_time_steps,
            )
            u_datas.append(
                EB.get_real_data_eval_algo(
                    trainer=trainer,
                    one_user_data=one_user_data,
                    N=N,
                    with_df=with_df,
                    algo_feed_args=algo_feed_args,
                    reward_time_steps=reward_time_steps,
                    with_approx_rewards=with_approx_rewards,
                    batch_c=algo_c,
                )
            )
        else:
            EB.train_real_data(
                trainer,
                N=N,
                one_user_data=one_user_data,
                num_iters=num_iters,
                init_seed=42 + user_idx,
                with_summaries=with_summaries
            )
            u_datas.append(
                EB.get_real_data_eval(
                    trainer,
                    one_user_data,
                    N=N,
                    with_red_queen=True,
                    with_df=with_df
                )
            )

        log_eval(u_datas[-1])

        if (epoch + 1) % save_every == 0 or with_df:
            file_name = 'u_data-{}.dill' if not with_df else 'u_data-{}-final.dill'
            op_file_name = os.path.join(op_dir, file_name.format(step))
            with open(op_file_name, 'wb') as f:
                dill.dump(u_datas, f)

            print(_now(), 'Saved: {}'.format(op_file_name))

            if step > until:
                print(
                    _now(),
                    'Have already run {} > {} iterations, not going further.'
                    .format(step, until)
                )
                break


if __name__ == '__main__':
    run()

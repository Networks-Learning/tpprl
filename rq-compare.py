#!/usr/bin/env python
import click
import dill
import os
import tpprl.exp_broadcaster as EB
from tpprl.utils import _now
import tensorflow as tf
import numpy as np
import warnings
import sys


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
@click.argument('all_user_data_file')
@click.argument('user_idx', type=int)
@click.argument('output_dir')
@click.option('--N', 'N', help='How many posts to consider in a window.', default=300)
@click.option('--q', help='Weight of the regularizer.', default=100.0)
@click.option('--gpu', help='Which GPU device to use.', default='/gpu:0')  # Is also effected by masking via CUDA_VISIBLE_DEVICES.
@click.option('--hidden-dims', 'hidden_dims', help='Which GPU device to use.', default=8)
@click.option('--epochs', 'epochs', help='How many batches to train for.', default=200)
@click.option('--num-iters', 'num_iters', help='How many batches to train for.', default=5)
@click.option('--save-every', 'save_every', help='How many epochs to save a copy to disk.', default=5)
@click.option('--only-cpu/--no-only-cpu', 'only_cpu', help='Whether to use GPUs at all.', default=False)
@click.option('--with-summaries/--no-with-summaries', 'with_summaries', help='Whether to produce summaries in output_dir.', default=False)
@click.option('--reward', 'reward_kind', help='What kind of reward to use.', default='r_2_reward')
@click.option('--reward-top-k', 'K', help='The K in top-k reward.', default=1)
@click.option('--restore/--no-restore', 'should_restore', help='Whether to restore from a previous save, if present.', default=True)
@click.option('--until', 'until', help='How many steps of iterations to run.', default=10000)
@click.option('--log-device-placement/--no-log-device-placement', 'log_device_placement', help='Whether to list which GPU is being used.', default=False)
@click.option('--allow-growth/--no-allow-growth', 'allow_growth', help='Whether to grow GPU memory or allocate all together.', default=True)
def run(all_user_data_file, user_idx, output_dir, q, N, gpu, reward_kind, K, should_restore,
        hidden_dims, only_cpu, with_summaries, epochs, num_iters, save_every, until,
        log_device_placement, allow_growth):
    """Read data from `all_user_data`, extract `user_idx` from the array and run code for it."""

    assert reward_kind in [EB.R_2_REWARD, EB.TOP_K_REWARD], '"{}" is not recognized as a reward_kind.'.format(reward_kind)

    save_dir = os.path.join(output_dir, 'train-save-user_idx-{}'.format(user_idx))
    if not os.path.exists(save_dir) and should_restore:
        warnings.warn('{} does not exist, will NOT RESTORE.'.format(save_dir))

    with open(all_user_data_file, 'rb') as f:
        all_user_data = dill.load(f)
        one_user_data = all_user_data[user_idx]

    print(_now(), 'Making the trainer ...')
    sim_opts = one_user_data['sim_opts'].update({'q': q})

    num_other_broadcasters = len(sim_opts.other_sources)
    num_followers = len(sim_opts.sink_ids)

    max_events = 50000
    decay_steps = 1
    with_advantage = True
    batch_size = 16

    trainer_opts = EB.mk_def_exp_recurrent_trainer_opts(
        seed=42,
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
        with_advantage=with_advantage,
        summary_dir=os.path.join(output_dir, 'train-summary-user_idx-{}'.format(user_idx)),
        save_dir=save_dir,
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

    user_opts_dict = {}
    user_opts_dict['trainer_opts_dict'] = trainer_opts._get_dict()
    user_opts_dict['num_other_broadcasters'] = len(trainer.sim_opts.other_sources)
    user_opts_dict['hidden_dims'] = trainer.num_hidden_states
    user_opts_dict['num_followers'] = len(trainer.sim_opts.sink_ids)
    user_opts_dict['seed'] = 42

    # Needed for experiments later
    user_opts_dict['N'] = N
    user_opts_dict['q'] = q

    os.makedirs(trainer.save_dir, exist_ok=True)
    with open(os.path.join(trainer.save_dir, 'user_opt_dict.dill'), 'wb') as f:
        dill.dump(user_opts_dict, f)

    trainer.initialize(finalize=True)

    if should_restore and os.path.exists(save_dir):
        try:
            trainer.restore()
        except (FileNotFoundError, AttributeError):
            warnings.warn('"{}" exists, but no save files were found. Not restoring.'.format(save_dir))

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
    u_datas = [EB.get_real_data_eval(trainer, one_user_data, N=N, with_red_queen=True)]
    log_eval(u_datas[-1])

    for epoch in range(epochs):
        # Ensure that the output is pushed to the SLURM file.
        sys.stdout.flush()
        EB.train_real_data(
            trainer,
            N=N,
            one_user_data=one_user_data,
            num_iters=num_iters,
            init_seed=42 + user_idx,
            with_summaries=with_summaries
        )

        step = trainer.sess.run(trainer.global_step)
        with_df = (epoch == epochs - 1) or (step > until)

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

#!/usr/bin/env python
import click
import dill
import os
import tpprl.exp_broadcaster as EB
import tensorflow as tf
import numpy as np


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
@click.option('--gpu', help='Which GPU device to use.', default='/gpu:0')
@click.option('--hidden-dims', 'hidden_dims', help='Which GPU device to use.', default=8)
@click.option('--epochs', 'epochs', help='How many batches to train for.', default=200)
@click.option('--num-iters', 'num_iters', help='How many batches to train for.', default=5)
@click.option('--save-every', 'save_every', help='How many epochs to save a copy to disk.', default=5)
@click.option('--only-cpu/--no-only-cpu', 'only_cpu', help='Whether to use GPUs at all.', default=False)
@click.option('--with-summaries/--no-with-summaries', 'with_summaries', help='Whether to produce summaries in output_dir.', default=False)
@click.option('--reward', 'reward_kind', help='What kind of reward to use.', default='r_2_reward')
@click.option('--reward-top-k', 'K', help='The K in top-k reward.', default=1)
def run(all_user_data_file, user_idx, output_dir, q, N, gpu, reward_kind, K,
        hidden_dims, only_cpu, with_summaries, epochs, num_iters, save_every):
    """Read data from `all_user_data`, extract `user_idx` from the array and run code for it."""

    assert reward_kind in [EB.R_2_REWARD, EB.TOP_K_REWARD], '"{}" is not recognized as a reward_kind.'.format(reward_kind)

    with open(all_user_data_file, 'rb') as f:
        all_user_data = dill.load(f)
        one_user_data = all_user_data[user_idx]

    print('Making the trainer ...')
    sim_opts = one_user_data['sim_opts'].update({'q': q})

    num_other_broadcasters = len(sim_opts.other_sources)
    num_followers = len(sim_opts.sink_ids)

    max_events = 50000
    decay_steps = 1
    reward_kind = EB.R_2_REWARD
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
        save_dir=os.path.join(output_dir, 'train-save-user_idx-{}'.format(user_idx)),
    )

    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    trainer = EB.ExpRecurrentTrainer(
        sim_opts=sim_opts,
        _opts=trainer_opts,
        sess=sess
    )
    print('trainer made.')
    trainer.initialize(finalize=True)

    op_dir = os.path.join(output_dir, 'u_data-user_idx-{}/'.format(user_idx))
    os.makedirs(op_dir, exist_ok=True)

    # start_time, end_time = one_user_data['user_event_times'][0], one_user_data['user_event_times'][-1]
    u_datas = [EB.get_real_data_eval(trainer, one_user_data, N=N, with_red_queen=True)]
    log_eval(u_datas[-1])

    for epoch in range(epochs):
        EB.train_real_data(
            trainer,
            N=N,
            one_user_data=one_user_data,
            num_iters=num_iters,
            init_seed=42 + user_idx,
            with_summaries=with_summaries
        )

        u_datas.append(EB.get_real_data_eval(trainer, one_user_data, N=N, with_red_queen=True))
        log_eval(u_datas[-1])

        if (epoch + 1) % save_every == 0 or epoch == epochs - 1:
            op_file_name = os.path.join(op_dir, 'u_data-{}.dill'.format(epoch))
            with open(op_file_name, 'wb') as f:
                dill.dump(u_datas, f)

            print('Saved: {}'.format(op_file_name))


if __name__ == '__main__':
    run()

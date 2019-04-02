# TODO: modify this file for finance
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
import os
import sys

from .cell_finance import TPPRExpMarkedCellStacked_finance as ExpTrader
from .util_finance import _now


def cmd(output_dir, should_restore, T, with_summaries,
        summary_suffix, only_cpu, q, epochs,
        num_iters, save_every, until):
    """Read initial difficulty of items from INITIAL_DIFFICULTY_CSV, ALPHA and
    BETA, train an optimal teacher and save the results to output_dir."""

    summary_dir = os.path.join(output_dir, '/output_event_{}_0_day.csv"'.format(summary_suffix))
    save_dir = os.path.join(output_dir, '/save/')

    os.makedirs(summary_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    teacher_opts = ExpTrader.mk_def_(
        hidden_dims=8,
        learning_rate=0.02,
        decay_rate=0.02,
        summary_dir=summary_dir,
        save_dir=save_dir,
        batch_size=32,
        only_cpu=only_cpu,
        T=T,
        q=q
    )

    config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False
    )
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    teacher = ExpTrader.ExpRecurrentTeacher(
        _opts=teacher_opts,
        sess=sess
    )

    teacher.initialize(finalize=True)

    if should_restore and os.path.exists(save_dir):
        try:
            teacher.restore()
            global_steps = teacher.sess.run(teacher.global_step)
            print(_now(), "Restored successfully to step {}.".format(global_steps))
        except (FileNotFoundError, AttributeError):
            warnings.warn('"{}" exists, but no save files were found. Not restoring.'
                          .format(save_dir))

    global_steps = teacher.sess.run(teacher.global_step)
    if global_steps > until:
        print(
            _now(),
            'Have already run {} > {} iterations, not going further.'
            .format(global_steps, until)
        )

    for epoch in range(epochs):
        sys.stdout.flush()

        teacher.train_many(
            num_iters=num_iters,
            init_seed=42,
            with_summaries=with_summaries,
            with_memorize_loss=False,
            save_every=save_every,
        )

        step = teacher.sess.run(teacher.global_step)
        if step > until:
            print(
                _now(),
                'Have already run {} > {} iterations, not going further.'
                .format(step, until)
            )
            break


if __name__ == '__main__':
    cmd(output_dir="/NL/tpprl-result/work/rl-finance/results_TF_RL_strategy/", should_restore=True, T=1254133800)
"""output_dir, should_restore, T, tau, with_summaries,
        summary_suffix, only_cpu, q, epochs,
        num_iters, save_every, until"""

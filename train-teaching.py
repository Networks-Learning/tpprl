#!/usr/bin/env python
import warnings
# This removes the annoying warning from h5py
warnings.simplefilter(action='ignore', category=FutureWarning)

import click
import os
import tpprl.exp_teacher as ET
from tpprl.utils import _now
import tensorflow as tf
import numpy as np
import sys


@click.command()
@click.argument('initial_difficulty_csv', type=click.Path(exists=True))
@click.argument('alpha', type=float)
@click.argument('beta', type=float)
@click.argument('output_dir', type=click.Path(exists=True))
@click.option('--epochs', 'epochs', help='How many epochs to train for.', default=1000, show_default=True)
@click.option('--num-iters', 'num_iters', help='How many iterations in each epoch.', default=50, show_default=True)
@click.option('--save-every', 'save_every', help='How many epochs to save a copy of the parameters to disk.', default=200, show_default=True)
@click.option('--T', 'T', help='The learning duration (in days).', default=14, show_default=True)
@click.option('--tau', 'tau', help='Delay before the test.', default=2, show_default=True)
@click.option('--with-summaries/--no-with-summaries', 'with_summaries', help='Whether to save summaries.', default=False, show_default=True)
@click.option('--summary-suffix', 'summary_suffix', help='Suffix to add to the summary directory', default='', show_default=True)
@click.option('--only-cpu/--no-only-cpu', 'only_cpu', help='Whether to use only the CPU for training.', default=True, show_default=True)
@click.option('--q', 'q', help='Weight for the intensity regularizer.', default=0.00025, show_default=True)
@click.option('--q-entropy', 'q_entropy', help='Weight for the entropy regularizer.', default=0.002, show_default=True)
@click.option('--restore/--no-restore', 'should_restore', help='Whether to restore from the last save or overwrite the previous progress (if it exists).', default=True, show_default=True)
@click.option('--until', 'until', help='How many steps of iterations to run.', default=20000, show_default=True)
@click.option('--with-mp/--no-with-mp', 'with_MP', help='Whether to use multiprocessing module to run simulations in parallel.', default=True, show_default=True)
@click.option('--with-recall-probs/--no-with-recall-probs', 'with_recall_probs', help='Whether to provide true probability of recall or only binary feedback to the agent.', default=False, show_default=True)
@click.option('--with-zero-wt/--no-with-zero-wt', 'with_zero_wt', help='Force wt to be zero.', default=False, show_default=True)
def cmd(initial_difficulty_csv, alpha, beta, output_dir, should_restore,
        T, tau, with_summaries, summary_suffix, only_cpu, q, q_entropy,
        epochs, num_iters, save_every, until, with_MP, with_recall_probs,
        with_zero_wt):
    """Read initial difficulty of items from INITIAL_DIFFICULTY_CSV, ALPHA and
    BETA, train an optimal teacher and save the results to output_dir."""

    with open(initial_difficulty_csv, 'r') as f:
        n_0s = [float(x.strip()) for x in f.readline().split(',')]

    num_items = len(n_0s)

    scenario_opts = {
        'T': T,
        'tau': tau,
        'n_0s': n_0s,
        'alphas': np.ones(num_items) * alpha,
        'betas': np.ones(num_items) * beta,
    }

    summary_dir = os.path.join(output_dir, 'summary/train-{}'.format(summary_suffix))
    save_dir = os.path.join(output_dir, 'save/')

    teacher_opts = ET.mk_def_teacher_opts(
        num_items=num_items,
        hidden_dims=8,
        learning_rate=0.02,
        decay_rate=0.02,
        summary_dir=summary_dir,
        save_dir=save_dir,
        batch_size=32,
        only_cpu=only_cpu,
        T=T,
        tau=tau,
        q=q,
        q_entropy=q_entropy,
        learning_bump=1.0,
        decay_steps=10,
        scenario_opts=scenario_opts,
        set_wt_zero=with_zero_wt,
    )

    config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False
    )
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    teacher = ET.ExpRecurrentTeacher(
        _opts=teacher_opts,
        sess=sess,
        num_items=num_items
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
            with_MP=with_MP,
            with_memorize_loss=False,
            save_every=save_every,
            with_recall_probs=with_recall_probs,
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
    cmd()

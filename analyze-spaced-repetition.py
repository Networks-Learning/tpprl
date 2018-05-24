#!/usr/bin/env python

import warnings
# This removes the annoying warning from h5py
warnings.simplefilter(action='ignore', category=FutureWarning)


import matplotlib
matplotlib.use('agg')

import seaborn as sns
sns.set(style='ticks', palette='Set1')
sns.despine()

import matplotlib.pyplot as plt

import click
import os
import numpy as np
import tpprl.exp_teacher as ET
from tpprl.utils import _now
import tensorflow as tf
from tpprl.plot_utils import latexify, format_axes


@click.command()
@click.argument('initial_difficulty_csv', type=click.Path(exists=True))
@click.argument('alpha', type=float)
@click.argument('beta', type=float)
@click.argument('save_dir', type=click.Path(exists=True))
@click.option('--T', 'T', help='The learning duration (in days).', default=14, show_default=True)
@click.option('--tau', 'tau', help='Delay before the test.', default=2, show_default=True)
@click.option('--only-cpu/--no-only-cpu', 'only_cpu', help='Whether to use only the CPU during evaluation.', default=True, show_default=True)
@click.option('--batches', 'batches', help='How many test batches to sample results from.', default=100, show_default=True)
@click.option('--verbose/--no-verbose', 'verbose', help='Produce verbose output.', default=True, show_default=True)
def cmd(initial_difficulty_csv, alpha, beta, save_dir, T, tau, only_cpu, batches, verbose):
    """Read the initial difficulty of items from INITIAL_DIFFICULTY_CSV, use
    the ALPHA and BETA specified, restore the teacher model from the given
    SAVE_DIR and compare the performance of the method against various
    baselines."""
    with open(initial_difficulty_csv, 'r') as f:
        n_0s = [float(x.strip()) for x in f.readline().split(',')]

    num_items = len(n_0s)

    init_seed = 1337
    scenario_opts = {
        'T': T,
        'tau': tau,
        'n_0s': n_0s,
        'alphas': np.ones(num_items) * alpha,
        'betas': np.ones(num_items) * beta,
    }

    summary_dir = None

    teacher_opts = ET.mk_def_teacher_opts(
        num_items=num_items,
        hidden_dims=8,
        save_dir=save_dir,
        only_cpu=only_cpu,
        T=T,
        tau=tau,
        scenario_opts=scenario_opts,

        # The values here do not matter because we will not be training
        # the NN here.
        summary_dir=summary_dir,
        learning_rate=0.02,
        decay_rate=0.02,
        batch_size=32,
        q=0.0001,
        q_entropy=0.002,
        learning_bump=1.0,
        decay_steps=10,
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

    # Restores to the latest version.
    teacher.restore()

    global_steps = teacher.sess.run(teacher.global_step)
    if verbose:
        print(_now(), "Restored successfully to step {}.".format(global_steps))

    # Evaluating the performance of RL.
    _f_d, RL_test_scens = ET.get_test_feed_dicts(teacher, range(init_seed, init_seed + batches))
    RL_rewards = [s.reward() for s in RL_test_scens]

    num_test_reviews = np.mean([x.get_num_events() for x in RL_test_scens])

    # Performance using uniform baseline
    rets_unif = [
        ET.uniform_random_baseline(
            scenario_opts, target_reviews=num_test_reviews,
            seed=seed + 8, verbose=False
        ) for seed in range(init_seed, init_seed + batches)
    ]

    # Performance if using Memorize.
    q_MEM = ET.sweep_memorize_q(scenario_opts, num_test_reviews, q_init=1.0,
                                verbose=verbose)

    rets_mem = [
        ET.memorize_baseline(
            scenario_opts, q_max=q_MEM,
            seed=seed + 8, verbose=False)
        for seed in range(init_seed, init_seed + batches)
    ]

    # Plotting reward (i.e. recall at T + tau)

    plt.figure()
    latexify(fig_width=2.25, largeFonts=False)
    colors = sns.color_palette(n_colors=3)

    Y = {
        'RL': RL_rewards,
        'MEM': [x['reward'] / (-100) for x in rets_mem],
        'Uniform': [[x['reward'] / (-100) for x in rets_unif]],
    }

    box = plt.boxplot([Y['RL'], Y['MEM'], Y['Uniform']],
                      whis=0,
                      showmeans=True,
                      showfliers=False,
                      showcaps=False,
                      patch_artist=True,
                      medianprops={'linewidth': 1.0},
                      boxprops={'linewidth': 1.0, 'edgecolor': colors[0],
                                'facecolor': colors[1], 'alpha': 0.3},
                      whiskerprops={'linewidth': 0})

    for idx in range(len(colors)):
        box['boxes'][idx].set_facecolor(colors[idx])
        box['boxes'][idx].set_edgecolor(colors[idx])
        box['means'][idx].set_markersize(5)
        box['means'][idx].set_markerfacecolor(colors[idx])
        box['medians'][idx].set_color(colors[idx])

    plt.yticks([0.0, 0.25, 0.50], ['0\%', '25\%', '50\%'])
    plt.xticks([1, 2, 3], [r'\textsc{TPPRL}', r'\textsc{Memorize}', 'Uniform'])
    plt.tight_layout()
    format_axes(plt.gca())

    plot_base = './output-plots/'
    os.makedirs(plot_base, exist_ok=True)

    plt.savefig(os.path.join(plot_base, 'recall-results-{}-{}.pdf'.format(T, tau)),
                bbox_inches='tight', pad_inches=0)

    # Plotting item difficulty

    plt.figure()
    latexify(fig_width=2.25, largeFonts=False)
    colors = sns.color_palette(n_colors=3)

    Y = {
        'RL': [scenario_opts['n_0s'][item]  for x in RL_test_scens for item in x.items],
        'MEM': [scenario_opts['n_0s'][item]  for x in rets_mem for item, _ in x['review_timings']],
        'Uniform': [scenario_opts['n_0s'][item]  for x in rets_unif for item, _ in x['review_timings']]
    }

    box = plt.boxplot([Y['RL'], Y['MEM'], Y['Uniform']],
                      whis=0,
                      showmeans=True,
                      showfliers=False,
                      showcaps=False,
                      patch_artist=True,
                      medianprops={'linewidth': 1.0},
                      boxprops={'linewidth': 1.0, 'edgecolor': colors[0],
                                'facecolor': colors[1], 'alpha': 0.3},
                      whiskerprops={'linewidth': 0})

    for idx in range(len(colors)):
        box['boxes'][idx].set_facecolor(colors[idx])
        box['boxes'][idx].set_edgecolor(colors[idx])
        box['means'][idx].set_markersize(5)
        box['means'][idx].set_markerfacecolor(colors[idx])
        box['medians'][idx].set_color(colors[idx])

    plt.xticks([1, 2, 3], [r'\textsc{TPPRL}', r'\textsc{Memorize}', 'Uniform'])
    plt.tight_layout()
    format_axes(plt.gca())
    plt.savefig(os.path.join(plot_base, 'item-difficulty.pdf'), bbox_inches='tight', pad_inches=0)

    # Plotting reviews per day
    RL_times = [np.floor(t) for s in RL_test_scens for t in np.cumsum(s.time_deltas)]
    MEM_times = [np.floor(t) for x in rets_mem for _, t in x['review_timings']]

    plt.figure()
    latexify(fig_width=2.25, largeFonts=False)

    c1, c2 = sns.color_palette(n_colors=2)

    f, (a1, a2) = plt.subplots(2, 1)
    a1.hist(RL_times, bins=np.arange(T + 1), density=True, color=c1, alpha=0.5, label='RL')
    a1.set_yticks([.04, .08])
    a1.set_yticklabels([r'4\%', r'8\%'])
    a1.set_ylabel('TPPRL')
    a1.set_ylim([0.04, 0.08])
    a1.set_xticks([0.5, 3.5, 6.5, 9.5, 13.5])
    a1.set_xticklabels([1, 4, 7, 10, 14])
    format_axes(a1)

    a2.hist(MEM_times, bins=np.arange(T + 1), density=True, color=c2, alpha=0.5, label=r'\textsc{Mem}')
    a2.set_yticks([0, .04, .08], [r'0\%', r'4\%', r'8\%'])
    a2.set_xticks([0.5, 3.5, 6.5, 9.5, 13.5])
    a2.set_xticklabels([1, 4, 7, 10, 14])
    a2.set_ylabel(r'\textsc{Memorize}')
    a2.set_ylim([0.04, 0.08])
    a2.set_yticks([.04, .08])
    a2.set_yticklabels([r'4\%', r'8\%'])
    format_axes(a2)

    # plt.legend(ncol=2, bbox_to_anchor=(0, 0, 1, 1.1))
    plt.tight_layout()
    plt.savefig(os.path.join(plot_base, 'reviews-every-day.pdf'), bbox_inches='tight', pad_inches=0)

    print(_now(), 'Done.')


if __name__ == '__main__':
    cmd()

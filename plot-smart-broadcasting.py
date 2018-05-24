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
import pandas as pd
import os
import re
import numpy as np
from tpprl.utils import _now
from tpprl.plot_utils import latexify, format_axes


@click.command()
@click.argument('analyzed_csv', type=click.Path(exists=True))
@click.option('--algo-feed/--no-algo-feed', 'algo_feed', help='Whether to assume that the feed was algorithmic or not.', default=True, show_default=True)
def cmd(analyzed_csv, algo_feed):
    """Produces the smart-broadcasting plots after reading values from ANALYZED_CSV."""

    chpt_regex = re.compile(r'-([\d]*)$')
    df = pd.read_csv('results-algo/top_k-q_0.33-s-fix-adjust_0.csv').dropna()

    # Derive to which epoch was this instance trained to.
    df['chpt'] = [int(chpt_regex.search(x)[1]) for x in df.chpt_file]

    other_key = 'RQ_algo_num_tweets_mean' if algo_feed else 'RQ_num_tweets_mean'

    # Determine the users for which number of tweets are close enough.
    index = (np.abs(df['RL_num_tweets_mean'] - df[other_key]) < 2)
    print(_now(), '{} users are valid.'.format(np.sum(index)))

    # Setting up output
    plot_base = './output-plots'
    os.makedirs(plot_base, exist_ok=True)

    # Calculating the top-k metric.

    if algo_feed:
        baseline_key = 'poisson_top_k_algo_mean'
        RL_key = 'RL_top_k_algo_mean'
        RQ_key = 'RQ_algo_top_k_algo_mean'
        karimi_key = 'karimi_top_k_algo_mean'
    else:
        baseline_key = 'poisson_top_k_mean'
        RL_key = 'RL_top_k_mean'
        RQ_key = 'RQ_top_k_mean'
        karimi_key = 'karimi_top_k_mean'

    baseline = df[baseline_key][index]
    Y = {}
    Y['RL'] = df[RL_key][index] / baseline
    Y['RQ'] = df[RQ_key][index] / baseline
    Y['karimi'] = df[karimi_key][index] / baseline

    # Plotting the top-k metric.
    plt.figure()
    colors = sns.color_palette(n_colors=3)
    latexify(fig_width=2.25, largeFonts=False)
    box = plt.boxplot([Y['RL'],
                       Y['RQ'],
                       Y['karimi']],
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

    plt.xticks([1, 2, 3], [r'TPPRL', r'\textsc{RedQueen}',
                           'Karimi'])
    plt.tight_layout()
    format_axes(plt.gca())
    plt.savefig(os.path.join(plot_base, 'algo-top-1.pdf'), bbox_inches='tight', pad_inches=0)

    # Calculating the avg-rank metric

    if algo_feed:
        baseline_key = 'poisson_avg_rank_algo_mean'
        RL_key = 'RL_avg_rank_algo_mean'
        RQ_key = 'RQ_algo_avg_rank_algo_mean'
        karimi_key = 'karimi_avg_rank_algo_mean'
    else:
        baseline_key = 'poisson_avg_rank_mean'
        RL_key = 'RL_avg_rank_mean'
        RQ_key = 'RQ_avg_rank_mean'
        karimi_key = 'karimi_avg_rank_mean'

    baseline = df[baseline_key][index]
    Y = {}
    Y['RL'] = df[RL_key][index] / baseline
    Y['RQ'] = df[RQ_key][index] / baseline
    Y['karimi'] = df[karimi_key][index] / baseline

    # Plotting the top-k metric.
    plt.figure()
    colors = sns.color_palette(n_colors=3)
    latexify(fig_width=2.25, largeFonts=False)
    box = plt.boxplot([Y['RL'],
                       Y['RQ'],
                       Y['karimi']],
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

    plt.xticks([1, 2, 3], [r'TPPRL', r'\textsc{RedQueen}',
                           'Karimi'])
    plt.tight_layout()
    format_axes(plt.gca())
    plt.savefig(os.path.join(plot_base, 'algo-avg-rank.pdf'), bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    cmd()

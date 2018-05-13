import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import logging

# The functions dependent on matplotlib and related libraries are kept here
# because they may cause problems while importing the modules form the command
# line without jupyter if matplotlib is configured in a certain way.


def plot_u(times, u, t_deltas, is_own_event, figsize=(16, 6)):
    """Plots the intensity output by our broadcaster.

    TODO: May not work if the max_events are reached.
    """

    t_deltas = np.asarray(t_deltas)
    is_own_event = np.asarray(is_own_event)

    seq_len = np.nonzero(t_deltas == 0)[0][0]  # First index where t_delta = 0
    abs_t = np.cumsum(t_deltas[:seq_len])
    abs_own = is_own_event[:seq_len]

    our_events = [t for (t, o) in zip(abs_t, abs_own) if o]
    other_events = [t for (t, o) in zip(abs_t, abs_own) if not o]

    u_max = np.max(u)

    plt.figure(figsize=(16, 6))

    c1, c2, c3 = sns.color_palette(n_colors=3)

    plt.plot(times, u, label='$u(t)$', color=c1)
    plt.vlines(our_events, 0, 0.75 * u_max, label='Us', alpha=0.5, color=c2)
    plt.vlines(other_events, 0, 0.75 * u_max, label='Others', alpha=0.5, color=c3)
    plt.xlabel('Time')
    plt.ylabel('$u(t)$')
    plt.legend()


def latexify(fig_width=None, fig_height=None, columns=1, largeFonts=False):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    assert(columns in [1, 2])

    if fig_width is None:
        fig_width = 3.39 if columns == 1 else 6.9  # width in inches

    if fig_height is None:
        golden_mean = (np.sqrt(5) - 1.0) / 2.0    # Aesthetic ratio
        fig_height = fig_width * golden_mean  # height in inches

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        logging.warning("WARNING: fig_height too large:" + fig_height +
                        "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
        fig_height = MAX_HEIGHT_INCHES

    params = {
        'backend': 'ps',
        'text.latex.preamble': ['\\usepackage{gensymb}'],
        'axes.labelsize': 10 if largeFonts else 7,  # fontsize for x and y labels (was 10)
        'axes.titlesize': 10 if largeFonts else 7,
        'font.size': 10 if largeFonts else 7,  # was 10
        'legend.fontsize': 10 if largeFonts else 7,  # was 10
        'xtick.labelsize': 10 if largeFonts else 7,
        'ytick.labelsize': 10 if largeFonts else 7,
        'text.usetex': True,
        'figure.figsize': [fig_width, fig_height],
        'font.family': 'serif',
        'xtick.minor.size': 0.5,
        'xtick.major.pad': 1.5,
        'xtick.major.size': 1,
        'ytick.minor.size': 0.5,
        'ytick.major.pad': 1.5,
        'ytick.major.size': 1
    }

    # matplotlib.rcParams.update(params)
    plt.rcParams.update(params)


def format_axes(ax):
    SPINE_COLOR = 'grey'
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.5)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction='out', color=SPINE_COLOR)

    return ax

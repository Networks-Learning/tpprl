import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

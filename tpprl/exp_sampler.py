"""This file contains the samplers implemented using numpy and RedQueen
compatible broadcasters (both single-threaded and multi-threaded versions)."""
import decorated_options as Deco
import numpy as np
import redqueen.opt_model as OM


class CDFSampler:
    """A generic sampler which assumes that the intensity u(t) has the form:

    f(vt * hi + bt + wt * (t - t0)).

    The function 'f' is left unimplemented and has to be implemented by sub-classes.

    Notationally, c = vt * hi + bt.
    """

    @Deco.optioned()
    def __init__(self, vt, wt, bt, init_h, t_min, seed=42):
        self.seed = seed
        self.vt = np.asarray(vt).squeeze()
        self.wt = np.asarray(wt).squeeze()
        self.bt = np.asarray(bt).squeeze()

        self.w = np.asarray(wt).squeeze()
        self.Q = 1.0

        self.random_state = np.random.RandomState(seed)
        self.t0 = t_min
        self.reset(t_min, init_h, reset_sample=True)

    def cdf(self, t):
        """Return the CDF calculated at 't', given the current state of the intensity.
        It also assumes that the last event was at self.t0.
        """
        raise NotImplementedError('cdf has to be implemented by the sub-class')

    def generate_sample(self):
        """Find a sample from the Exp process."""
        raise NotImplementedError('generate_sample has to be implemented by the sub-class.')

    def reset_only_sample(self, cur_time):
        """Resets only the present sample.

        This allows generating multiple samples (by updating t0) from the
        intensity one after the other without registering any new events."""

        self.c = self.c + (self.w * (cur_time - self.t0))
        self.t0 = cur_time
        self.u_unif = self.random_state.rand()
        self.Q = 1.0

        return self.generate_sample()

    def reset(self, cur_time, init_h, reset_sample):
        """Reset the sampler for generating another event."""

        if reset_sample:
            self.u_unif = self.random_state.rand()
            self.Q = 1.0
        else:
            self.Q *= (1 - self.cdf(cur_time))

        self.h = init_h
        self.c = np.squeeze(self.vt.dot(self.h) + self.bt)
        self.t0 = cur_time

        return self.generate_sample()

    def register_event(self, time, new_h, own_event):
        """Saves the event and generated a new time for the next event."""
        return self.reset(time, new_h, reset_sample=own_event)

    def get_last_hidden_state(self):
        return self.h

    def get_last_c(self):
        return self.c

    def int_u(self, dt, c):
        """Value of U(dt) - U(0)."""
        raise NotImplementedError('int_u needs to be implemented by the sub-class.')

    def log_u(self, t, c):
        """Value of log u(t)."""
        raise NotImplementedError('Needs to be implemented by the sub-class.')

    def int_u_2(self, t, c):
        """Value of U^2(dt) - U^2(0)."""
        raise NotImplementedError('Needs to be implemented by the sub-class.')

    def calc_quad_loss(self, event_time_deltas, c_is):
        """Calculates the regularise loss.
        The last entry of event_time_deltas should be T - t_last.
        The first entry of hidden_states should be the initial state.
        """
        return sum(self.int_u_2(dt, c)
                   for dt, c in zip(event_time_deltas, c_is))

    def calc_LL(self, event_time_deltas, c_is, is_own_event):
        """Calculates the log-likelihood.
        The last entry of event_time_deltas should be T - t_last.
        The first entry of hidden_states should be the initial state.
        The last entry of is_own_event correspond to the phantom event at the end of the survival.
        """
        assert not is_own_event[-1], "The last entry cannot be an event."

        LL_log = sum(self.log_u(dt, c)
                     for dt, c, o in zip(event_time_deltas, c_is, is_own_event)
                     if o)
        LL_int = sum(self.int_u(dt, c) for dt, c in zip(event_time_deltas, c_is))

        return LL_log - LL_int


class ExpCDFSampler(CDFSampler):
    """This is an exponential sampler."""

    def cdf(self, t):
        """Calculates the CDF assuming that the last event was at self.t0"""
        return 1 - np.exp((np.exp(self.c) / self.w) * (1 - np.exp(self.w * (t - self.t0))))

    def generate_sample(self):
        """Find a sample from the Exp process."""
        # Have the uniform sample already drawn
        D = 1 - (self.w / np.exp(self.c)) * np.log((1 - self.u_unif) / self.Q)
        if D <= 0:
            # This is the probability that no event ever happens
            return np.inf
        else:
            return self.t0 + (1 / self.w) * np.log(D)

    def int_u(self, dt, c):
        return (1 / self.wt) * (np.exp(c + self.wt * dt) - np.exp(c))

    def log_u(self, dt, c):
        return c + self.wt * dt

    def int_u_2(self, dt, c):
        return (1 / (2 * self.wt)) * (np.exp(2 * c + 2 * self.wt * dt) -
                                      np.exp(2 * c))


class SigmoidCDFSampler(CDFSampler):
    """This is an sigmoidal intensity sampler.

    Additionally, it assumes that the sigmoid is multiplied by 'k' to scale the intensity.
    """

    @Deco.optioned()
    def __init__(self, vt, wt, bt, init_h, t_min, seed=42, k=1.0):
        self.k = k
        super().__init__(vt, wt, bt, init_h, t_min, seed=seed)

    def cdf(self, t):
        C = (1 + np.exp(self.c)) / (1 + np.exp(self.c + self.wt * (t - self.t0)))
        return 1 - C ** (self.k / self.wt)

    def generate_sample(self):
        D = (1 + np.exp(self.c)) * ((1 - self.u_unif) / self.Q) ** (- self.k / self.wt) - 1
        # print('D = ', D)
        if D <= 0:
            # This is the case when no event ever happens.
            return np.inf
        else:
            return self.t0 + (np.log(D) - self.c) / self.wt

    def log_u(self, dt, c):
        return np.log(1 / (1 + np.exp(-(c + self.wt * dt))))

    def int_u(self, dt, c):
        return (self.k / self.wt) * (np.log1p(np.exp(c + self.wt * dt)) - np.log1p(np.exp(c)))

    def int_u_2(self, dt, c):
        return ((self.k ** 2) / self.wt) * (1 / (1 + np.exp(c + self.wt * dt)) +
                                            np.log1p(np.exp(c + self.wt * dt)) -
                                            1 / (1 + np.exp(c)) -
                                            np.log1p(np.exp(c)))


def gen_rand_vecs(dims, number, random_state):
    return np.asarray([x / np.linalg.norm(x) for x in
                       [random_state.standard_normal(dims) for _ in range(number)]])


def make_prefs(sink_ids, src_ids, seed=42):
    RS = np.random.RandomState(seed=seed)
    sink_prefs = gen_rand_vecs(2, len(sink_ids), RS)
    src_prefs = gen_rand_vecs(2, len(src_ids), RS)
    return {
        'sink_id_map': dict([(x, idx) for idx, x in enumerate(sink_ids)]),
        'src_id_map': dict([(x, idx) for idx, x in enumerate(src_ids)]),
        'sink_prefs': sink_prefs,
        'src_prefs': src_prefs,
        'seed': seed
    }


def algo_rank_of(past_events, sink_id, src_id, all_prefs, c=1.0, t=None):
    """Find the algorithm rank of src_id on the feed of sink_id."""
    rel_events = [(ev, all_prefs['src_id_map'][ev.src_id])
                  for ev in past_events if sink_id in ev.sink_ids]

    if len(rel_events) == 0:
        return 0

    if t is None:
        t = past_events[-1].cur_time

    sink_idx = all_prefs['sink_id_map'][sink_id]
    sink_perf_vec = all_prefs['sink_prefs'][sink_idx]
    src_prefs = all_prefs['src_prefs']

    importance = sorted(
        [(np.exp(c * (t - ev.cur_time) * (np.dot(sink_perf_vec, src_prefs[src_idx]) - 1)),
          ev.src_id)
         for ev, src_idx in rel_events],
        reverse=True
    )

    for idx, (_, ev_src_id) in enumerate(importance):
        if src_id == ev_src_id:
            return idx

    return len(importance)


def algo_ranks_from_events(events, sink_ids, src_id, all_prefs, c=1.0):
    """Calculates the algorithmic feed ranks from a set of events."""
    algo_ranks = []
    for idx in range(len(events)):
        cur_ranks = [None] * len(sink_ids)
        for sink_id in sink_ids:
            sink_idx = all_prefs['sink_id_map'][sink_id]
            cur_ranks[sink_idx] = algo_rank_of(events[0:idx],
                                               sink_id=sink_id,
                                               src_id=src_id,
                                               all_prefs=all_prefs)
        algo_ranks.append(cur_ranks)

    return np.asarray(algo_ranks)


def avg_algo_rank(past_events, algo_ranks, end_time):
    """Calculate the heuristic average rank for a priority feed."""
    survival_time = end_time - past_events[-1].cur_time
    dt = np.asarray([ev.time_delta for ev in past_events[1:]] + [survival_time])
    r_t = algo_ranks.mean(1)
    return np.sum(r_t * dt)


def algo_true_rank(sink_ids, src_id, events, start_time, end_time,
                   steps, all_prefs, square=False, c=1.0):
    """A more accurate calculation (but more expensive) of the average algorithm rank."""
    t_delta = (end_time - start_time) / steps
    times = np.arange(start_time, end_time, t_delta)
    ranks = []
    rank = 0
    idx = 0

    for t in times:
        while idx + 1 < len(events) and events[idx + 1].cur_time < t:
            idx += 1

        rank = np.mean(
            [algo_rank_of(past_events=events[:idx],
                          sink_id=x,
                          src_id=src_id,
                          all_prefs=all_prefs,
                          t=t,
                          c=c) ** (1.0 if not square else 2.0)
             for x in sink_ids]
        )
        ranks.append(rank)

    return times, np.asarray(ranks)


def algo_top_k(sink_ids, src_id, events, start_time, end_time, K,
               steps, all_prefs, c=1.0):
    """A more accurate calculation (but more expensive) of time spent in top-k."""
    t_delta = (end_time - start_time) / steps
    times = np.arange(start_time, end_time, t_delta)
    top_ks = []
    top_k = 0
    idx = 0

    for t in times:
        while idx + 1 < len(events) and events[idx + 1].cur_time < t:
            idx += 1

        top_k = np.mean(
            [1.0 if (algo_rank_of(past_events=events[:idx], sink_id=x, src_id=src_id, all_prefs=all_prefs, t=t, c=c)) < K else 0.0
             for x in sink_ids]
        )
        top_ks.append(top_k)

    return times, np.asarray(top_ks)


class ExpRecurrentBroadcasterMP(OM.Broadcaster):
    """This is a broadcaster which follows the intensity function as defined by
    RMTPP paper and updates the hidden state upon receiving each event.

    TODO: The problem is that calculation of the gradient and the loss/LL
    becomes too complicated with numerical stability issues very quickly. Need
    to implement adaptive scaling to handle that issue.
    """

    @Deco.optioned()
    def __init__(self, src_id, seed, t_min,
                 Wm, Wh, Wr, Wt, Bh, sim_opts,
                 wt, vt, bt, init_h, src_embed_map,
                 algo_feed=False, algo_feed_args=None):
        super(ExpRecurrentBroadcasterMP, self).__init__(src_id, seed)
        self.sink_ids = sim_opts.sink_ids
        self.end_time = sim_opts.end_time
        self.init = False

        # Used to create h_next
        self.Wm = Wm
        self.Wh = Wh
        self.Wr = Wr
        self.Wt = Wt
        self.Bh = Bh
        self.cur_h = init_h
        self.src_embed_map = src_embed_map
        self.algo_feed = algo_feed
        self.algo_feed_args = algo_feed_args
        self.algo_ranks = []
        self.c_is = []
        self.time_deltas = []

        # Needed for the sampler
        self.params = Deco.Options(**{
            'wt': wt,
            'vt': vt,
            'bt': bt,
            'init_h': init_h
        })

        self.exp_sampler = ExpCDFSampler(_opts=self.params,
                                         t_min=t_min,
                                         seed=seed + 1)

    def get_all_c_is(self):
        return self.c_is + [self.exp_sampler.c]

    def get_all_time_deltas(self):
        return self.time_deltas + [self.end_time - self.state.time]

    def update_hidden_state(self, src_id, time_delta):
        """Returns the hidden state after a post by src_id and time delta."""

        if not self.algo_feed:
            r_t = np.nan_to_num(
                self.state.get_wall_rank(
                    self.src_id,
                    self.sink_ids,
                    dict_form=False,
                    assume_first=True
                ).astype(float)
            )
        else:
            r_t = np.array([
                algo_rank_of(self.state.events, sink_id,
                             self.src_id, self.algo_feed_args)
                for sink_id in self.sink_ids
            ])

        self.algo_ranks.append(r_t)

        return np.tanh(
            self.Wm[self.src_embed_map[src_id], :][:, np.newaxis] +
            self.Wh.dot(self.cur_h) +
            self.Wr.dot(r_t.reshape(-1, 1)) +  # TODO: untested
            self.Wt * time_delta +
            self.Bh
        )

    def get_next_interval(self, event):
        if not self.init:
            self.init = True
            self.state.set_track_src_id(self.src_id, self.sink_ids)
            # Nothing special to do for the first event.

        self.state.apply_event(event)

        if event is None:
            # This is the first event. Post immediately to join the party?
            # Or hold off?
            # Currently, it is waiting.
            return self.exp_sampler.generate_sample() - self.last_self_event_time
        else:
            self.c_is.append(self.exp_sampler.c)
            self.time_deltas.append(event.time_delta)
            self.cur_h = self.update_hidden_state(event.src_id, event.time_delta)
            next_post_time = self.exp_sampler.register_event(
                event.cur_time,
                self.cur_h,
                own_event=event.src_id == self.src_id
            )
            next_delta = next_post_time - self.last_self_event_time
            # print(next_delta)
            assert next_delta >= 0
            return next_delta


class ExpRecurrentBroadcaster(OM.Broadcaster):
    """This is a broadcaster which follows the intensity function as defined by
    RMTPP paper and updates the hidden state upon receiving each event.

    TODO: The problem is that calculation of the gradient and the loss/LL
    becomes too complicated with numerical stability issues very quickly. Need
    to implement adaptive scaling to handle that issue.

    Also, this embeds the event history implicitly and the state function does
    not explicitly model the loss function J(.) faithfully. This is an issue
    with the theory.
    """

    @Deco.optioned()
    def __init__(self, src_id, seed, trainer, t_min=0):
        super(ExpRecurrentBroadcaster, self).__init__(src_id, seed)
        self.init = False

        self.trainer = trainer

        self.params = Deco.Options(**self.trainer.sess.run({
            # 'Wm': trainer.tf_Wm,
            # 'Wh': trainer.tf_Wh,
            # 'Bh': trainer.tf_Bh,
            # 'Wt': trainer.tf_Wt,
            # 'Wr': trainer.tf_Wr,

            'wt': trainer.tf_wt,
            'vt': trainer.tf_vt,
            'bt': trainer.tf_bt,
            'init_h': trainer.tf_h
        }))

        self.cur_h = self.params.init_h

        self.exp_sampler = ExpCDFSampler(_opts=self.params,
                                         t_min=t_min,
                                         seed=seed + 1)

    def update_hidden_state(self, src_id, time_delta):
        """Returns the hidden state after a post by src_id and time delta."""
        # Best done using self.sess.run here.
        raw_ranks = self.state.get_wall_rank(self.src_id, self.sink_ids, dict_form=False, assume_first=True)
        r_t = np.nan_to_num(raw_ranks.astype(float))

        feed_dict = {
            self.trainer.tf_b_idx: np.asarray([self.trainer.src_embed_map[src_id]]),
            self.trainer.tf_t_delta: np.asarray([time_delta]).reshape(-1),
            self.trainer.tf_h: self.cur_h,
            self.trainer.tf_rank: r_t.reshape(-1, 1)
        }
        return self.trainer.sess.run(self.trainer.tf_h_next,
                                     feed_dict=feed_dict)

    def get_next_interval(self, event):
        if not self.init:
            self.init = True
            self.state.set_track_src_id(self.src_id,
                                        self.trainer.sim_opts.sink_ids)
            # Nothing special to do for the first event.

        self.state.apply_event(event)

        if event is None:
            # This is the first event. Post immediately to join the party?
            # Or hold off?
            # Currently, it is waiting.
            return self.exp_sampler.generate_sample() - self.last_self_event_time
        else:
            self.cur_h = self.update_hidden_state(event.src_id, event.time_delta)
            next_post_time = self.exp_sampler.register_event(
                event.cur_time,
                self.cur_h,
                own_event=event.src_id == self.src_id
            )
            next_delta = next_post_time - self.last_self_event_time
            # print(next_delta)
            assert next_delta >= 0
            return next_delta


OM.SimOpts.registerSource('ExpRecurrentBroadcaster', ExpRecurrentBroadcaster)

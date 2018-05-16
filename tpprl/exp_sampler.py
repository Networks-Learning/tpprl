"""This file contains the samplers implemented using numpy and RedQueen
compatible broadcasters (both single-threaded and multi-threaded versions)."""
import decorated_options as Deco
import numpy as np
import redqueen.opt_model as OM
import redqueen.utils as RU
import multiprocessing as MP
from collections import defaultdict


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


def make_prefs(sink_ids, src_ids, src_lifetime_dict, seed=42):
    RS = np.random.RandomState(seed=seed)
    sink_prefs = gen_rand_vecs(2, len(sink_ids), RS)
    src_prefs = gen_rand_vecs(2, len(src_ids), RS)
    return {
        'sink_id_map': dict([(x, idx) for idx, x in enumerate(sink_ids)]),
        'src_id_map': dict([(x, idx) for idx, x in enumerate(src_ids)]),
        'sink_prefs': sink_prefs,
        'src_prefs': src_prefs,
        'seed': seed,
        'lifetime': src_lifetime_dict
    }


def algo_rank_of(past_events, sink_id, src_id, all_prefs, c=1.0, t=None):
    """Find the algorithm rank of src_id on the feed of sink_id."""

    if len(past_events) == 0:
        return 0

    if t is None:
        t = past_events[-1].cur_time

    lifetime = all_prefs['lifetime']
    sink_idx = all_prefs['sink_id_map'][sink_id]
    sink_pref_vec = all_prefs['sink_prefs'][sink_idx]
    src_prefs = all_prefs['src_prefs']

    src_importance = {src_id: np.dot(sink_pref_vec, src_prefs[all_prefs['src_id_map'][src_id]])
                      for src_id in all_prefs['src_id_map'].keys()}

    feed = sorted(
        [(src_importance[ev.src_id] if (t - ev.cur_time) < lifetime[ev.src_id] else -100000,
          ev.cur_time,
          ev.src_id)
         for ev in past_events if sink_id in ev.sink_ids],
        key=lambda x: (x[0], x[1])
    )

    # print(feed)

    for idx, (_, _, ev_src_id) in enumerate(feed[::-1]):
        if src_id == ev_src_id:
            return idx

    # If our broadcaster is not present in there, then just assume he is at
    # the bottom.
    return len(feed)

# Old implementation:
    # importance = sorted(
    #     [(np.exp(c * (t - ev.cur_time) * (np.dot(sink_perf_vec, src_prefs[src_idx]) - 1)),
    #       ev.src_id)
    #      for ev, src_idx in rel_events],
    #     key=lambda x: x[0]
    # )

    # Though Python's sorted is stable, the behavior is unclear if reverse=True
    # importance = sorted(
    #     [(np.dot(sink_perf_vec, src_prefs[src_idx]), ev.src_id)
    #      for ev, src_idx in rel_events],
    #     key=lambda x: x[0]
    # )

    # print(importance)


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
                                               all_prefs=all_prefs,
                                               c=c)
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
        # if idx > 1:
        #     print(t, events[:idx][-1].cur_time, rank)

        ranks.append(rank)

    return times, np.asarray(ranks)


def algo_true_rank_avg_reward(sink_ids, src_id, events, start_time, end_time,
                              steps, all_prefs, c=1.0, square=False):
    times, rank = algo_true_rank(
        sink_ids=sink_ids,
        src_id=src_id,
        events=events,
        start_time=start_time,
        end_time=end_time,
        steps=steps,
        all_prefs=all_prefs,
        square=square,
        c=1.0,
    )
    return np.sum(rank) * (times[1] - times[0])


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
            [1.0 if (algo_rank_of(past_events=events[:idx],
                                  sink_id=x,
                                  src_id=src_id,
                                  all_prefs=all_prefs,
                                  t=t,
                                  c=c)) < K else 0.0
             for x in sink_ids]
        )
        top_ks.append(top_k)

    return times, np.asarray(top_ks)


def algo_top_k_reward(sink_ids, src_id, events, start_time, end_time, K,
                      steps, all_prefs, c=1.0):
    times, top_ks = algo_top_k(sink_ids=sink_ids,
                               src_id=src_id,
                               events=events,
                               start_time=start_time,
                               end_time=end_time,
                               K=K,
                               steps=steps,
                               all_prefs=all_prefs,
                               c=1.0)
    return np.sum(top_ks) * (times[1] - times[0])


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
                 algo_feed=False, algo_feed_args=None, algo_c=1.0):
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
        self.algo_c = algo_c

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
                algo_rank_of(self.state.events, sink_id=sink_id,
                             src_id=self.src_id,
                             all_prefs=self.algo_feed_args,
                             c=self.algo_c)
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


class OptAlgo(OM.Broadcaster):
    """This is the RedQueen broadcaster with the ranks being provided from
    the algorithmic feeds instead of working from the true ranks.

    This will be used as a heuristic to compare the performance of RL algo to.
    """

    def __init__(self, src_id, seed, algo_feed_args, q=1.0, s=1.0, algo_c=0.5):
        super(OptAlgo, self).__init__(src_id, seed)
        self.q = q
        self.s = s
        self.algo_c = algo_c
        self.sqrt_s_by_q = None
        self.old_rate = 0
        self.init = False
        self.algo_feed_args = algo_feed_args

    def get_next_interval(self, event):
        if not self.init:
            self.init = True
            self.state.set_track_src_id(self.src_id, self.sink_ids)

            if isinstance(self.s, dict):
                self.s_vec = np.asarray([self.s[x]
                                         for x in sorted(self.sink_ids)])
            else:
                # Assuming that the self.q is otherwise a scalar number.
                # Or a vector with the same number of elements as sink_ids
                self.s_vec = np.ones(len(self.sink_ids), dtype=float) * self.s

            self.sqrt_s_by_q = np.sqrt(self.s_vec / self.q)

        self.state.apply_event(event)

        if event is None:
            # Tweet immediately if this is the first event.
            self.old_rate = 0
            return 0
        elif event.src_id == self.src_id:
            # No need to tweet if we are on top of all walls
            self.old_rate = 0
            return np.inf
        else:
            # check status of all walls and find position in it.
            # r_t = self.state.get_wall_rank(self.src_id, self.sink_ids,
            #                                dict_form=False)

            r_t = np.array([
                algo_rank_of(self.state.events, sink_id=sink_id,
                             src_id=self.src_id,
                             all_prefs=self.algo_feed_args,
                             c=self.algo_c)
                for sink_id in self.sink_ids
            ])

            # TODO: If multiple walls are updated at the same time, should the
            # drawing happen only once after all the updates have been applied
            # or one at a time? Does that make a difference? Probably not. A
            # lot more work if the events are sent one by one per wall, though.
            new_rate, old_rate = self.sqrt_s_by_q.dot(r_t), self.old_rate
            diff_rate = new_rate - old_rate
            self.old_rate = new_rate
            cur_time = event.cur_time

            if diff_rate > 0:
                # Super-positioning.
                t_delta_new = self.random_state.exponential(scale=1.0 / diff_rate)

                if self.last_self_event_time + self.t_delta > cur_time + t_delta_new:
                    return cur_time + t_delta_new - self.last_self_event_time
                else:
                    # Stick to the old estimate.
                    pass
            else:
                unif = self.random_state.uniform()
                if new_rate / old_rate < unif:
                    # Rejection sampling would have rejected this time
                    # So sample the next time using the new rate, starting from previous sample.
                    if new_rate == 0:
                        return np.inf

                    t_delta_new = self.t_delta + self.random_state.exponential(scale=1.0 / new_rate)
                    return cur_time + t_delta_new - self.last_self_event_time
                else:
                    # Stick to the old estimate.
                    pass


def calc_q_capacity_iter_algo(sim_opts, q, algo_c, algo_feed_args,
                              seeds=None, max_events=None, t_min=0):
    if seeds is None:
        seeds = range(10)

    sim_opts = sim_opts.update({'q': q})

    capacities = np.zeros(len(seeds), dtype=float)
    for idx, seed in enumerate(seeds):
        opt_algo = OptAlgo(src_id=sim_opts.src_id, seed=100 + seed,
                           algo_feed_args=algo_feed_args, algo_c=algo_c, q=q)
        m = sim_opts.create_manager_with_broadcaster(opt_algo)
        m.state.time = t_min
        m.run_dynamic(max_events=max_events)
        capacities[idx] = RU.num_tweets_of(m.get_state().get_dataframe(),
                                           broadcaster_id=sim_opts.src_id)

    return capacities


def sweep_q_algo(sim_opts, capacity_cap, algo_feed_args, algo_c, tol=1e-2,
                 verbose=False, q_init=1000.0, max_events=None,
                 max_iters=float('inf'), t_min=0, only_tol=False):
    # We know that on average, the ∫u(t)dt decreases with increasing 'q'

    def terminate_cond(new_capacity):
        return abs(new_capacity - capacity_cap) / capacity_cap < tol or \
            (not only_tol and np.ceil(capacity_cap - 1) <= new_capacity <= np.ceil(capacity_cap + 1))

    # if q_init is None:
    #     wall_mgr = sim_opts.create_manager_for_wall()
    #     wall_mgr.run_dynamic()
    #     r_t = rank_of_src_in_df(wall_mgr.state.get_dataframe(), -1)
    #     q_init = (4 * (r_t.iloc[-1].mean() ** 2) * (sim_opts.end_time) ** 2) / (np.pi * np.pi * (capacity_cap + 1) ** 4)
    #     if verbose:
    #         logTime('q_init = {}'.format(q_init))

    # Step 1: Find the upper/lower bound by exponential increase/decrease
    init_cap = calc_q_capacity_iter_algo(
        sim_opts=sim_opts, q=q_init, algo_feed_args=algo_feed_args,
        algo_c=algo_c, max_events=max_events, t_min=t_min
    ).mean()

    if verbose:
        RU.logTime('Initial capacity = {}, target capacity = {}, q_init = {}'
                   .format(init_cap, capacity_cap, q_init))

    if terminate_cond(init_cap):
        return q_init

    q = q_init
    if init_cap < capacity_cap:
        iters = 0
        while True:
            iters += 1
            q_hi = q
            q /= 2.0
            q_lo = q
            capacity = calc_q_capacity_iter_algo(
                sim_opts=sim_opts, q=q, algo_feed_args=algo_feed_args,
                algo_c=algo_c, max_events=max_events, t_min=t_min
            ).mean()
            if verbose:
                RU.logTime('q = {}, capacity = {}'.format(q, capacity))
            if terminate_cond(capacity):
                return q
            if capacity >= capacity_cap:
                break
            if iters > max_iters:
                if verbose:
                    RU.logTime('Breaking because of max-iters: {}.'.format(max_iters))
                return q
    else:
        iters = 0
        while True:
            iters += 1
            q_lo = q
            q *= 2.0
            q_hi = q
            capacity = calc_q_capacity_iter_algo(
                sim_opts=sim_opts, q=q, algo_feed_args=algo_feed_args,
                algo_c=algo_c, max_events=max_events, t_min=t_min
            ).mean()

            if verbose:
                RU.logTime('q = {}, capacity = {}'.format(q, capacity))
            # TODO: will break if capacity_cap is too low ~ 1 event.
            if terminate_cond(capacity):
                return q
            if capacity <= capacity_cap:
                break
            if iters > max_iters:
                if verbose:
                    RU.logTime('Breaking because of max-iters: {}.'.format(max_iters))
                return q

    if verbose:
        RU.logTime('q_hi = {}, q_lo = {}'.format(q_hi, q_lo))

    # Step 2: Keep bisecting on 's' until we arrive at a close enough solution.
    while True:
        q = (q_hi + q_lo) / 2.0
        new_capacity = calc_q_capacity_iter_algo(
            sim_opts=sim_opts, q=q, algo_feed_args=algo_feed_args,
            algo_c=algo_c, max_events=max_events, t_min=t_min
        ).mean()

        if verbose:
            RU.logTime('new_capacity = {}, q = {}'.format(new_capacity, q))

        if terminate_cond(new_capacity):
            # Have converged
            break
        elif new_capacity > capacity_cap:
            q_lo = q
        else:
            q_hi = q

    # Step 3: Return
    return q

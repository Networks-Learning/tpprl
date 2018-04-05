import numpy as np
import os
import redqueen.opt_model as OM
import redqueen.utils as RU
import tensorflow as tf
import decorated_options as Deco
import exp_sampler
import multiprocessing as MP

from utils import variable_summaries, _now, average_gradients


SAVE_DIR = 'tpprl-log'
os.makedirs(SAVE_DIR, exist_ok=True)

R_2_REWARD = 'r_2_reward'
TOP_K_REWARD = 'top_k_reward'


class ExpRecurrentBroadcasterMP(OM.Broadcaster):
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
    def __init__(self, src_id, seed, t_min,
                 Wm, Wh, Wr, Wt, Bh, sim_opts,
                 wt, vt, bt, init_h, src_embed_map):
        super(ExpRecurrentBroadcasterMP, self).__init__(src_id, seed)
        self.sink_ids = sim_opts.sink_ids
        self.init = False

        # Used to create h_next
        self.Wm = Wm
        self.Wh = Wh
        self.Wr = Wr
        self.Wt = Wt
        self.Bh = Bh
        self.cur_h = init_h
        self.src_embed_map = src_embed_map

        # Needed for the sampler
        self.params = Deco.Options(**{
            'wt': wt,
            'vt': vt,
            'bt': bt,
            'init_h': init_h
        })

        self.exp_sampler = exp_sampler.ExpCDFSampler(_opts=self.params,
                                                     t_min=t_min,
                                                     seed=seed + 1)

    def update_hidden_state(self, src_id, time_delta):
        """Returns the hidden state after a post by src_id and time delta."""
        # Best done using self.sess.run here.
        r_t = self.state.get_wall_rank(self.src_id, self.sink_ids, dict_form=False)
        return np.tanh(
            self.Wm[self.src_embed_map[src_id], :][:, np.newaxis] +
            self.Wh.dot(self.cur_h) +
            self.Wr * np.asarray([np.mean(r_t)]).reshape(-1) +
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
            return self.exp_sampler.generate_sample()
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


def reward_fn(df, reward_kind, reward_opts, sim_opts):
    """Calculate the reward for this trajectory."""
    if reward_kind == R_2_REWARD:
        return RU.int_r_2(df, sim_opts)
    elif reward_kind == TOP_K_REWARD:
        return -RU.time_in_top_k(df, sim_opts=sim_opts, **reward_opts)
    else:
        raise NotImplementedError('{} reward function is not implemented.'
                                  .foramt(reward_kind))


def _worker_sim(params):
    """Worker for the parallel simulation runner."""
    rl_b_args, seed = params
    rl_b_opts = Deco.Options(**rl_b_args)

    exp_b = ExpRecurrentBroadcasterMP(_opts=rl_b_opts, seed=seed * 3)
    run_sim_opts = rl_b_opts.sim_opts.update({})

    mgr = run_sim_opts.create_manager_with_broadcaster(exp_b)
    mgr.run_dynamic(max_events=rl_b_opts.max_events + 1)
    df = mgr.get_state().get_dataframe()

    reward = reward_fn(
        df=df,
        reward_kind=rl_b_opts.reward_kind,
        reward_opts=rl_b_opts.reward_opts,
        sim_opts=rl_b_opts.sim_opts
    )

    return df, reward


def run_sims_MP(trainer, seeds, processes=None):
    """Run simulations using multiprocessing."""
    rl_b_args = {
        'src_id': trainer.src_id,
        't_min': trainer.t_min,

        'sim_opts': trainer.sim_opts,
        'max_events': trainer.max_events,
        'src_embed_map': trainer.src_embed_map,

        'Wm': trainer.sess.run(trainer.tf_Wm),
        'Wh': trainer.sess.run(trainer.tf_Wh),
        'Wr': trainer.sess.run(trainer.tf_Wr),
        'Wt': trainer.sess.run(trainer.tf_Wt),
        'Bh': trainer.sess.run(trainer.tf_Bh),

        'wt': trainer.sess.run(trainer.tf_wt),
        'vt': trainer.sess.run(trainer.tf_vt),
        'bt': trainer.sess.run(trainer.tf_bt),
        'init_h': trainer.sess.run(trainer.tf_h),

        'reward_kind': trainer.reward_kind,
        'reward_opts': {
            'K': trainer.reward_top_k
        }
    }

    # return [_worker_sim((rl_b_args, seed)) for seed in seeds]

    with MP.Pool(processes=processes) as pool:
        return pool.map(_worker_sim, [(rl_b_args, seed) for seed in seeds])


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

        self.exp_sampler = exp_sampler.ExpCDFSampler(_opts=self.params,
                                                     t_min=t_min,
                                                     seed=seed + 1)

    def update_hidden_state(self, src_id, time_delta):
        """Returns the hidden state after a post by src_id and time delta."""
        # Best done using self.sess.run here.
        r_t = self.state.get_wall_rank(self.src_id, self.sink_ids, dict_form=False)

        feed_dict = {
            self.trainer.tf_b_idx: np.asarray([self.trainer.src_embed_map[src_id]]),
            self.trainer.tf_t_delta: np.asarray([time_delta]).reshape(-1),
            self.trainer.tf_h: self.cur_h,
            self.trainer.tf_rank: np.asarray([np.mean(r_t)]).reshape(-1)
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
            return self.exp_sampler.generate_sample()
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


class TPPRSigmoidCell(tf.contrib.rnn.RNNCell):
    """u(t) = k * sigmoid(vt * ht + wt * dt + bt)"""

    def __init__(self, hidden_state_size, output_size, src_id, tf_dtype,
                 Wm, Wr, Wh, Wt, Bh,
                 wt, vt, bt, k):
        self._output_size = output_size
        self._hidden_state_size = hidden_state_size
        self.src_id = src_id
        self.tf_dtype = tf_dtype

        self.tf_Wm = Wm
        self.tf_Wr = Wr
        self.tf_Wh = Wh
        self.tf_Wt = Wt
        self.tf_Bh = Bh

        self.tf_wt = wt
        self.tf_vt = vt
        self.tf_bt = bt

        self.tf_k = k

    def u_theta(self, dt, c):
        return self.tf_k / (1 + tf.exp(-(c + self.tf_wt * dt)))

    def int_u(self, dt, c):
        return (self.tf_k / self.tf_wt) * (
            tf.log1p(tf.exp(c + self.tf_wt * dt)) -
            tf.log1p(tf.exp(c))
        )

    def int_u_2(self, dt, c):
        return (np.square(self.tf_k) / self.tf_wt) * (
            tf.sigmoid(-(c + self.tf_wt * dt)) +
            tf.log1p(tf.exp(c + self.tf_wt * dt)) -
            tf.sigmoid(-c) -
            tf.log1p(tf.exp(c))
        )

    def __call__(self, inp, h_prev):
        raw_broadcaster_idx, rank, t_delta = inp
        inf_batch_size = tf.shape(raw_broadcaster_idx)[0]

        broadcaster_idx = tf.squeeze(raw_broadcaster_idx, axis=-1)

        h_next = tf.nn.tanh(
            tf.nn.embedding_lookup(self.tf_Wm, broadcaster_idx) +
            tf.matmul(h_prev, self.tf_Wh, transpose_b=True) +
            tf.matmul(rank, self.tf_Wr, transpose_b=True) +
            tf.matmul(t_delta, self.tf_Wt, transpose_b=True) +
            tf.transpose(self.tf_Bh),
            name='h_next'
        )

        c = tf.matmul(h_prev, self.tf_vt) + self.tf_wt * t_delta + self.tf_bt

        u_theta = self.u_theta(t_delta, c, name='u_theta')
        # print('u_theta = ', u_theta)

        # t_0 = tf.zeros(name='zero_time', shape=(inf_batch_size, 1), dtype=self.tf_dtype)
        # u_theta_0 = self.u_theta(t_0, name='u_theta_0')

        LL_log = tf.where(
            tf.equal(broadcaster_idx, 0),
            tf.squeeze(tf.log(u_theta), axis=-1),
            tf.zeros(dtype=self.tf_dtype, shape=(inf_batch_size,)),
            name='LL_log'
        )
        # print('LL_log = ', LL_log)
        LL_int = self.int_u(t_delta, c)
        # print('LL_int = ', LL_int)
        loss = self.int_u_2(t_delta, c)
        # print('loss = ', loss)

        return ((h_next,
                 tf.expand_dims(LL_log, axis=-1, name='LL_log'),
                 LL_int,
                 loss),
                h_next)

    @property
    def output_size(self):
        return self._output_size

    @property
    def state_size(self):
        return self._hidden_state_size


class TPPRExpCell(tf.contrib.rnn.RNNCell):
    """u(t) = exp(vt * ht + wt * dt + bt)"""

    def __init__(self, hidden_state_size, output_size, src_id, tf_dtype,
                 Wm, Wr, Wh, Wt, Bh,
                 wt, vt, bt):
        self._output_size = output_size
        self._hidden_state_size = hidden_state_size
        self.src_id = src_id
        self.tf_dtype = tf_dtype

        self.tf_Wm = Wm
        self.tf_Wr = Wr
        self.tf_Wh = Wh
        self.tf_Wt = Wt
        self.tf_Bh = Bh

        self.tf_wt = wt
        self.tf_vt = vt
        self.tf_bt = bt

    def u_theta(self, h, t_delta, name):
        return tf.exp(
            tf.matmul(h, self.tf_vt) +
            self.tf_wt * t_delta +
            self.tf_bt,
            name=name
        )

    def __call__(self, inp, h_prev):
        raw_broadcaster_idx, rank, t_delta = inp
        inf_batch_size = tf.shape(raw_broadcaster_idx)[0]

        broadcaster_idx = tf.squeeze(raw_broadcaster_idx, axis=-1)

        h_next = tf.nn.tanh(
            tf.nn.embedding_lookup(self.tf_Wm, broadcaster_idx) +
            tf.matmul(h_prev, self.tf_Wh, transpose_b=True) +
            tf.matmul(rank, self.tf_Wr, transpose_b=True) +
            tf.matmul(t_delta, self.tf_Wt, transpose_b=True) +
            tf.transpose(self.tf_Bh),
            name='h_next'
        )

        u_theta = self.u_theta(h_prev, t_delta, name='u_theta')
        # print('u_theta = ', u_theta)

        t_0 = tf.zeros(name='zero_time', shape=(inf_batch_size, 1), dtype=self.tf_dtype)
        u_theta_0 = self.u_theta(h_prev, t_0, name='u_theta_0')

        LL_log = tf.where(
            tf.equal(broadcaster_idx, 0),
            tf.squeeze(tf.log(u_theta), axis=-1),
            tf.zeros(dtype=self.tf_dtype, shape=(inf_batch_size,))
        )
        # print('LL_log = ', LL_log)
        LL_int = (1 / self.tf_wt) * (u_theta - u_theta_0)
        # print('LL_int = ', LL_int)
        loss = (1 / (2 * self.tf_wt)) * (tf.square(u_theta) - tf.square(u_theta_0))
        # print('loss = ', loss)

        return ((h_next,
                 tf.expand_dims(LL_log, axis=-1, name='LL_log'),
                 LL_int,
                 loss),
                h_next)

    @property
    def output_size(self):
        return self._output_size

    @property
    def state_size(self):
        return self._hidden_state_size


def mk_def_exp_recurrent_trainer_opts(num_other_broadcasters, hidden_dims,
                                      seed=42, **kwargs):
    """Make default option set."""
    RS  = np.random.RandomState(seed=seed)

    def_exp_recurrent_trainer_opts = Deco.Options(
        t_min=0,
        scope=None,
        with_dynamic_rnn=True,
        decay_steps=100,
        decay_rate=0.001,

        Wh=RS.randn(hidden_dims, hidden_dims) * 0.1 + np.diag(np.ones(hidden_dims)),  # Careful initialization
        Wm=RS.randn(num_other_broadcasters + 1, hidden_dims),
        Wr=RS.randn(hidden_dims, 1),
        Wt=RS.randn(hidden_dims, 1),
        Bh=RS.randn(hidden_dims, 1),

        vt=RS.randn(hidden_dims, 1),
        wt=np.abs(RS.rand(1)) * -1,
        bt=np.abs(RS.randn(1)),

        # TODO: Instead of init_h, just use hidden_dims directly as an argument.
        num_hidden_states=hidden_dims,

        max_events=5000,
        batch_size=16,

        learning_rate=.01,
        clip_norm=1.0,

        momentum=0.9,

        device_cpu='/cpu:0',
        device_gpu='/gpu:0',
        only_cpu=False,

        save_dir=SAVE_DIR,

        summary_dir=None,  # Expected: './tpprl.summary/train-{}/'.format(run)
        reward_kind=R_2_REWARD,
        reward_top_k=-1,
    )

    return def_exp_recurrent_trainer_opts.set(**kwargs)


class ExpRecurrentTrainer:
    @Deco.optioned()
    def __init__(self, Wm, Wh, Wt, Wr, Bh, vt, wt, bt, num_hidden_states,
                 sess, sim_opts, scope, t_min, batch_size, max_events,
                 learning_rate, clip_norm, with_dynamic_rnn,
                 summary_dir, save_dir, decay_steps, decay_rate, momentum,
                 reward_top_k, reward_kind, device_cpu, device_gpu, only_cpu):
        """Initialize the trainer with the policy parameters."""

        self.reward_top_k = reward_top_k
        self.reward_kind = reward_kind

        self.t_min = t_min
        self.summary_dir = summary_dir
        self.save_dir = save_dir

        self.src_embed_map = {x.src_id: idx + 1
                              for idx, x in enumerate(sim_opts.create_other_sources())}
        self.src_embed_map[sim_opts.src_id] = 0

        self.tf_dtype = tf.float32
        self.np_dtype = np.float32

        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.clip_norm = clip_norm

        self.q = sim_opts.q  # Loss is blown up by a factor of q / 2
        self.batch_size = batch_size

        self.tf_batch_size = None
        self.max_events = max_events
        self.num_hidden_states = num_hidden_states

        # init_h = np.reshape(init_h, (-1, 1))
        Bh = np.reshape(Bh, (-1, 1))

        self.scope = scope or type(self).__name__

        var_device = device_cpu if only_cpu else device_gpu

        with tf.variable_scope(self.scope):
            with tf.variable_scope('hidden_state'):
                with tf.device(var_device):
                    self.tf_Wm = tf.get_variable(name='Wm', shape=Wm.shape,
                                                 initializer=tf.constant_initializer(Wm))
                    self.tf_Wh = tf.get_variable(name='Wh', shape=Wh.shape,
                                                 initializer=tf.constant_initializer(Wh))
                    self.tf_Wt = tf.get_variable(name='Wt', shape=Wt.shape,
                                                 initializer=tf.constant_initializer(Wt))
                    self.tf_Wr = tf.get_variable(name='Wr', shape=Wr.shape,
                                                 initializer=tf.constant_initializer(Wr))
                    self.tf_Bh = tf.get_variable(name='Bh', shape=Bh.shape,
                                                 initializer=tf.constant_initializer(Bh))

                    # Needed to calculate the hidden state for one step.
                    self.tf_h = tf.get_variable(name='h', initializer=tf.zeros((self.num_hidden_states, 1), dtype=self.tf_dtype))

                self.tf_b_idx = tf.placeholder(name='b_idx', shape=1, dtype=tf.int32)
                self.tf_t_delta = tf.placeholder(name='t_delta', shape=1, dtype=self.tf_dtype)
                self.tf_rank = tf.placeholder(name='rank', shape=1, dtype=self.tf_dtype)

                self.tf_h_next = tf.nn.tanh(
                    tf.transpose(
                        tf.nn.embedding_lookup(self.tf_Wm, self.tf_b_idx, name='b_embed')
                    ) +
                    tf.matmul(self.tf_Wh, self.tf_h) +
                    self.tf_Wr * self.tf_rank +
                    self.tf_Wt * self.tf_t_delta +
                    self.tf_Bh,
                    name='h_next'
                )

            with tf.variable_scope('output'):
                with tf.device(var_device):
                    self.tf_bt = tf.get_variable(name='bt', shape=bt.shape,
                                                 initializer=tf.constant_initializer(bt))
                    self.tf_vt = tf.get_variable(name='vt', shape=vt.shape,
                                                 initializer=tf.constant_initializer(vt))
                    self.tf_wt = tf.get_variable(name='wt', shape=wt.shape,
                                                 initializer=tf.constant_initializer(wt))
                # self.tf_t_delta = tf.placeholder(name='t_delta', shape=1, dtype=self.tf_dtype)
                # self.tf_u_t = tf.exp(
                #     tf.tensordot(self.tf_vt, self.tf_h, axes=1) +
                #     self.tf_t_delta * self.tf_wt +
                #     self.tf_bt,
                #     name='u_t'
                # )

            # Create a large dynamic_rnn kind of network which can calculate
            # the gradients for a given given batch of simulations.
            with tf.variable_scope('training'):
                self.tf_batch_rewards = tf.placeholder(name='rewards',
                                                       shape=(self.tf_batch_size, 1),
                                                       dtype=self.tf_dtype)
                self.tf_batch_t_deltas = tf.placeholder(name='t_deltas',
                                                        shape=(self.tf_batch_size, max_events),
                                                        dtype=self.tf_dtype)
                self.tf_batch_b_idxes = tf.placeholder(name='b_idxes',
                                                       shape=(self.tf_batch_size, max_events),
                                                       dtype=tf.int32)
                self.tf_batch_ranks = tf.placeholder(name='ranks',
                                                     shape=(self.tf_batch_size, max_events),
                                                     dtype=self.tf_dtype)
                self.tf_batch_seq_len = tf.placeholder(name='seq_len',
                                                       shape=(self.tf_batch_size, 1),
                                                       dtype=tf.int32)
                self.tf_batch_last_interval = tf.placeholder(name='last_interval',
                                                             shape=self.tf_batch_size,
                                                             dtype=self.tf_dtype)

                # Inferred batch size
                inf_batch_size = tf.shape(self.tf_batch_b_idxes)[0]

                self.tf_batch_init_h = tf_batch_h_t = tf.zeros(name='init_h',
                                                               shape=(inf_batch_size, self.num_hidden_states),
                                                               dtype=self.tf_dtype)

                # self.LL = tf.zeros(name='log_likelihood', dtype=self.tf_dtype, shape=(self.tf_batch_size))
                # self.loss = tf.zeros(name='loss', dtype=self.tf_dtype, shape=(self.tf_batch_size))

                t_0 = tf.zeros(name='event_time', shape=(inf_batch_size,), dtype=self.tf_dtype)

                def batch_u_theta(batch_t_deltas):
                    return tf.exp(
                        tf.matmul(tf_batch_h_t, self.tf_vt) +
                        self.tf_wt * tf.expand_dims(batch_t_deltas, 1) +
                        self.tf_bt
                    )

                # TODO: Convert this to a tf.while_loop, perhaps.
                # The performance benefit is debatable, but the graph may be constructed faster.
                # TODO: Another idea is to convert it to use dynamic_rnn.
                if not with_dynamic_rnn:
                    self.h_states = []
                    self.LL_log_terms = []
                    self.LL_int_terms = []
                    self.loss_terms = []

                    for evt_idx in range(max_events):
                        # Perhaps this can be melded into the old definition of tf_h_next
                        # above by using the batch size dimension as None?
                        # TODO: Investigate
                        tf_batch_h_t = tf.where(
                            tf.tile(evt_idx < self.tf_batch_seq_len, [1, self.num_hidden_states]),
                            tf.nn.tanh(
                                tf.nn.embedding_lookup(self.tf_Wm,
                                                       self.tf_batch_b_idxes[:, evt_idx]) +
                                tf.matmul(tf_batch_h_t, self.tf_Wh, transpose_b=True) +
                                tf.matmul(tf.expand_dims(self.tf_batch_ranks[:, evt_idx], 1),
                                          self.tf_Wr, transpose_b=True) +
                                tf.matmul(tf.expand_dims(self.tf_batch_t_deltas[:, evt_idx], 1),
                                          self.tf_Wt, transpose_b=True) +
                                tf.tile(tf.transpose(self.tf_Bh), [inf_batch_size, 1])
                            ),
                            tf_batch_h_t
                            # tf.zeros(dtype=self.tf_dtype, shape=(inf_batch_size, self.num_hidden_states))
                            # The gradient of a constant w.r.t. a variable is None or 0
                        )
                        tf_batch_u_theta = tf.where(
                            evt_idx < self.tf_batch_seq_len,
                            batch_u_theta(self.tf_batch_t_deltas[:, evt_idx]),
                            tf.zeros(dtype=self.tf_dtype, shape=(inf_batch_size, 1))
                        )

                        self.h_states.append(tf_batch_h_t)
                        self.LL_log_terms.append(tf.where(
                            tf.squeeze(evt_idx < self.tf_batch_seq_len),
                            tf.where(
                                tf.equal(self.tf_batch_b_idxes[:, evt_idx], 0),
                                tf.squeeze(tf.log(tf_batch_u_theta)),
                                tf.zeros(dtype=self.tf_dtype, shape=(inf_batch_size,))),
                            tf.zeros(dtype=self.tf_dtype, shape=(inf_batch_size,))))

                        self.LL_int_terms.append(tf.where(
                            tf.squeeze(evt_idx < self.tf_batch_seq_len),
                            (1 / self.tf_wt) * tf.squeeze(
                                tf_batch_u_theta -
                                batch_u_theta(t_0)
                            ),
                            tf.zeros(dtype=self.tf_dtype, shape=(inf_batch_size,))))

                        self.loss_terms.append(tf.where(
                            tf.squeeze(evt_idx < self.tf_batch_seq_len),
                            (1 / (2 * self.tf_wt)) * tf.squeeze(
                                tf.square(tf_batch_u_theta) -
                                tf.square(batch_u_theta(t_0))
                            ),
                            tf.zeros(dtype=self.tf_dtype, shape=(inf_batch_size,))))

                    self.LL = tf.add_n(self.LL_log_terms) - tf.add_n(self.LL_int_terms)
                    self.loss = tf.add_n(self.loss_terms)
                else:
                    rnn_cell = TPPRExpCell(
                        hidden_state_size=(None, self.num_hidden_states),
                        output_size=[self.num_hidden_states] + [1] * 3,
                        src_id=sim_opts.src_id,
                        tf_dtype=self.tf_dtype,
                        Wm=self.tf_Wm, Wr=self.tf_Wr, Wh=self.tf_Wh,
                        Wt=self.tf_Wt, Bh=self.tf_Bh,
                        wt=self.tf_wt, vt=self.tf_vt, bt=self.tf_bt
                    )

                    ((self.h_states, LL_log_terms, LL_int_terms, loss_terms), tf_batch_h_t) = tf.nn.dynamic_rnn(
                        rnn_cell,
                        inputs=(tf.expand_dims(self.tf_batch_b_idxes, axis=-1),
                                tf.expand_dims(self.tf_batch_ranks, axis=-1),
                                tf.expand_dims(self.tf_batch_t_deltas, axis=-1)),
                        sequence_length=tf.squeeze(self.tf_batch_seq_len, axis=-1),
                        dtype=self.tf_dtype,
                        initial_state=self.tf_batch_init_h
                    )
                    self.LL_log_terms = tf.squeeze(LL_log_terms, axis=-1)
                    self.LL_int_terms = tf.squeeze(LL_int_terms, axis=-1)
                    self.loss_terms = tf.squeeze(loss_terms, axis=-1)

                    self.LL = tf.reduce_sum(self.LL_log_terms, axis=1) - tf.reduce_sum(self.LL_int_terms, axis=1)
                    self.loss = tf.reduce_sum(self.loss_terms, axis=1)

            with tf.name_scope('calc_u'):
                # These are operations needed to calculate u(t) in post-processing.
                # These can be done entirely in numpy-space, but since we have a
                # version in tensorflow, they have been moved here to avoid
                # memory leaks.
                # Otherwise, new additions to the graph were made whenever the
                # function calc_u was called.

                self.calc_u_h_states = tf.placeholder(
                    name='calc_u_h_states',
                    shape=(self.tf_batch_size, self.max_events, self.num_hidden_states),
                    dtype=self.tf_dtype
                )
                self.calc_u_batch_size = tf.placeholder(
                    name='calc_u_batch_size',
                    shape=(None,),
                    dtype=tf.int32
                )

                self.calc_u_c_is_init = tf.matmul(self.tf_batch_init_h, self.tf_vt) + self.tf_bt
                self.calc_u_c_is_rest = tf.split(
                    tf.squeeze(
                        tf.matmul(
                            self.calc_u_h_states,
                            tf.tile(
                                tf.expand_dims(self.tf_vt, 0),
                                [self.calc_u_batch_size[0], 1, 1]
                            )
                        ) + self.tf_bt,
                        axis=-1
                    ),
                    self.max_events,
                    axis=1,
                    name='calc_u_c_is_rest'
                )

                self.calc_u_t_deltas = tf.split(
                    self.tf_batch_t_deltas,
                    self.max_events,
                    axis=1,
                    name='calc_u_t_deltas'
                )

                self.calc_u_is_own_event = tf.split(
                    tf.equal(self.tf_batch_b_idxes, 0),
                    self.max_events,
                    axis=1,
                    name='calc_u_is_own_event'
                )

        # Here, outside the loop, add the survival term for the batch to
        # both the loss and to the LL.
        self.LL_last = -(1 / self.tf_wt) * tf.squeeze(
            batch_u_theta(self.tf_batch_last_interval) - batch_u_theta(t_0)
        )

        self.loss_last = (1 / (2 * self.tf_wt)) * tf.squeeze(
            tf.square(batch_u_theta(self.tf_batch_last_interval)) -
            tf.square(batch_u_theta(t_0))
        )

        self.LL += self.LL_last
        self.loss += self.loss_last
        self.loss *= self.q / 2

        self.all_tf_vars = [self.tf_Wh, self.tf_Wm, self.tf_Wt, self.tf_Bh,
                            self.tf_Wr, self.tf_bt, self.tf_vt, self.tf_wt]

        # The gradients are added over the batch if made into a single call.
        # TODO: Perhaps there is a faster way of calculating these gradients?
        self.LL_grads = {x: [tf.gradients(y, x)
                             for y in tf.split(self.LL, self.batch_size)]
                         for x in self.all_tf_vars}
        self.loss_grads = {x: [tf.gradients(y, x)
                               for y in tf.split(self.loss, self.batch_size)]
                           for x in self.all_tf_vars}

        # Attempt to calculate the gradient within TensorFlow for the entire
        # batch, without moving to the CPU.
        self.tower_gradients = [
            # TODO: This looks horribly inefficient and should be replaced
            # by matrix multiplication soon.
            [(((tf.gather(self.tf_batch_rewards, idx) + tf.gather(self.loss, idx)) * self.LL_grads[x][idx][0] +
               self.loss_grads[x][idx][0]),
              x) for x in self.all_tf_vars]
            for idx in range(self.batch_size)
        ]

        self.avg_gradient = average_gradients(self.tower_gradients)
        self.clipped_avg_gradients, self.grad_norm = \
            tf.clip_by_global_norm([grad for grad, _ in self.avg_gradient],
                                   clip_norm=self.clip_norm)

        self.clipped_avg_gradient = list(zip(
            self.clipped_avg_gradients,
            [var for _, var in self.avg_gradient]
        ))

        # self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        with tf.device(device_cpu):
            # Global step needs to be on the CPU (Why?)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.tf_learning_rate = tf.train.inverse_time_decay(
            self.learning_rate,
            global_step=self.global_step,
            decay_steps=self.decay_steps,
            decay_rate=self.decay_rate
        )

        self.opt = tf.train.AdamOptimizer(learning_rate=self.tf_learning_rate,
                                          beta1=momentum)
        self.sgd_op = self.opt.apply_gradients(self.avg_gradient,
                                               global_step=self.global_step)
        self.sgd_clipped_op = self.opt.apply_gradients(self.clipped_avg_gradient,
                                                       global_step=self.global_step)

        self.sim_opts = sim_opts
        self.src_id = sim_opts.src_id
        self.sess = sess

        # There are other global variables as well, like the ones which the
        # ADAM optimizer uses.
        self.saver = tf.train.Saver(tf.global_variables(),
                                    keep_checkpoint_every_n_hours=0.25,
                                    max_to_keep=100)

        with tf.device(device_cpu):
            tf.contrib.training.add_gradients_summaries(self.avg_gradient)

            for v in self.all_tf_vars:
                variable_summaries(v)

            variable_summaries(self.tf_learning_rate, name='learning_rate')
            variable_summaries(self.loss, name='loss')
            variable_summaries(self.LL, name='LL')
            variable_summaries(self.loss_last, name='loss_last_term')
            variable_summaries(self.LL_last, name='LL_last_term')
            variable_summaries(self.h_states, name='hidden_states')
            variable_summaries(self.LL_log_terms, name='LL_log_terms')
            variable_summaries(self.LL_int_terms, name='LL_int_terms')
            variable_summaries(self.loss_terms, name='loss_terms')
            variable_summaries(tf.cast(self.tf_batch_seq_len, self.tf_dtype),
                               name='batch_seq_len')

            self.tf_merged_summaries = tf.summary.merge_all()

    def initialize(self, finalize=True):
        """Initialize the graph."""
        self.sess.run(tf.global_variables_initializer())
        if finalize:
            # No more nodes will be added to the graph beyond this point.
            # Recommended way to prevent memory leaks afterwards, esp. if the
            # session will be used in a multi-threaded manner.
            # https://stackoverflow.com/questions/38694111/
            self.sess.graph.finalize()

    def _create_exp_broadcaster(self, seed):
        """Create a new exp_broadcaster with the current params."""
        return ExpRecurrentBroadcaster(src_id=self.src_id, seed=seed, trainer=self)

    def run_sim(self, seed, randomize_other_sources=True):
        """Run one simulation and return the dataframe.
        Will be thread-safe and can be called multiple times."""
        run_sim_opts = self.sim_opts.copy()

        if randomize_other_sources:
            run_sim_opts = run_sim_opts.randomize_other_sources(using_seed=seed)

        exp_b = self._create_exp_broadcaster(seed=seed * 3)
        mgr = run_sim_opts.create_manager_with_broadcaster(exp_b)

        # The +1 is to allow us to detect when the number of events is
        # too large to fit in our buffer.
        # Otherwise, we always assume that the sequence didn't produce another
        # event till the end of the time (causing overflows in the survival terms).
        mgr.run_dynamic(max_events=self.max_events + 1)
        return mgr.get_state().get_dataframe()

    def get_feed_dict(self, batch_df, is_test=False, pre_comp_batch_rewards=None):
        """Produce a feed_dict for the given batch."""

        assert all(len(df.sink_id.unique()) == 1 for df in batch_df), "Can only handle one sink at the moment."

        # If the batch sizes are not the same, then the learning cannot happen.
        # However, the forward pass works just as well.
        if not is_test and len(batch_df) != self.batch_size:
            raise ValueError("A training batch should consist of {} simulations, not {}."
                             .format(self.batch_size, len(batch_df)))

        batch_size = len(batch_df)

        full_shape = (batch_size, self.max_events)

        if pre_comp_batch_rewards is not None:
            batch_rewards = np.reshape(pre_comp_batch_rewards, (-1, 1))
        else:
            batch_rewards = np.asarray([
                reward_fn(
                    df=x,
                    reward_kind=self.reward_kind,
                    reward_opts={'K': self.reward_top_k},
                    sim_opts=self.sim_opts
                )
                for x in batch_df
            ])[:, np.newaxis]

        batch_t_deltas = np.zeros(shape=full_shape, dtype=float)

        batch_b_idxes = np.zeros(shape=full_shape, dtype=int)
        batch_ranks = np.zeros(shape=full_shape, dtype=float)
        batch_init_h = np.zeros(shape=(batch_size, self.num_hidden_states), dtype=float)
        batch_last_interval = np.zeros(shape=batch_size, dtype=float)

        # This is one of the reasons why this can handle only one sink right now.
        batch_seq_len = np.asarray([np.minimum(x.shape[0], self.max_events)
                                    for x in batch_df], dtype=int)[:, np.newaxis]

        for idx, df in enumerate(batch_df):
            # They are sorted by time already.
            batch_len = int(batch_seq_len[idx])
            rank_in_tau = RU.rank_of_src_in_df(df=df, src_id=self.src_id).mean(axis=1)
            batch_ranks[idx, 0:batch_len] = rank_in_tau.values[0:batch_len]
            batch_b_idxes[idx, 0:batch_len] = df.src_id.map(self.src_embed_map).values[0:batch_len]
            batch_t_deltas[idx, 0:batch_len] = df.time_delta.values[0:batch_len]
            if batch_len == df.shape[0]:
                # This batch has consumed all the events
                batch_last_interval[idx] = self.sim_opts.end_time - df.t.iloc[-1]
            else:
                batch_last_interval[idx] = df.time_delta[batch_len]

        return {
            self.tf_batch_b_idxes: batch_b_idxes,
            self.tf_batch_rewards: batch_rewards,
            self.tf_batch_seq_len: batch_seq_len,
            self.tf_batch_t_deltas: batch_t_deltas,
            self.tf_batch_ranks: batch_ranks,
            self.tf_batch_init_h: batch_init_h,
            self.tf_batch_last_interval: batch_last_interval,
        }

    def get_batch_grad(self, batch):
        """Returns the true gradient, given a feed dictionary generated by get_feed_dict."""
        feed_dict = self.get_feed_dict(batch)
        batch_rewards = [
            self.reward_fn(
                x,
                reward_kind=self.reward_kind,
                reward_opts={'K': self.reward_top_k},
                sim_opts=self.sim_opts
            )
            for x in batch
        ]

        # The gradients are already summed over the batch dimension.
        LL_grads, losses, loss_grads = self.sess.run([self.LL_grads, self.loss, self.loss_grads],
                                                     feed_dict=feed_dict)

        true_grads = []
        for batch_idx in range(len(batch)):
            reward = batch_rewards[batch_idx]
            loss = losses[batch_idx]
            # TODO: Is there a better way of working with IndexesSlicedValue
            # than converting it to a dense numpy array? Probably not.
            batch_grad = {}
            for x in self.all_tf_vars:
                LL_grad = LL_grads[x][batch_idx][0]

                if hasattr(LL_grad, 'dense_shape'):
                    np_LL_grad = np.zeros(LL_grad.dense_shape, dtype=self.np_dtype)
                    np_LL_grad[LL_grad.indices] = LL_grad.values
                else:
                    np_LL_grad = LL_grad

                loss_grad = loss_grads[x][batch_idx][0]

                if hasattr(loss_grad, 'dense_shape'):
                    np_loss_grad = np.zeros(loss_grad.dense_shape)
                    np_loss_grad[loss_grad.indices] = loss_grad.values
                else:
                    np_loss_grad = loss_grad

                batch_grad[x] = (reward + loss) * np_LL_grad + np_loss_grad

            true_grads.append(batch_grad)

        return true_grads

    def train_many(self, num_iters, init_seed=42,
                   clipping=True, with_summaries=False, with_MP=False):
        """Run one SGD op given a batch of simulation."""

        seed_start = init_seed + self.sess.run(self.global_step) * self.batch_size

        if with_summaries:
            assert self.summary_dir is not None
            os.makedirs(self.summary_dir, exist_ok=True)
            train_writer = tf.summary.FileWriter(self.summary_dir,
                                                 self.sess.graph)

        train_op = self.sgd_op if not clipping else self.sgd_clipped_op

        for epoch in range(num_iters):
            batch = []
            seed_end = seed_start + self.batch_size

            seeds = range(seed_start, seed_end)
            if not with_MP:
                batch = [self.run_sim(seed) for seed in seeds]
                pre_comp_batch_rewards = None
            else:
                batch, pre_comp_batch_rewards = zip(*run_sims_MP(trainer=self, seeds=seeds))

            num_events = [df.event_id.nunique() for df in batch]
            num_our_events = [RU.num_tweets_of(df, sim_opts=self.sim_opts)
                              for df in batch]

            f_d = self.get_feed_dict(
                batch,
                pre_comp_batch_rewards=pre_comp_batch_rewards
            )

            if with_summaries:
                reward, LL, loss, grad_norm, summaries, step, lr, _ = \
                    self.sess.run([self.tf_batch_rewards, self.LL, self.loss,
                                   self.grad_norm, self.tf_merged_summaries,
                                   self.global_step, self.tf_learning_rate,
                                   train_op],
                                  feed_dict=f_d)
                train_writer.add_summary(summaries, step)
            else:
                reward, LL, loss, grad_norm, step, lr, _ = \
                    self.sess.run([self.tf_batch_rewards, self.LL, self.loss,
                                   self.grad_norm, self.global_step,
                                   self.tf_learning_rate, train_op],
                                  feed_dict=f_d)

            mean_LL = np.mean(LL)
            mean_loss = np.mean(loss)
            mean_reward = np.mean(reward)

            print('{} Run {}, LL {:.5f}, loss {:.5f}, Rwd {:.5f}'
                  ', CTG {:.5f}, seeds {}--{}, grad_norm {:.5f}, step = {}'
                  ', lr = {:.5f}, events = {:.2f}/{:.2f}'
                  .format(_now(), epoch, mean_LL, mean_loss,
                          mean_reward, mean_reward + mean_loss,
                          seed_start, seed_end - 1, grad_norm, step, lr,
                          np.mean(num_our_events), np.mean(num_events)))

            chkpt_file = os.path.join(self.save_dir, 'tpprl.ckpt')
            self.saver.save(self.sess, chkpt_file, global_step=self.global_step,)

            # Ready for the next epoch.
            seed_start = seed_end

    def restore(self, restore_dir=None, epoch_to_recover=None):
        """Restores the model from a saved checkpoint."""

        if restore_dir is None:
            restore_dir = self.save_dir

        chkpt = tf.train.get_checkpoint_state(restore_dir)

        if epoch_to_recover is not None:
            suffix = '-{}'.format(epoch_to_recover)
            file = [x for x in chkpt.all_model_checkpoint_paths
                    if x.endswith(suffix)]
            if len(file) < 1:
                raise FileNotFoundError('Epoch {} not found.'
                                        .format(epoch_to_recover))
            self.saver.restore(self.sess, file[0])
        else:
            self.saver.restore(self.sess, chkpt.model_checkpoint_path)

    def calc_u(self, h_states, feed_dict, batch_size, times):
        """Calculate u(t) at the times provided."""
        # TODO: May not work if max_events is hit.

        # This immediately assumes that the
        feed_dict[self.calc_u_h_states] = h_states
        feed_dict[self.calc_u_batch_size] = [batch_size]

        tf_seq_len = np.squeeze(
            self.sess.run(self.tf_batch_seq_len, feed_dict=feed_dict),
            axis=-1
        ) + 1  # +1 to include the survival term.

        assert np.all(tf_seq_len < self.max_events), "Cannot handle events > max_events right now."
        # This will involve changing how the survival term is added, is_own_event is added, etc.

        tf_c_is = (
            [
                self.sess.run(
                    self.calc_u_c_is_init,
                    feed_dict=feed_dict
                )
            ] +
            self.sess.run(
                self.calc_u_c_is_rest,
                feed_dict=feed_dict
            )
        )
        tf_c_is = list(zip(*tf_c_is))

        tf_t_deltas = (
            self.sess.run(
                self.calc_u_t_deltas,
                feed_dict=feed_dict
            ) +
            # Cannot add last_interval at the end of the array because
            # the sequence may have ended before that.
            # Instead, we add tf_t_deltas of 0 to make the length of this
            # array the same as of tf_c_is
            [np.asarray([0.0] * batch_size)]
        )
        tf_t_deltas = list(zip(*tf_t_deltas))

        tf_is_own_event = (
            self.sess.run(
                self.calc_u_is_own_event,
                feed_dict=feed_dict
            ) +
            [np.asarray([False] * batch_size)]
        )

        tf_is_own_event = [
            [bool(x) for x in y]
            for y in list(zip(*tf_is_own_event))
        ]

        last_intervals = self.sess.run(
            self.tf_batch_last_interval,
            feed_dict=feed_dict
        )

        for idx in range(batch_size):
            assert tf_is_own_event[idx][tf_seq_len[idx] - 1]
            tf_is_own_event[idx][tf_seq_len[idx] - 1] = False

            assert tf_t_deltas[idx][tf_seq_len[idx] - 1] == 0
            # tf_t_deltas[idx] is a tuple,
            # we to change it to a list to update a value and then convert
            # back to a tuple.
            old_t_deltas = list(tf_t_deltas[idx])
            old_t_deltas[tf_seq_len[idx] - 1] = last_intervals[idx]
            tf_t_deltas[idx] = tuple(old_t_deltas)

        vt = self.sess.run(self.tf_vt)
        wt = self.sess.run(self.tf_wt)
        bt = self.sess.run(self.tf_bt)

        # TODO: This will break as soon as we move away from zeros
        # as the initial state.
        init_h = np.asarray([0] * self.num_hidden_states)

        sampler_LL = []
        sampler_loss = []

        for idx in range(batch_size):
            # TODO: Split based on the kind of intensity function.

            # The seed doesn't make a difference.
            sampler = exp_sampler.ExpCDFSampler(vt=vt, wt=wt, bt=bt,
                                                init_h=init_h, t_min=self.t_min,
                                                seed=42)
            sampler_LL.append(
                float(
                    sampler.calc_LL(
                        tf_t_deltas[idx][:tf_seq_len[idx]],
                        tf_c_is[idx][:tf_seq_len[idx]],
                        tf_is_own_event[idx][:tf_seq_len[idx]]
                    )
                )
            )
            sampler_loss.append(
                (self.q / 2) *
                float(
                    sampler.calc_quad_loss(
                        tf_t_deltas[idx][:tf_seq_len[idx]],
                        tf_c_is[idx][:tf_seq_len[idx]]
                    )
                )
            )

        u = np.zeros((batch_size, times.shape[0]), dtype=float)

        for batch_idx in range(batch_size):
            abs_time = 0
            abs_idx = 0
            c = tf_c_is[batch_idx][0]

            for time_idx, t in enumerate(times):
                # We do not wish to update the c for the last survival interval.
                # Hence, the -1 in len(tf_t_deltas[batch_idx] - 1
                while abs_idx < len(tf_t_deltas[batch_idx]) - 1 and abs_time + tf_t_deltas[batch_idx][abs_idx] < t:
                    abs_time += tf_t_deltas[batch_idx][abs_idx]
                    abs_idx += 1
                    c = tf_c_is[batch_idx][abs_idx]

                # TODO: Split based on the kind of intensity function.
                u[batch_idx, time_idx] = np.exp(c + wt * (t - abs_time))

        return {
            'c_is': tf_c_is,
            'is_own_event': tf_is_own_event,
            't_deltas': tf_t_deltas,
            'seq_len': tf_seq_len,

            'vt': vt,
            'wt': wt,
            'bt': bt,

            'LL': sampler_LL,
            'loss': sampler_loss,

            'times': times,
            'u': u,
        }


# ###################################
# # This is an attempt to remain more faithful to the theory proposed in the paper.
# # Here, we will do model predictive control of sorts by determining what is the
# # current intensity of each broadcaster and the predicted mean next event time.
# #  - Issue: the mean may be infinity for certain kinds of broadcasters.
# #           - Maybe consider the median instead, which may also be infinite?
# #           - TODO: Verify because ∫tf(t)dt does not look like it will go to
# #             infinity if f(t) decays too quickly?
# #           - Model infinity by using an indicator variable (more elegant)
# #           - This is especially important if we use the RMTPP formulation since
# #             the model can produce infinite mean times? (See TODO above).
# #           -
#
# class ExpPredBroadcaster(OM.Broadcaster):
#     """This is a broadcaster which follows the intensity function as defined by
#     RMTPP paper and updates the hidden state upon receiving each event by predicting
#     the (known) intensity function of the next broadcaster.
#     """
#
#     @Deco.optioned()
#     def __init__(self, src_id, seed, trainer, t_min=0):
#         super(ExpPredBroadcaster, self).__init__(src_id, seed)
#         self.init = False
#
#         self.trainer = trainer
#
#         params = Deco.Options(**self.trainer.sess.run({
#             'wt': trainer.tf_wt,
#             'vt': trainer.tf_vt,
#             'bt': trainer.tf_bt,
#             'init_h': trainer.tf_h
#         }))
#
#         self.cur_h = params.init_h
#
#         self.exp_sampler = exp_sampler.ExpCDFSampler(_opts=params,
#                                                      t_min=t_min,
#                                                      seed=seed + 1)
#
#     def update_hidden_state(self, src_id, time_delta):
#         """Returns the hidden state after a post by src_id and time delta."""
#         # Best done using self.sess.run here
#         # Actually best done by requesting the trainer to change the state.
#         # This couples the broadcaster and the trainer together rather badly,
#         # doesn't it? Maybe not so much.
#         r_t = self.state.get_wall_rank(self.src_id, self.sink_ids, dict_form=False)
#         return self.trainer._update_hidden_state(src_id, time_delta, r_t)
#
#     def get_next_interval(self, event):
#         if not self.init:
#             self.init = True
#             # Nothing special to do for the first event.
#
#         self.state.apply_event(event)
#
#         if event is None:
#             # This is the first event. Post immediately to join the party?
#             # Or hold off?
#             return self.exp_sampler.generate_sample()
#         else:
#             self.cur_h = self.update_hidden_state(event.src_id, event.time_delta)
#             next_post_time = self.exp_sampler.register_event(
#                 event.cur_time,
#                 self.cur_h,
#                 own_event=event.src_id == self.src_id
#             )
#             next_delta = next_post_time - self.last_self_event_time
#             # print(next_delta)
#             assert next_delta >= 0
#             return next_delta
#
#
# OM.SimOpts.registerSource('ExpPredBroadcaster', ExpPredBroadcaster)
#
#
# class ExpPredTrainer:
#
#     def _update_hidden_state(self, src_id, time_delta, r_t):
#
#         feed_dict = {
#             self.tf_b_idx: np.asarray([self.trainer.src_embed_map[src_id]]),
#             self.tf_t_delta: np.asarray([time_delta]).reshape(-1),
#             self.tf_h: self.cur_h,
#             self.tf_rank: np.asarray([np.mean(r_t)]).reshape(-1)
#         }
#         return self.sess.run(self.trainer.tf_h_next,
#                                      feed_dict=feed_dict)
#
#     @Deco.optioned()
#     def __init__(self, Wm, Wl, Wdt, Wt, Wr, Bh, vt, wt, bt, init_h, init_l, init_pred_dt,
#                  sess, sim_opts, scope=None, t_min=0, batch_size=16, max_events=100):
#         """Initialize the trainer with the policy parameters."""
#
#         self.src_embed_map = {x.src_id: idx + 1
#                               for idx, x in enumerate(sim_opts.create_other_sources())}
#         self.src_embed_map[sim_opts.src_id] = 0
#
#         self.batch_size = batch_size
#         self.max_events = max_events
#         self.num_hidden_states = init_h.shape[0]
#         self.sim_opts = sim_opts
#         self.src_id = sim_opts.src_id
#         self.sess = sess
#
#         # init_h = np.reshape(init_h, (-1, 1))
#         Bh = np.reshape(Bh, (-1, 1))
#         # TODO: May need to do the same for init_l, init_pred_dt
#
#         self.scope = scope or type(self).__name__
#
#         # TODO: Create all these variables on the CPU and the training vars on the GPU
#         # by using tf.device explicitly?
#         with tf.variable_scope(self.scope):
#             with tf.variable_scope("hidden_state"):
#                 self.tf_Wm  = tf.get_variable(name="Wm", shape=Wm.shape,
#                                               initializer=tf.constant_initializer(Wm))
#                 self.tf_Wl  = tf.get_variable(name="Wl", shape=Wl.shape,
#                                               initializer=tf.constant_initializer(Wl))
#                 self.tf_Wdt = tf.get_variable(name="Wdt", shape=Wdt.shape,
#                                               initializer=tf.constant_initializer(Wdt))
#                 self.tf_Wt  = tf.get_variable(name="Wt", shape=Wt.shape,
#                                               initializer=tf.constant_initializer(Wt))
#                 self.tf_Wr  = tf.get_variable(name="Wr", shape=Wr.shape,
#                                               initializer=tf.constant_initializer(Wr))
#                 self.tf_Bh  = tf.get_variable(name="Bh", shape=Bh.shape,
#                                              initializer=tf.constant_initializer(Bh))
#
#                 self.tf_l   = tf.get_variable(name="l", shape=init_l.shape,
#                                               initializer=tf.constant_initializer(init_l))
#                 self.tf_pred_dt  = tf.get_variable(name="pred_dt", shape=init_pred_dt.shape,
#                                               initializer=tf.constant_initializer(init_pred_dt))
#                 self.tf_h = tf.get_variable(name="h", shape=(self.num_hidden_states, 1),
#                                             initializer=tf.constant_initializer(init_h))
#                 self.tf_b_idx = tf.placeholder(name="b_idx", shape=1, dtype=tf.int32)
#                 self.tf_t_delta = tf.placeholder(name="t_delta", shape=1, dtype=tf.float32)
#                 self.tf_rank = tf.placeholder(name="rank", shape=1, dtype=tf.float32)
#
#                 # TODO: The transposes hurt my eyes and the GPU efficiency.
#                 self.tf_h_next = tf.nn.relu(
#                     tf.transpose(
#                         tf.nn.embedding_lookup(self.tf_Wm, self.tf_b_idx, name="b_embed")
#                     ) +
#                     tf.matmul(self.tf_Wl, self.tf_l) +
#                     tf.matmul(self.tf_Wdt, self.tf_pred_dt) +
#                     self.tf_Wr * self.tf_rank +
#                     self.tf_Wt * self.tf_t_delta +
#                     self.tf_Bh,
#                     name="h_next"
#                 )
#
#             with tf.variable_scope("output"):
#                 self.tf_bt = tf.get_variable(name="bt", shape=bt.shape,
#                                              initializer=tf.constant_initializer(bt))
#                 self.tf_vt = tf.get_variable(name="vt", shape=vt.shape,
#                                              initializer=tf.constant_initializer(vt))
#                 self.tf_wt = tf.get_variable(name="wt", shape=wt.shape,
#                                              initializer=tf.constant_initializer(wt))
#                 # self.tf_t_delta = tf.placeholder(name="t_delta", shape=1, dtype=tf.float32)
#                 # self.tf_u_t = tf.exp(
#                 #     tf.tensordot(self.tf_vt, self.tf_h, axes=1) +
#                 #     self.tf_t_delta * self.tf_wt +
#                 #     self.tf_bt,
#                 #     name="u_t"
#                 # )
#
#             # Create a large dynamic_rnn kind of network which can calculate
#             # the gradients for a given given batch of simulations.
#             with tf.variable_scope("training"):
#                 self.tf_batch_rewards = tf.placeholder(name="rewards",
#                                                  shape=(batch_size, 1),
#                                                  dtype=tf.float32)
#                 self.tf_batch_t_deltas = tf.placeholder(name="t_deltas",
#                                                   shape=(batch_size, max_events),
#                                                   dtype=tf.float32)
#                 self.tf_batch_pred_dt = tf.placeholder(name="pred_dt",
#                                                        shape=(batch_size,
#                                                               len(self.sim_opts.other_sources),
#                                                               max_events),
#                                                        dtype=tf.float32)
#                 self.tf_batch_l       = tf.placeholder(name="ls",
#                                                        shape=(batch_size,
#                                                               len(self.sim_opts.other_sources),
#                                                               max_events),
#                                                        dtype=tf.float32)
#                 self.tf_batch_b_idxes = tf.placeholder(name="b_idxes",
#                                                  shape=(batch_size, max_events),
#                                                  dtype=tf.int32)
#                 self.tf_batch_ranks = tf.placeholder(name="ranks",
#                                                shape=(batch_size, max_events),
#                                                dtype=tf.float32)
#                 self.tf_batch_seq_len = tf.placeholder(name="seq_len",
#                                                  shape=(batch_size, 1),
#                                                  dtype=tf.int32)
#                 self.tf_batch_last_interval = tf.placeholder(name="last_interval",
#                                                              shape=batch_size,
#                                                              dtype=tf.float32)
#
#                 self.tf_batch_init_h = tf_batch_h_t = tf.zeros(name="init_h",
#                                               shape=(batch_size, self.num_hidden_states),
#                                               dtype=tf.float32)
#
#                 self.LL = tf.zeros(name="log_likelihood", dtype=tf.float32, shape=(batch_size))
#                 self.loss = tf.zeros(name="loss", dtype=tf.float32, shape=(batch_size))
#
#                 t_0 = tf.zeros(name="event_time", shape=batch_size, dtype=tf.float32)
#
#                 def batch_u_theta(batch_t_deltas):
#                     return tf.exp(
#                             tf.matmul(tf_batch_h_t, self.tf_vt) +
#                             self.tf_wt * tf.expand_dims(batch_t_deltas, 1) +
#                             self.tf_bt
#                         )
#
#
#                 # TODO: Convert this to a tf.while_loop, perhaps.
#                 # The performance benefit is debatable.
#                 for evt_idx in range(max_events):
#                     tf_batch_h_t = tf.where(
#                         tf.tile(evt_idx <= self.tf_batch_seq_len, [1, self.num_hidden_states]),
#                         tf.nn.relu(
#                             tf.nn.embedding_lookup(self.tf_Wm,
#                                                    self.tf_batch_b_idxes[:, evt_idx]) +
#                             tf.matmul(tf_batch_h_t, self.tf_Wh, transpose_b=True) +
#                             tf.matmul(tf.expand_dims(self.tf_batch_ranks[:, evt_idx], 1),
#                                       self.tf_Wr, transpose_b=True) +
#                             tf.matmul(tf.expand_dims(self.tf_batch_t_deltas[:, evt_idx], 1),
#                                       self.tf_Wt, transpose_b=True) +
#                             tf.tile(tf.transpose(self.tf_Bh), [batch_size, 1])
#                         ),
#                         tf.zeros(dtype=tf.float32, shape=(batch_size, self.num_hidden_states))
#                         # The gradient of a constant w.r.t. a variable is None or 0
#                     )
#                     tf_batch_u_theta = tf.where(
#                         evt_idx <= self.tf_batch_seq_len,
#                         batch_u_theta(self.tf_batch_t_deltas[:, evt_idx]),
#                         tf.zeros(dtype=tf.float32, shape=(batch_size, 1))
#                     )
#
#                     self.LL += tf.where(tf.squeeze(evt_idx <= self.tf_batch_seq_len),
#                                     tf.where(tf.equal(self.tf_batch_b_idxes[:, evt_idx], 0),
#                                         tf.squeeze(tf.log(tf_batch_u_theta)),
#                                         tf.zeros(dtype=tf.float32, shape=batch_size)) +
#                                     (1 / self.tf_wt) * tf.squeeze(
#                                         batch_u_theta(t_0) -
#                                         tf_batch_u_theta
#                                     ),
#                                     tf.zeros(dtype=tf.float32, shape=batch_size))
#
#                     self.loss += tf.where(tf.squeeze(evt_idx <= self.tf_batch_seq_len),
#                                     -(1 / (2 * self.tf_wt)) * tf.squeeze(
#                                         tf.square(batch_u_theta(t_0)) -
#                                         tf.square(tf_batch_u_theta)
#                                     ),
#                                     tf.zeros(dtype=tf.float32, shape=(batch_size)))
#
#         # Here, outside the loop, add the survival term for the batch to
#         # both the loss and to the LL.
#         self.LL += (1 / self.tf_wt) * tf.squeeze(
#             batch_u_theta(t_0) - batch_u_theta(self.tf_batch_last_interval)
#         )
#         self.loss += - (1 / (2 * self.tf_wt)) * tf.squeeze(
#             tf.square(batch_u_theta(t_0)) - tf.square(self.tf_batch_last_interval)
#         )
#
#         # sim_feed_dict = {
#         #     self.tf_Wm: Wm,
#         #     self.tf_Wh: Wh,
#         #     self.tf_Wt: Wt,
#         #     self.tf_Bh: Bh,
#
#         #     self.tf_bt: bt,
#         #     self.tf_vt: vt,
#         #     self.tf_wt: wt,
#         # }
#
#     def initialize(self):
#         """Initialize the graph."""
#         self.sess.run(tf.global_variables_initializer())
#         # No more nodes will be added to the graph beyond this point.
#         # Recommended way to prevent memory leaks afterwards, esp. if the
#         # session will be used in a multi-threaded manner.
#         # https://stackoverflow.com/questions/38694111/
#         self.sess.graph.finalize()
#
#     def _create_exp_broadcaster(self, seed):
#         """Create a new exp_broadcaster with the current params."""
#         return ExpPredBroadcaster(src_id=self.src_id, seed=seed, trainer=self)
#
#     def run_sim(self, seed):
#         """Run one simulation and return the dataframe.
#         Will be thread-safe and can be called multiple times."""
#         run_sim_opts = self.sim_opts.update({})
#         exp_b = self._create_exp_broadcaster(seed=seed * 3)
#
#         mgr = run_sim_opts.create_manager_with_broadcaster(exp_b)
#         mgr.run_dynamic()
#         return mgr.get_state().get_dataframe()
#
#     def reward_fn(self, df):
#         """Calculate the reward for a given trajectory."""
#         rank_in_tau = RU.rank_of_src_in_df(df=df, src_id=self.src_id).mean(axis=1)
#         rank_dt = np.diff(np.concatenate([rank_in_tau.index.values,
#                                           [self.sim_opts.end_time]]))
#         return np.sum((rank_in_tau ** 2) * rank_dt)
#
#     def get_feed_dict(self, batch_df):
#         """Produce a feed_dict for the given batch."""
#         assert all(len(df.sink_id.unique()) == 1 for df in batch_df), "Can only handle one sink at the moment."
#         assert len(batch_df) == self.batch_size, "The batch should consist of {} simulations, not {}.".format(self.batch_size, len(batch_df))
#
#         full_shape = (self.batch_size, self.max_events)
#
#         batch_rewards = np.asarray([self.reward_fn(x) for x in batch_df])[:, np.newaxis]
#         batch_t_deltas = np.zeros(shape=full_shape, dtype=float)
#
#         batch_b_idxes = np.zeros(shape=full_shape, dtype=int)
#         batch_ranks = np.zeros(shape=full_shape, dtype=float)
#         batch_seq_len = np.asarray([np.minimum(x.shape[0], self.max_events) for x in batch_df], dtype=int)[:, np.newaxis]
#         batch_init_h = np.zeros(shape=(self.batch_size, self.num_hidden_states), dtype=float)
#
#         batch_last_interval = np.zeros(shape=self.batch_size, dtype=float)
#
#         for idx, df in enumerate(batch_df):
#             # They are sorted by time already.
#             batch_len = int(batch_seq_len[idx])
#             rank_in_tau = RU.rank_of_src_in_df(df=df, src_id=self.src_id).mean(axis=1)
#             batch_ranks[idx, 0:batch_len] = rank_in_tau.values[0:batch_len]
#             batch_b_idxes[idx, 0:batch_len] = df.src_id.map(self.src_embed_map).values[0:batch_len]
#             batch_t_deltas[idx, 0:batch_len] = df.time_delta.values[0:batch_len]
#             if batch_len == df.shape[0]:
#                 # This batch has consumed all the events
#                 batch_last_interval[idx] = self.sim_opts.end_time - df.t.iloc[-1]
#             else:
#                 batch_last_interval[idx] = df.time_delta[batch_len]
#
#         return {
#             self.tf_batch_b_idxes: batch_b_idxes,
#             self.tf_batch_rewards: batch_rewards,
#             self.tf_batch_seq_len: batch_seq_len,
#             self.tf_batch_t_deltas: batch_t_deltas,
#             self.tf_batch_ranks: batch_ranks,
#             self.tf_batch_init_h: batch_init_h,
#             self.tf_batch_last_interval: batch_last_interval,
#         }
#
#     def calc_grad(self, df):
#         """Calculate the gradient with respect to a certain run."""
#         # 1. Keep updating the u_{\theta}(t) in the tensorflow graph starting from
#         #    t = 0 with each event and calculating the gradient.
#         # 2. Finally, sum together the gradient calculated for the complete
#         #    sequence.
#
#         # Actually, we can calculate the gradient analytically in this case.
#         # Not quite: we can integrate analytically, but differentiation is
#         # still a little tricky because of the hidden state.
#         R_tau = self.reward_fn(df, src_id=self.src_id)
#
#         # Loop over the events.
#         unique_events = df.groupby('event_id').first()
#         for t_delta, src_id in unique_events[['time_delta', 'src_id']].values:
#             # TODO
#             pass
#

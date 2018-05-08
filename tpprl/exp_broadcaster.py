import numpy as np
import os
import redqueen.utils as RU
import tensorflow as tf
import decorated_options as Deco
import exp_sampler
import multiprocessing as MP
import warnings

from utils import variable_summaries, _now, average_gradients, get_test_dfs

from cells import TPPRExpCell, TPPRExpCellStacked

SAVE_DIR = 'tpprl-log'
os.makedirs(SAVE_DIR, exist_ok=True)

R_2_REWARD = 'r_2_reward'
TOP_K_REWARD = 'top_k_reward'
TARGET_TOP_K_REWARD = 'target_reward'


def reward_fn(df, reward_kind, reward_opts, sim_opts):
    """Calculate the reward for this trajectory."""
    if reward_kind == R_2_REWARD:
        return RU.int_r_2(df, sim_opts)
    elif reward_kind == TOP_K_REWARD:
        return -RU.time_in_top_k(df, sim_opts=sim_opts, K=reward_opts['K'])
    elif reward_kind == TARGET_TOP_K_REWARD:
        target_loss = (reward_opts['target'] - RU.num_tweets_of(df, broadcaster_id=sim_opts.src_id)) ** 2
        return -RU.time_in_top_k(df, sim_opts=sim_opts, K=reward_opts['K']) + reward_opts['s'] * target_loss
    else:
        raise NotImplementedError('{} reward function is not implemented.'
                                  .format(reward_kind))


def make_reward_opts(trainer):
    """Collects all reward related options into a dictionary."""
    return {
        'K': trainer.reward_top_k,
        'target': trainer.reward_episode_target,
        's': trainer.reward_target_weight,
    }


def get_test_perf(trainer, seeds, t_min=None, t_max=None):
    """Takes the trainer and performs simulations for the given set of seeds."""

    if t_min is not None or t_max is not None:
        warnings.warn('t_min and t_max are deprecated.')

    seeds = list(seeds)
    t_min, t_max = trainer.t_min, trainer.t_max

    dfs = get_test_dfs(trainer, seeds, t_min, t_max)
    f_d = trainer.get_feed_dict(dfs, is_test=True)
    h_states = trainer.sess.run(trainer.h_states, feed_dict=f_d)

    times = np.arange(t_min, t_max, (t_max - t_min) / 5000)
    u_data = trainer.calc_u(h_states=h_states, feed_dict=f_d,
                            batch_size=len(seeds), times=times)
    rewards = [reward_fn(df=df,
                         reward_kind=trainer.reward_kind,
                         reward_opts=make_reward_opts(trainer),
                         sim_opts=trainer.sim_opts)
               for df in dfs]
    u_data['rewards'] = rewards
    return u_data


def _worker_sim(params):
    """Worker for the parallel simulation runner."""
    rl_b_args, seed = params
    rl_b_opts = Deco.Options(**rl_b_args)

    # Need to select the sampler somehow.
    exp_b = exp_sampler.ExpRecurrentBroadcasterMP(
        _opts=rl_b_opts,
        seed=seed * 3
    )
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
        'max_events': trainer.abs_max_events,
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
        'reward_opts': make_reward_opts(trainer),
    }

    # return [_worker_sim((rl_b_args, seed)) for seed in seeds]

    with MP.Pool(processes=processes) as pool:
        return pool.map(_worker_sim, [(rl_b_args, seed) for seed in seeds])


def mk_def_exp_recurrent_trainer_opts(num_other_broadcasters, hidden_dims,
                                      seed=42, num_followers=1, **kwargs):
    """Make default option set."""
    RS  = np.random.RandomState(seed=seed)

    def_exp_recurrent_trainer_opts = Deco.Options(
        t_min=0,
        scope=None,
        with_dynamic_rnn=True,
        decay_steps=100,
        decay_rate=0.001,
        num_hidden_states=hidden_dims,
        learning_rate=.01,
        clip_norm=1.0,

        num_followers=num_followers,

        Wh=RS.randn(hidden_dims, hidden_dims) * 0.1 + np.diag(np.ones(hidden_dims)),  # Careful initialization
        Wm=RS.randn(num_other_broadcasters + 1, hidden_dims),
        Wr=RS.randn(hidden_dims, num_followers),
        Wt=RS.randn(hidden_dims, 1),
        Bh=RS.randn(hidden_dims, 1),

        vt=RS.randn(hidden_dims, 1),
        wt=np.abs(RS.rand(1)) * -1,
        bt=np.abs(RS.randn(1)),

        # The graph execution time depends on this parameter even though each
        # trajectory may contain much fewer events. So it is wise to set
        # it such that it is just above the total number of events likely
        # to be seen.
        momentum=0.9,
        max_events=5000,
        batch_size=16,

        device_cpu='/cpu:0',
        device_gpu='/gpu:0',
        only_cpu=False,

        save_dir=SAVE_DIR,

        # Expected: './tpprl.summary/train-{}/'.format(run)
        summary_dir=None,

        reward_kind=R_2_REWARD,

        # May need a better way of passing reward_fn arguments
        reward_top_k=-1,
        reward_episode_target=-1,
        reward_target_weight=0,

        decay_q_rate=0.0,

        # Whether or not to use the advantage formulation.
        with_advantage=True,
    )

    return def_exp_recurrent_trainer_opts.set(**kwargs)


class ExpRecurrentTrainer:
    @Deco.optioned()
    def __init__(self, Wm, Wh, Wt, Wr, Bh, vt, wt, bt, num_hidden_states,
                 sess, sim_opts, scope, t_min, batch_size, max_events,
                 learning_rate, clip_norm, with_dynamic_rnn,
                 summary_dir, save_dir, decay_steps, decay_rate, momentum,
                 device_cpu, device_gpu, only_cpu, with_advantage,
                 num_followers, decay_q_rate,
                 reward_top_k, reward_kind,
                 reward_episode_target, reward_target_weight):
        """Initialize the trainer with the policy parameters."""

        self.reward_top_k = reward_top_k
        self.reward_kind = reward_kind
        self.reward_episode_target = reward_episode_target
        self.reward_target_weight = reward_target_weight
        self.decay_q_rate = decay_q_rate

        self.t_min = t_min
        self.t_max = sim_opts.end_time

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
        self.tf_max_events = None
        self.num_followers = num_followers

        self.abs_max_events = max_events
        self.num_hidden_states = num_hidden_states

        # init_h = np.reshape(init_h, (-1, 1))
        Bh = np.reshape(Bh, (-1, 1))

        self.scope = scope or type(self).__name__

        var_device = device_cpu if only_cpu else device_gpu

        # self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        with tf.device(device_cpu):
            # Global step needs to be on the CPU (Why?)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

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
                self.tf_rank = tf.placeholder(name='rank', shape=(num_followers, 1), dtype=self.tf_dtype)

                self.tf_h_next = tf.nn.tanh(
                    tf.transpose(
                        tf.nn.embedding_lookup(self.tf_Wm, self.tf_b_idx, name='b_embed')
                    ) +
                    tf.matmul(self.tf_Wh, self.tf_h) +
                    tf.matmul(self.tf_Wr, self.tf_rank) +
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
                                                        shape=(self.tf_batch_size, self.tf_max_events),
                                                        dtype=self.tf_dtype)
                self.tf_batch_b_idxes = tf.placeholder(name='b_idxes',
                                                       shape=(self.tf_batch_size, self.tf_max_events),
                                                       dtype=tf.int32)
                self.tf_batch_ranks = tf.placeholder(name='ranks',
                                                     shape=(self.tf_batch_size, self.tf_max_events, self.num_followers),
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

                # t_0 = tf.zeros(name='event_time', shape=(inf_batch_size,), dtype=self.tf_dtype)

                # def batch_u_theta(batch_t_deltas):
                #     return tf.exp(
                #         tf.matmul(tf_batch_h_t, self.tf_vt) +
                #         self.tf_wt * tf.expand_dims(batch_t_deltas, 1) +
                #         self.tf_bt
                #     )

                assert with_dynamic_rnn, "The non-dynamic-RNN path has been removed."

                # Un-stacked version (for ease of debugging)

                with tf.name_scope('batched'):
                    self.rnn_cell = TPPRExpCell(
                        hidden_state_size=(None, self.num_hidden_states),
                        output_size=[self.num_hidden_states] + [1] * 3,
                        src_id=sim_opts.src_id,
                        tf_dtype=self.tf_dtype,
                        Wm=self.tf_Wm, Wr=self.tf_Wr, Wh=self.tf_Wh,
                        Wt=self.tf_Wt, Bh=self.tf_Bh,
                        wt=self.tf_wt, vt=self.tf_vt, bt=self.tf_bt
                    )

                    ((self.h_states, LL_log_terms, LL_int_terms, loss_terms), tf_batch_h_t) = tf.nn.dynamic_rnn(
                        self.rnn_cell,
                        inputs=(tf.expand_dims(self.tf_batch_b_idxes, axis=-1),
                                self.tf_batch_ranks,
                                tf.expand_dims(self.tf_batch_t_deltas, axis=-1)),
                        sequence_length=tf.squeeze(self.tf_batch_seq_len, axis=-1),
                        dtype=self.tf_dtype,
                        initial_state=self.tf_batch_init_h
                    )
                    self.LL_log_terms = tf.squeeze(LL_log_terms, axis=-1)
                    self.LL_int_terms = tf.squeeze(LL_int_terms, axis=-1)
                    self.loss_terms = tf.squeeze(loss_terms, axis=-1)

                    # Survival terms for LL and loss.
                    # self.LL_last = -(1 / self.tf_wt) * tf.squeeze(
                    #     batch_u_theta(self.tf_batch_last_interval) - batch_u_theta(t_0)
                    # )

                    # self.loss_last = (1 / (2 * self.tf_wt)) * tf.squeeze(
                    #     tf.square(batch_u_theta(self.tf_batch_last_interval)) -
                    #     tf.square(batch_u_theta(t_0))
                    # )

                    self.LL_last = self.rnn_cell.last_LL(tf_batch_h_t, self.tf_batch_last_interval)
                    self.loss_last = self.rnn_cell.last_loss(tf_batch_h_t, self.tf_batch_last_interval)

                    self.LL = tf.reduce_sum(self.LL_log_terms, axis=1) - tf.reduce_sum(self.LL_int_terms, axis=1) + self.LL_last
                    self.loss = (self.q / 2) * (tf.reduce_sum(self.loss_terms, axis=1) + self.loss_last) * tf.pow(tf.cast(self.global_step, self.tf_dtype), self.decay_q_rate)

                # Stacked version (for performance)

                with tf.name_scope('stacked'):

                    (self.Wm_mini, self.Wr_mini, self.Wh_mini,
                     self.Wt_mini, self.Bh_mini, self.wt_mini,
                     self.vt_mini, self.bt_mini) = [
                         tf.stack(x, name=name)
                         for x, name in zip(
                                 zip(*[
                                     (tf.identity(self.tf_Wm), tf.identity(self.tf_Wr),
                                      tf.identity(self.tf_Wh), tf.identity(self.tf_Wt),
                                      tf.identity(self.tf_Bh), tf.identity(self.tf_wt),
                                      tf.identity(self.tf_vt), tf.identity(self.tf_bt))
                                     for _ in range(self.batch_size)
                                 ]),
                                 ['Wm', 'Wr', 'Wh', 'Wt', 'Bh', 'wt', 'vt', 'bt']
                         )
                    ]

                    self.rnn_cell_stack = TPPRExpCellStacked(
                        hidden_state_size=(None, self.num_hidden_states),
                        output_size=[self.num_hidden_states] + [1] * 3,
                        src_id=sim_opts.src_id,
                        tf_dtype=self.tf_dtype,
                        Wm=self.Wm_mini, Wr=self.Wr_mini,
                        Wh=self.Wh_mini, Wt=self.Wt_mini,
                        Bh=self.Bh_mini, wt=self.wt_mini,
                        vt=self.vt_mini, bt=self.bt_mini
                    )

                    ((self.h_states_stack, LL_log_terms_stack, LL_int_terms_stack, loss_terms_stack), tf_batch_h_t_mini) = tf.nn.dynamic_rnn(
                        self.rnn_cell_stack,
                        inputs=(tf.expand_dims(self.tf_batch_b_idxes, axis=-1),
                                self.tf_batch_ranks,
                                tf.expand_dims(self.tf_batch_t_deltas, axis=-1)),
                        sequence_length=tf.squeeze(self.tf_batch_seq_len, axis=-1),
                        dtype=self.tf_dtype,
                        initial_state=self.tf_batch_init_h
                    )

                    # In this version, the stacking had been done by creating
                    # batch_size RNNCells. However, because the multiplication
                    # inside the cells now has to be done element by element,
                    # this version is way too slow, both in the forward and the
                    # backward pass.
                    #
                    # (tf_batch_b_idxes_mini, tf_batch_t_deltas_mini,
                    #  tf_batch_ranks_mini, tf_batch_seq_len_mini,
                    #  tf_batch_last_interval_mini, tf_batch_init_h_mini) = [
                    #      tf.split(tensor, self.batch_size, axis=0)
                    #      for tensor in [self.tf_batch_b_idxes,
                    #                     self.tf_batch_t_deltas,
                    #                     self.tf_batch_ranks,
                    #                     self.tf_batch_seq_len,
                    #                     self.tf_batch_last_interval,
                    #                     self.tf_batch_init_h]
                    # ]

                    # h_states_stack = []
                    # LL_log_terms_stack = []
                    # LL_int_terms_stack = []
                    # loss_terms_stack = []

                    # LL_last_term_stack = []
                    # loss_last_term_stack = []


                    # for idx in range(self.batch_size):
                    #     rnn_cell = TPPRExpCell(
                    #         hidden_state_size=(1, self.num_hidden_states),
                    #         output_size=[self.num_hidden_states] + [1] * 3,
                    #         src_id=sim_opts.src_id,
                    #         tf_dtype=self.tf_dtype,
                    #         Wm=self.Wm_mini[idx], Wr=self.Wr_mini[idx],
                    #         Wh=self.Wh_mini[idx], Wt=self.Wt_mini[idx],
                    #         Bh=self.Bh_mini[idx], wt=self.wt_mini[idx],
                    #         vt=self.vt_mini[idx], bt=self.bt_mini[idx]
                    #     )

                    #     ((h_states_mini, LL_log_terms_mini, LL_int_terms_mini, loss_terms_mini), tf_batch_h_t_mini) = tf.nn.dynamic_rnn(
                    #         rnn_cell,
                    #         inputs=(tf.expand_dims(tf_batch_b_idxes_mini[idx], axis=-1),
                    #                 tf.expand_dims(tf_batch_ranks_mini[idx], axis=-1),
                    #                 tf.expand_dims(tf_batch_t_deltas_mini[idx], axis=-1)),
                    #         sequence_length=tf.squeeze(tf_batch_seq_len_mini[idx], axis=-1),
                    #         dtype=self.tf_dtype,
                    #         initial_state=tf_batch_init_h_mini[idx]
                    #     )

                    #     h_states_stack.append(h_states_mini[0])
                    #     LL_log_terms_stack.append(LL_log_terms_mini[0])
                    #     LL_int_terms_stack.append(LL_int_terms_mini[0])
                    #     loss_terms_stack.append(loss_terms_mini[0])

                    # self.h_states_stack = tf.stack(h_states_stack)

                    # self.LL_log_terms_stack = tf.squeeze(tf.stack(LL_log_terms_stack), axis=-1)
                    # self.LL_int_terms_stack = tf.squeeze(tf.stack(LL_int_terms_stack), axis=-1)
                    # self.loss_terms_stack = tf.squeeze(tf.stack(loss_terms_stack), axis=-1)

                    self.LL_log_terms_stack = tf.squeeze(LL_log_terms_stack, axis=-1)
                    self.LL_int_terms_stack = tf.squeeze(LL_int_terms_stack, axis=-1)
                    self.loss_terms_stack = tf.squeeze(loss_terms_stack, axis=-1)

                    # LL_last_term_stack = rnn_cell.last_LL(tf_batch_h_t_mini, self.tf_batch_last_interval)
                    # loss_last_term_stack = rnn_cell.last_loss(tf_batch_h_t_mini, self.tf_batch_last_interval)

                    self.LL_last_term_stack = self.rnn_cell_stack.last_LL(tf_batch_h_t_mini, self.tf_batch_last_interval)
                    self.loss_last_term_stack = self.rnn_cell_stack.last_loss(tf_batch_h_t_mini, self.tf_batch_last_interval)

                    self.LL_stack = (tf.reduce_sum(self.LL_log_terms_stack, axis=1) - tf.reduce_sum(self.LL_int_terms_stack, axis=1)) + self.LL_last_term_stack
                    self.loss_stack = (self.q / 2) * (tf.reduce_sum(self.loss_terms_stack, axis=1) + self.loss_last_term_stack) * tf.pow(tf.cast(self.global_step, self.tf_dtype), self.decay_q_rate)


            with tf.name_scope('calc_u'):
                # These are operations needed to calculate u(t) in post-processing.
                # These can be done entirely in numpy-space, but since we have a
                # version in tensorflow, they have been moved here to avoid
                # memory leaks.
                # Otherwise, new additions to the graph were made whenever the
                # function calc_u was called.

                self.calc_u_h_states = tf.placeholder(
                    name='calc_u_h_states',
                    shape=(self.tf_batch_size, self.tf_max_events, self.num_hidden_states),
                    dtype=self.tf_dtype
                )
                self.calc_u_batch_size = tf.placeholder(
                    name='calc_u_batch_size',
                    shape=(None,),
                    dtype=tf.int32
                )

                self.calc_u_c_is_init = tf.matmul(self.tf_batch_init_h, self.tf_vt) + self.tf_bt
                self.calc_u_c_is_rest = tf.squeeze(
                    tf.matmul(
                        self.calc_u_h_states,
                        tf.tile(
                            tf.expand_dims(self.tf_vt, 0),
                            [self.calc_u_batch_size[0], 1, 1]
                        )
                    ) + self.tf_bt,
                    axis=-1,
                    name='calc_u_c_is_rest'
                )

                self.calc_u_is_own_event = tf.equal(self.tf_batch_b_idxes, 0)

        # TODO: The all_tf_vars and all_mini_vars MUST be kept in sync.
        self.all_tf_vars = [self.tf_Wh, self.tf_Wm, self.tf_Wt, self.tf_Bh,
                            self.tf_Wr, self.tf_bt, self.tf_vt, self.tf_wt]

        self.all_mini_vars = [self.Wh_mini, self.Wm_mini, self.Wt_mini, self.Bh_mini,
                              self.Wr_mini, self.bt_mini, self.vt_mini, self.wt_mini]

        with tf.name_scope('split_grad'):
            # The gradients are added over the batch if made into a single call.
            self.LL_grads = {x: [tf.gradients(y, x)
                                 for y in tf.split(self.LL, self.batch_size)]
                             for x in self.all_tf_vars}
            self.loss_grads = {x: [tf.gradients(y, x)
                                   for y in tf.split(self.loss, self.batch_size)]
                               for x in self.all_tf_vars}

            avg_reward = tf.reduce_mean(self.tf_batch_rewards, axis=0) if with_advantage else 0.0

            # Attempt to calculate the gradient within TensorFlow for the entire
            # batch, without moving to the CPU.
            self.tower_gradients = [
                [(((tf.gather(self.tf_batch_rewards, idx) + tf.gather(self.loss, idx) - avg_reward) * self.LL_grads[x][idx][0] +
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

        with tf.name_scope('stack_grad'):
            self.LL_grad_stacked = {x: tf.gradients(self.LL_stack, x)
                                    for x in self.all_mini_vars}
            self.loss_grad_stacked = {x: tf.gradients(self.loss_stack, x)
                                      for x in self.all_mini_vars}

            self.avg_gradient_stack = []

            # TODO: Can we calculate natural gradients here easily?
            # This is one of the baseline rewards we can calculate.
            avg_reward = tf.reduce_mean(self.tf_batch_rewards, axis=0) if with_advantage else 0.0

            # Removing the average reward converts this coefficient into the advantage function.
            coef = tf.squeeze(self.tf_batch_rewards, axis=-1) + self.loss_stack - avg_reward

            for x, y in zip(self.all_mini_vars, self.all_tf_vars):
                LL_grad = self.LL_grad_stacked[x][0]
                loss_grad = self.loss_grad_stacked[x][0]

                dim = len(LL_grad.get_shape())
                if dim == 1:
                    self.avg_gradient_stack.append(
                        (tf.reduce_mean(LL_grad * coef + loss_grad, axis=0), y)
                    )
                elif dim == 2:
                    self.avg_gradient_stack.append(
                        (
                            tf.reduce_mean(
                                LL_grad * tf.tile(tf.reshape(coef, (-1, 1)),
                                                  [1, tf.shape(LL_grad)[1]]) +
                                loss_grad,
                                axis=0
                            ),
                            y
                        )
                    )
                elif dim == 3:
                    self.avg_gradient_stack.append(
                        (
                            tf.reduce_mean(
                                LL_grad * tf.tile(tf.reshape(coef, (-1, 1, 1)),
                                                  [1, tf.shape(LL_grad)[1], tf.shape(LL_grad)[2]]) +
                                loss_grad,
                                axis=0
                            ),
                            y
                        )
                    )

            self.clipped_avg_gradients_stack, self.grad_norm_stack = \
                tf.clip_by_global_norm([grad for grad, _ in self.avg_gradient_stack],
                                       clip_norm=self.clip_norm)

            self.clipped_avg_gradient_stack = list(zip(
                self.clipped_avg_gradients_stack,
                [var for _, var in self.avg_gradient_stack]
            ))

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
        self.sgd_stacked_op = self.opt.apply_gradients(self.clipped_avg_gradient_stack,
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
            tf.contrib.training.add_gradients_summaries(self.avg_gradient_stack)

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
        return exp_sampler.ExpRecurrentBroadcaster(src_id=self.src_id,
                                                   seed=seed,
                                                   trainer=self)

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
        mgr.run_dynamic(max_events=self.abs_max_events + 1)
        return mgr.get_state().get_dataframe()

    def get_feed_dict(self, batch_df, is_test=False, pre_comp_batch_rewards=None):
        """Produce a feed_dict for the given batch."""

        # assert all(df.sink_id.nunique() == 1 for df in batch_df), "Can only handle one sink at the moment."

        all_followers = self.sim_opts.sink_ids
        all_followers = sorted(all_followers)
        num_followers = len(all_followers)

        # If the batch sizes are not the same, then the learning cannot happen.
        # However, the forward pass works just as well.
        #
        # if not is_test and len(batch_df) != self.batch_size:
        #     raise ValueError("A training batch should consist of {} simulations, not {}."
        #                      .format(self.batch_size, len(batch_df)))

        batch_size = len(batch_df)

        if self.tf_max_events is None:
            max_events = max(df.event_id.nunique() for df in batch_df)
        else:
            max_events = self.abs_max_events

        full_shape = (batch_size, max_events)

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
        batch_ranks = np.zeros(shape=full_shape + (num_followers,), dtype=float)
        batch_init_h = np.zeros(shape=(batch_size, self.num_hidden_states), dtype=float)
        batch_last_interval = np.zeros(shape=batch_size, dtype=float)

        batch_seq_len = np.asarray([np.minimum(x.event_id.nunique(), max_events)
                                    for x in batch_df], dtype=int)[:, np.newaxis]

        for idx, df in enumerate(batch_df):
            # They are sorted by time already.
            batch_len = int(batch_seq_len[idx])

            # If the rank of our broadcaster is 'nan', then make it zero.
            rank_in_tau = RU.rank_of_src_in_df(df=df, src_id=self.src_id).fillna(0.0)

            # If there is a follower who has not seen a single event in the df,
            # Then
            for follower_id in all_followers:
                if follower_id not in rank_in_tau.columns:
                    rank_in_tau[follower_id] = 0

            batch_ranks[idx, 0:batch_len, :] = rank_in_tau[all_followers].values[0:batch_len, :]

            df_unique = df.groupby('event_id').first()
            batch_b_idxes[idx, 0:batch_len] = df_unique.src_id.map(self.src_embed_map).values[0:batch_len]
            batch_t_deltas[idx, 0:batch_len] = df_unique.time_delta.values[0:batch_len]

            if batch_len == df.event_id.nunique():
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
            reward_fn(
                x,
                reward_kind=self.reward_kind,
                reward_opts=make_reward_opts(self),
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
                   clipping=True, stack_grad=True,
                   with_summaries=False, with_MP=False):
        """Run one SGD op given a batch of simulation."""

        seed_start = init_seed + self.sess.run(self.global_step) * self.batch_size

        if with_summaries:
            assert self.summary_dir is not None
            os.makedirs(self.summary_dir, exist_ok=True)
            train_writer = tf.summary.FileWriter(self.summary_dir,
                                                 self.sess.graph)

        if stack_grad:
            assert clipping, "stacked gradients are always clipped."
            train_op = self.sgd_stacked_op
            grad_norm_op = self.grad_norm_stack
            LL_op = self.LL_stack
            loss_op = self.loss_stack
        else:
            train_op = self.sgd_op if not clipping else self.sgd_clipped_op
            grad_norm_op = self.grad_norm
            LL_op = self.LL
            loss_op = self.loss

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
                    self.sess.run([self.tf_batch_rewards, LL_op, loss_op,
                                   grad_norm_op, self.tf_merged_summaries,
                                   self.global_step, self.tf_learning_rate,
                                   train_op],
                                  feed_dict=f_d)
                train_writer.add_summary(summaries, step)
            else:
                reward, LL, loss, grad_norm, step, lr, _ = \
                    self.sess.run([self.tf_batch_rewards, LL_op, loss_op,
                                   grad_norm_op, self.global_step,
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
        # TODO: May not work if abs_max_events is hit.

        feed_dict[self.calc_u_h_states] = h_states
        feed_dict[self.calc_u_batch_size] = [batch_size]

        tf_seq_len = np.squeeze(
            self.sess.run(self.tf_batch_seq_len, feed_dict=feed_dict),
            axis=-1
        ) + 1  # +1 to include the survival term.

        assert self.tf_max_events is None or np.all(tf_seq_len < self.abs_max_events), "Cannot handle events > max_events right now."
        # This will involve changing how the survival term is added, is_own_event is added, etc.

        tf_c_is_arr = self.sess.run(self.calc_u_c_is_rest, feed_dict=feed_dict)
        tf_c_is = (
            [
                self.sess.run(
                    self.calc_u_c_is_init,
                    feed_dict=feed_dict
                )
            ] +
            np.split(tf_c_is_arr, tf_c_is_arr.shape[1], axis=1)
        )
        tf_c_is = list(zip(*tf_c_is))

        tf_t_deltas_arr = self.sess.run(self.tf_batch_t_deltas, feed_dict=feed_dict)
        tf_t_deltas = (
            np.split(tf_t_deltas_arr, tf_t_deltas_arr.shape[1], axis=1) +
            # Cannot add last_interval at the end of the array because
            # the sequence may have ended before that.
            # Instead, we add tf_t_deltas of 0 to make the length of this
            # array the same as of tf_c_is
            [np.asarray([0.0] * batch_size)]
        )
        tf_t_deltas = list(zip(*tf_t_deltas))

        tf_is_own_event_arr = self.sess.run(self.calc_u_is_own_event, feed_dict=feed_dict)
        tf_is_own_event = (
            np.split(tf_is_own_event_arr, tf_is_own_event_arr.shape[1], axis=1) +
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
            # assert tf_is_own_event[idx][tf_seq_len[idx] - 1]
            tf_is_own_event[idx][tf_seq_len[idx] - 1] = False

            assert tf_t_deltas[idx][tf_seq_len[idx] - 1] == 0
            assert tf_t_deltas[idx][tf_seq_len[idx] - 2] > 0
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

            # The seed doesn't make a difference because we will not
            # take samples from this sampler, we will only ask it to
            # calculate the square loss and the LL.
            #
            # TODO: This sampler needs to change from ExpCDFSampler to
            # SigmoidCDFSampler.
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
                # while abs_idx < len(tf_t_deltas[batch_idx]) - 1 and abs_time + tf_t_deltas[batch_idx][abs_idx] < t:
                while abs_idx < tf_seq_len[batch_idx] - 1 and abs_time + tf_t_deltas[batch_idx][abs_idx] < t:
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

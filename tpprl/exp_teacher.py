import numpy as np
import os
import tensorflow as tf
import decorated_options as Deco
import warnings
import multiprocessing as MP
import redqueen.utils as RU
import heapq

SAVE_DIR = 'teacher-log'
MAX_EVENTS = 100000
REWARD_SCALING = 100

# DEBUG ONLY
try:
    from .utils import variable_summaries, _now
    from .cells import TPPRExpMarkedCellStacked
    from .exp_sampler import ExpCDFSampler
except ModuleNotFoundError:
    warnings.warn('Could not import local modules. Assuming they have been loaded using %run -i')


def softmax(x):
    e_x = np.exp(x - x.max())
    return e_x / e_x.sum()


class Student:
    def __init__(self, n_0s, alphas, betas, seed):
        """n_0 is the initial expertise in all items."""
        self.ns = np.asarray([x for x in n_0s])
        self.alphas = alphas
        self.betas = betas
        self.seed = seed
        self.last_review_times = np.zeros_like(n_0s)
        self.RS = np.random.RandomState(seed)
        self.int_m1_sq = 0

    def review(self, item, cur_time):
        recall = self.recall(item, cur_time)
        n_i = self.ns[item]

        # Calculate the (1 - m(t))^2 loss of the past interval.
        t_delta = cur_time - self.last_review_times[item]
        x = t_delta + (1. / n_i) * (
            2.0 * np.exp(-n_i * t_delta) -
            0.5 * np.exp(-2.0 * n_i * t_delta) -
            1.5
        )
        # print('x = ', x, 'n_i =', n_i, 't_delta = ', t_delta, '2.0 * np.exp(-n_i * t_delta) = ', 2.0 * np.exp(-n_i * t_delta), '0.5 * np.exp(-2.0 * n_i * t_delta) = ', 0.5 * np.exp(-2.0 * n_i * t_delta), ' total =', (
        #     2.0 * np.exp(-n_i * t_delta) -
        #     0.5 * np.exp(-2.0 * n_i * t_delta) -
        #     1.5
        # ))
        self.int_m1_sq  += x

        if recall:
            n_new = n_i * (1 - self.alphas[item])
        else:
            n_new = n_i * (1 + self.betas[item])

        # The numbers need to be more carefully tuned?
        self.ns[item] = min(max(1e-6, n_new), 1e2)
        self.last_review_times[item] = cur_time

        return recall

    def get_m1_sq(self, episode_end_time):
        m1 = self.int_m1_sq
        for item in range(len(self.ns)):
            t_delta = episode_end_time - self.last_review_times[item]
            n_i = self.ns[item]
            m1 += t_delta + (1. / n_i) * (
                2.0 * np.exp(-n_i * t_delta) -
                0.5 * np.exp(-2.0 * n_i * t_delta) -
                1.5
            )
        return m1

    def recall(self, item, t):
        m_t = self.prob_recall(item, t)
        return self.RS.rand() < m_t

    def prob_recall(self, item, t):
        return np.exp(-(self.ns[item] * (t - self.last_review_times[item])))


def mk_standard_student(scenario_opts, seed):
    alphas = scenario_opts['alphas']
    betas = scenario_opts['betas']
    n_0s = scenario_opts['n_0s']
    return Student(n_0s, alphas, betas, seed)


class Scenario:
    def __init__(self, scenario_opts, seed,
                 Wm, Wh, Wr, Wt, Bh, Vy,
                 wt, vt, bt, init_h):

        self.num_items = Vy.shape[1]
        self.student = mk_standard_student(scenario_opts, seed=seed * 2)

        self.Wm = Wm
        self.Wh = Wh
        self.Wr = Wr
        self.Wt = Wt
        self.Bh = Bh
        self.Vy = Vy

        self._init = False
        self.cur_h = init_h
        self.T = scenario_opts['T']
        self.default_tau = scenario_opts['tau']
        self.RS = np.random.RandomState(seed)
        self.last_time = -1

        self.c_is = []
        self.hidden_states = []
        self.time_deltas = []
        self.recalls = []
        self._recall_probs = []
        self.items = []
        self.item_probs = []

        self.exp_sampler = ExpCDFSampler(wt=wt, vt=vt, bt=bt, init_h=init_h,
                                         t_min=0,
                                         seed=seed + 1)

    def get_m1_sq(self):
        """Calculates the (1 - m)^2 reward for this scenario."""
        assert self._init
        return self.student.get_m1_sq(self.T)

    def get_all_c_is(self):
        assert self._init
        return np.asarray(self.c_is + [self.exp_sampler.c])

    def get_last_interval(self):
        assert self._init
        return self.T - self.last_time

    def get_all_time_deltas(self):
        assert self._init
        return np.asarray(self.time_deltas +
                          [self.T - self.last_time])

    def get_all_hidden_states(self):
        assert self._init
        return np.asarray(self.hidden_states + [self.cur_h])

    def get_num_events(self):
        assert self._init
        return len(self.c_is)

    def get_item_probs(self):
        assert self._init
        return self.item_probs

    def get_recalls(self):
        assert self._init
        return self.recalls

    def update_hidden_state(self, item, t, time_delta):
        """Returns the hidden state after a post by src_id and time delta."""
        self._recall_probs.append(self.student.prob_recall(item, t))
        recall = float(self.student.review(item, t))
        self.recalls.append(recall)

        return np.tanh(
            self.Wm[item, :][:, np.newaxis] +
            self.Wh.dot(self.cur_h) +
            self.Wr.dot(recall) +
            self.Wt * time_delta +
            self.Bh
        )

    def generate_sample(self, p):
        t_next = self.exp_sampler.generate_sample()
        item_next = self.RS.choice(np.arange(self.num_items), p=p)
        return (t_next, item_next)

    def run(self, max_events=None):
        """Execute a study episode."""
        assert not self._init
        self._init = True

        if max_events is None:
            max_events = float('inf')

        idx = 0
        t = 0

        p = softmax(self.Vy.T.dot(self.cur_h)).squeeze(axis=-1)
        self.item_probs.append(p)
        (t_next, item_next) = self.generate_sample(p)

        while idx < max_events and t_next < self.T:
            idx += 1
            time_delta = t_next - t

            self.items.append(item_next)
            self.c_is.append(self.exp_sampler.c)
            self.hidden_states.append(self.cur_h)
            self.time_deltas.append(time_delta)
            self.last_time = t

            t = t_next

            self.cur_h = self.update_hidden_state(item_next, t, time_delta)
            self.exp_sampler.register_event(t, self.cur_h, own_event=True)

            p = softmax(self.Vy.T.dot(self.cur_h)).squeeze(axis=-1)
            self.item_probs.append(p)
            (t_next, item_next) = self.generate_sample(p)

        return self

    def reward(self, tau=None):
        """Returns the result of a test conducted at T + tau."""
        if tau is None:
            tau = self.default_tau

        return np.mean([self.student.recall(item, self.T + tau)
                       for item in range(self.num_items)])


def mk_def_teacher_opts(hidden_dims, num_items,
                        scenario_opts, seed=42, **kwargs):
    """Make default option set."""
    RS  = np.random.RandomState(seed=seed)

    def_exp_recurrent_teacher_opts = Deco.Options(
        t_min=0,
        scope=None,
        decay_steps=100,
        decay_rate=0.001,
        num_hidden_states=hidden_dims,
        learning_rate=.01,
        learning_bump=1.0,
        clip_norm=1.0,
        tau=15.0,

        Wh=RS.randn(hidden_dims, hidden_dims) * 0.1 + np.diag(np.ones(hidden_dims)),  # Careful initialization
        Wm=RS.randn(num_items, hidden_dims),
        Wr=RS.randn(hidden_dims, 1),
        Wt=RS.randn(hidden_dims, 1),
        # Vy=RS.randn(hidden_dims, num_items),
        Vy=np.ones((hidden_dims, num_items)),  # Careful initialization
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
        T=100.0,

        device_cpu='/cpu:0',
        device_gpu='/gpu:0',
        only_cpu=False,

        save_dir=SAVE_DIR,

        # Expected: './tpprl.summary/train-{}/'.format(run)
        summary_dir=None,

        decay_q_rate=0.0,

        # Whether or not to use the advantage formulation.
        with_baseline=True,

        q=0.0005,
        q_entropy=0.01,

        scenario_opts=scenario_opts,
    )

    return def_exp_recurrent_teacher_opts.set(**kwargs)


class ExpRecurrentTeacher:
    @Deco.optioned()
    def __init__(self, Vy, Wm, Wh, Wt, Wr, Bh, vt, wt, bt, num_hidden_states,
                 sess, scope, batch_size, max_events, q, q_entropy,
                 learning_bump, learning_rate, clip_norm, t_min, T,
                 summary_dir, save_dir, decay_steps, decay_rate, momentum,
                 device_cpu, device_gpu, only_cpu, with_baseline,
                 num_items, decay_q_rate, scenario_opts, tau):
        """Initialize the trainer with the policy parameters."""

        self.decay_q_rate = decay_q_rate
        self.scenario_opts = scenario_opts

        self.t_min = 0
        self.t_max = T
        # self.T = T
        # self.tau = tau

        self.summary_dir = summary_dir
        self.save_dir = save_dir

        # self.src_embed_map = {x.src_id: idx + 1
        #                       for idx, x in enumerate(sim_opts.create_other_sources())}

        # To handle multiple reloads of redqueen related modules.
        self.src_embed_map = np.arange(num_items)

        self.tf_dtype = tf.float32
        self.np_dtype = np.float32

        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.clip_norm = clip_norm

        self.q = q
        self.q_entropy = q_entropy
        self.batch_size = batch_size

        self.tf_batch_size = None
        self.tf_max_events = None
        self.num_items = num_items

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
                self.tf_recall = tf.placeholder(name='recall', shape=(1, 1), dtype=self.tf_dtype)

                self.tf_h_next = tf.nn.tanh(
                    tf.transpose(
                        tf.nn.embedding_lookup(self.tf_Wm, self.tf_b_idx, name='b_embed')
                    ) +
                    tf.matmul(self.tf_Wh, self.tf_h) +
                    tf.matmul(self.tf_Wr, self.tf_recall) +
                    self.tf_Wt * self.tf_t_delta +
                    self.tf_Bh,
                    name='h_next'
                )

            with tf.variable_scope('output'):
                with tf.device(var_device):
                    self.tf_Vy = tf.get_variable(name='Vy', shape=Vy.shape,
                                                 initializer=tf.constant_initializer(Vy))
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
                self.tf_batch_recalls = tf.placeholder(name='recalls',
                                                       shape=(self.tf_batch_size, self.tf_max_events),
                                                       dtype=self.tf_dtype)
                self.tf_batch_seq_len = tf.placeholder(name='seq_len',
                                                       shape=(self.tf_batch_size, 1),
                                                       dtype=tf.int32)
                self.tf_batch_last_interval = tf.placeholder(name='last_interval',
                                                             shape=self.tf_batch_size,
                                                             dtype=self.tf_dtype)

                # Inferred batch size
                inf_batch_size = tf.shape(self.tf_batch_b_idxes)[0]

                self.tf_batch_init_h = tf.zeros(
                    name='init_h',
                    shape=(inf_batch_size, self.num_hidden_states),
                    dtype=self.tf_dtype
                )

                # Stacked version (for performance)

                with tf.name_scope('stacked'):
                    with tf.device(var_device):
                        (self.Wm_mini, self.Wr_mini, self.Wh_mini,
                         self.Wt_mini, self.Bh_mini, self.wt_mini,
                         self.vt_mini, self.bt_mini, self.Vy_mini) = [
                             tf.stack(x, name=name)
                             for x, name in zip(
                                     zip(*[
                                         (tf.identity(self.tf_Wm), tf.identity(self.tf_Wr),
                                          tf.identity(self.tf_Wh), tf.identity(self.tf_Wt),
                                          tf.identity(self.tf_Bh), tf.identity(self.tf_wt),
                                          tf.identity(self.tf_vt), tf.identity(self.tf_bt),
                                          tf.identity(self.tf_Vy))
                                         for _ in range(self.batch_size)
                                     ]),
                                     ['Wm', 'Wr', 'Wh', 'Wt', 'Bh', 'wt', 'vt', 'bt', 'Vy']
                             )
                        ]

                        self.rnn_cell_stack = TPPRExpMarkedCellStacked(
                            hidden_state_size=(None, self.num_hidden_states),
                            output_size=[self.num_hidden_states] + [1] * 4,
                            tf_dtype=self.tf_dtype,
                            Wm=self.Wm_mini, Wr=self.Wr_mini,
                            Wh=self.Wh_mini, Wt=self.Wt_mini,
                            Bh=self.Bh_mini, wt=self.wt_mini,
                            vt=self.vt_mini, bt=self.bt_mini,
                            Vy=self.Vy_mini
                        )

                        ((self.h_states_stack, LL_log_terms_stack,
                          LL_int_terms_stack, loss_terms_stack,
                          entropy_terms_stack),
                         tf_batch_h_t_mini) = tf.nn.dynamic_rnn(
                            self.rnn_cell_stack,
                            inputs=(tf.expand_dims(self.tf_batch_b_idxes, axis=-1),
                                    tf.expand_dims(self.tf_batch_recalls, axis=-1),
                                    tf.expand_dims(self.tf_batch_t_deltas, axis=-1)),
                            sequence_length=tf.squeeze(self.tf_batch_seq_len, axis=-1),
                            dtype=self.tf_dtype,
                            initial_state=self.tf_batch_init_h
                        )

                        self.LL_log_terms_stack = tf.squeeze(LL_log_terms_stack, axis=-1)
                        self.LL_int_terms_stack = tf.squeeze(LL_int_terms_stack, axis=-1)
                        self.loss_terms_stack = tf.squeeze(loss_terms_stack, axis=-1)
                        self.entropy_terms_stack = tf.squeeze(entropy_terms_stack, axis=-1)

                        # LL_last_term_stack = rnn_cell.last_LL(tf_batch_h_t_mini, self.tf_batch_last_interval)
                        # loss_last_term_stack = rnn_cell.last_loss(tf_batch_h_t_mini, self.tf_batch_last_interval)

                        self.LL_last_term_stack = self.rnn_cell_stack.last_LL(tf_batch_h_t_mini, self.tf_batch_last_interval)
                        self.loss_last_term_stack = self.rnn_cell_stack.last_loss(tf_batch_h_t_mini, self.tf_batch_last_interval)

                        self.LL_stack = (tf.reduce_sum(self.LL_log_terms_stack, axis=1) - tf.reduce_sum(self.LL_int_terms_stack, axis=1)) + self.LL_last_term_stack

                        decay_term = tf.pow(tf.cast(self.global_step, self.tf_dtype), self.decay_q_rate)
                        tf_seq_len = tf.squeeze(self.tf_batch_seq_len, axis=-1)
                        self.entropy_stack = tf.where(
                            tf.equal(tf_seq_len, 0),
                            tf.zeros(shape=(inf_batch_size,), dtype=self.tf_dtype),
                            tf.reduce_sum(self.entropy_terms_stack, axis=1) / tf.cast(tf_seq_len, self.tf_dtype),
                            name='entropy_stack'
                        )
                        self.loss_stack = decay_term * (
                            (self.q / 2) * (tf.reduce_sum(self.loss_terms_stack, axis=1) +
                                            self.loss_last_term_stack) -
                            self.q_entropy * self.entropy_stack
                        )

            with tf.name_scope('calc_u'):
                with tf.device(var_device):
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
                            self.tf_Wr, self.tf_bt, self.tf_vt, self.tf_wt, self.tf_Vy]

        self.all_mini_vars = [self.Wh_mini, self.Wm_mini, self.Wt_mini, self.Bh_mini,
                              self.Wr_mini, self.bt_mini, self.vt_mini, self.wt_mini, self.Vy_mini]

        with tf.name_scope('stack_grad'):
            with tf.device(var_device):
                self.LL_grad_stacked = {x: tf.gradients(self.LL_stack, x)
                                        for x in self.all_mini_vars}
                self.loss_grad_stacked = {x: tf.gradients(self.loss_stack, x)
                                          for x in self.all_mini_vars}

                self.avg_gradient_stack = []

                # TODO: Can we calculate natural gradients here easily?
                # This is one of the baseline rewards we can calculate.
                avg_baseline = tf.reduce_mean(self.tf_batch_rewards, axis=0) + tf.reduce_mean(self.loss_stack, axis=0) if with_baseline else 0.0

                # Removing the average reward converts this coefficient into the advantage function.
                coef = tf.squeeze(self.tf_batch_rewards, axis=-1) + self.loss_stack - avg_baseline

                for x, y in zip(self.all_mini_vars, self.all_tf_vars):
                    LL_grad = self.LL_grad_stacked[x][0]

                    # This is needed if the loss does not depend on certain parameters.
                    # if x == self.Vy_mini:
                    #     loss_grad = 0
                    # else:
                    #     loss_grad = self.loss_grad_stacked[x][0]

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
                    tf.clip_by_global_norm(
                        [grad for grad, _ in self.avg_gradient_stack],
                        clip_norm=self.clip_norm
                    )

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

        self.learning_bump = learning_bump
        self.opt = tf.train.AdamOptimizer(
            learning_rate=learning_bump * self.tf_learning_rate,
            beta1=momentum
        )
        self.sgd_stacked_op = self.opt.apply_gradients(
            self.clipped_avg_gradient_stack,
            global_step=self.global_step
        )

        self.sess = sess

        # There are other global variables as well, like the ones which the
        # ADAM optimizer uses.
        self.saver = tf.train.Saver(
            tf.global_variables(),
            keep_checkpoint_every_n_hours=0.25,
            max_to_keep=1000
        )

        with tf.device(device_cpu):
            tf.contrib.training.add_gradients_summaries(self.avg_gradient_stack)

            for v in self.all_tf_vars:
                variable_summaries(v)

            variable_summaries(self.tf_learning_rate, name='learning_rate')
            variable_summaries(self.loss_stack, name='loss_stack')
            variable_summaries(self.LL_stack, name='LL_stack')
            variable_summaries(self.loss_last_term_stack, name='loss_last_term_stack')
            variable_summaries(self.LL_last_term_stack, name='LL_last_term_stack')
            variable_summaries(self.h_states_stack, name='hidden_states_stack')
            variable_summaries(self.LL_log_terms_stack, name='LL_log_terms_stack')
            variable_summaries(self.LL_int_terms_stack, name='LL_int_terms_stack')
            variable_summaries(self.loss_terms_stack, name='loss_terms_stack')
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

    def train_many(self, num_iters, init_seed=42, with_summaries=False,
                   with_MP=False, save_every=25, with_recall_probs=False,
                   with_memorize_loss=False):
        """Run one SGD op given a batch of simulation."""

        seed_start = init_seed + self.sess.run(self.global_step) * self.batch_size

        if with_summaries:
            assert self.summary_dir is not None
            os.makedirs(self.summary_dir, exist_ok=True)
            train_writer = tf.summary.FileWriter(self.summary_dir,
                                                 self.sess.graph)
        train_op = self.sgd_stacked_op
        grad_norm_op = self.grad_norm_stack
        LL_op = self.LL_stack
        loss_op = self.loss_stack
        entropy_op = self.entropy_stack

        chkpt_file = os.path.join(self.save_dir, 'tpprl.ckpt')

        pool = None
        try:
            if with_MP:
                pool = MP.Pool()

            for iter_idx in range(num_iters):
                seed_end = seed_start + self.batch_size

                seeds = range(seed_start, seed_end)

                if with_MP:
                    raw_scenarios = [mk_scenario_from_teacher(self, seed)
                                     for seed in seeds]
                    scenarios = pool.map(_scenario_worker, raw_scenarios)
                else:
                    scenarios = [run_scenario(self, seed) for seed in seeds]

                num_events = [s.get_num_events() for s in scenarios]
                num_items_practised = [len(set(s.items)) for s in scenarios]

                num_correct = [np.sum(s.get_recalls()) for s in scenarios]

                f_d = get_feed_dict(self, scenarios,
                                    with_recall_probs=with_recall_probs,
                                    with_memorize_loss=with_memorize_loss)

                if with_summaries:
                    reward, LL, loss, entropy, grad_norm, summaries, step, lr, _ = \
                        self.sess.run([self.tf_batch_rewards, LL_op, loss_op,
                                       entropy_op, grad_norm_op,
                                       self.tf_merged_summaries,
                                       self.global_step, self.tf_learning_rate,
                                       train_op],
                                      feed_dict=f_d)
                    train_writer.add_summary(summaries, step)
                else:
                    reward, LL, loss, entropy, grad_norm, step, lr, _ = \
                        self.sess.run([self.tf_batch_rewards, LL_op, loss_op,
                                       entropy_op, grad_norm_op,
                                       self.global_step, self.tf_learning_rate,
                                       train_op],
                                      feed_dict=f_d)

                mean_LL = np.mean(LL)
                std_LL = np.std(LL)

                mean_loss = np.mean(loss)
                std_loss = np.std(loss)

                mean_reward = np.mean(reward)
                std_reward = np.std(reward)

                mean_CTG = np.mean(loss + reward)
                std_CTG = np.std(loss + reward)

                mean_entropy = np.mean(entropy)
                std_entropy = np.std(entropy)

                mean_events = np.mean(num_events)
                std_events = np.std(num_events)

                mean_correct = np.mean(num_correct)
                std_correct = np.std(num_correct)

                print('{} Run {}, LL {:.2f}±{:.2f}, entropy {:.2f}±{:.2f}, loss {:.2f}±{:.2f}, Rwd {:.3f}±{:.3f}'
                      ', CTG {:.3f}±{:.3f}, seeds {}--{}, grad_norm {:.2f}, step = {}'
                      ', lr = {:.5f}, events = {:.2f}±{:.2f}/{:.2f}±{:.2f}, items = {:.2f}/{}, wt={:.5f}, bt={:.5f}'
                      .format(_now(), iter_idx,
                              mean_LL, std_LL,
                              mean_entropy, std_entropy,
                              mean_loss, std_loss,
                              mean_reward, std_reward,
                              mean_CTG, std_CTG,
                              seed_start, seed_end - 1,
                              grad_norm, step, lr,
                              mean_correct, std_correct,
                              mean_events, std_events,
                              np.mean(num_items_practised), self.num_items,
                              self.sess.run(self.tf_wt)[0], self.sess.run(self.tf_bt)[0]))

                # Ready for the next iter_idx.
                seed_start = seed_end

                if iter_idx % save_every == 0:
                    print('Saving model!')
                    self.saver.save(self.sess,
                                    chkpt_file,
                                    global_step=self.global_step,)

        finally:
            if pool is not None:
                pool.close()

            if with_summaries:
                train_writer.flush()

            print('Saving model!')
            self.saver.save(self.sess, chkpt_file, global_step=self.global_step,)

    def restore(self, restore_dir=None, epoch_to_recover=None):
        """Restores the model from a saved checkpoint."""

        if restore_dir is None:
            restore_dir = self.save_dir

        chkpt = tf.train.get_checkpoint_state(restore_dir)

        if epoch_to_recover is not None:
            suffix = '-{}.meta'.format(epoch_to_recover)
            file = [x for x in chkpt.all_model_checkpoint_paths
                    if x.endswith(suffix)]
            if len(file) < 1:
                raise FileNotFoundError('Epoch {} not found.'
                                        .format(epoch_to_recover))
            self.saver.restore(self.sess, file[0])
        else:
            self.saver.restore(self.sess, chkpt.model_checkpoint_path)

    def calc_u(self, h_states, feed_dict, batch_size, times, batch_time_start=None):
        """Calculate u(t) at the times provided."""
        # TODO: May not work if abs_max_events is hit.

        if batch_time_start is None:
            batch_time_start = np.zeros(batch_size)

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

            # This quantity may be zero for real-data.
            # assert tf_t_deltas[idx][tf_seq_len[idx] - 2] > 0

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
            sampler = ExpCDFSampler(vt=vt, wt=wt, bt=bt,
                                    init_h=init_h,
                                    t_min=batch_time_start[idx],
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
            abs_time = batch_time_start[idx]
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


def get_feed_dict(teacher, scenarios, with_recall_probs=False, with_memorize_loss=False):
    """Produce a feed_dict for the given list of scenarios."""

    # assert all(df.sink_id.nunique() == 1 for df in batch_df), "Can only handle one sink at the moment."

    batch_size = len(scenarios)
    max_events = max(s.get_num_events() for s in scenarios)
    num_items = scenarios[0].num_items

    full_shape = (batch_size, max_events)

    if with_memorize_loss:
        batch_rewards = np.asarray([
            s.get_m1_sq() for s in scenarios
        ])[:, np.newaxis]
    else:
        if not with_recall_probs:
            batch_rewards = np.asarray([
                -REWARD_SCALING * s.reward() for s in scenarios
            ])[:, np.newaxis]
        else:
            # We are exploiting the fact that we are training against a simulator.
            #
            # Will use this to investigate appropriate parameter space before
            # switching to stochastic rewards.
            batch_rewards = np.asarray([
                -REWARD_SCALING * np.mean([s.student.prob_recall(item, s.T + s.default_tau)
                                           for item in range(num_items)])
                for s in scenarios
            ])[:, np.newaxis]

    batch_last_interval = np.asarray([
        s.get_last_interval() for s in scenarios
    ], dtype=float)

    batch_seq_len = np.asarray([
        s.get_num_events() for s in scenarios
    ], dtype=float)[:, np.newaxis]

    batch_t_deltas = np.zeros(shape=full_shape, dtype=float)
    batch_b_idxes = np.zeros(shape=full_shape, dtype=int)
    batch_recalls = np.zeros(shape=full_shape, dtype=float)
    batch_init_h = np.zeros(shape=(batch_size, teacher.num_hidden_states), dtype=float)

    for idx, scen in enumerate(scenarios):
        # They are sorted by time already.
        batch_len = int(batch_seq_len[idx])

        batch_recalls[idx, 0:batch_len] = scen.recalls
        batch_b_idxes[idx, 0:batch_len] = scen.items
        batch_t_deltas[idx, 0:batch_len] = scen.time_deltas

    return {
        teacher.tf_batch_b_idxes: batch_b_idxes,
        teacher.tf_batch_rewards: batch_rewards,
        teacher.tf_batch_seq_len: batch_seq_len,
        teacher.tf_batch_t_deltas: batch_t_deltas,
        teacher.tf_batch_recalls: batch_recalls,
        teacher.tf_batch_init_h: batch_init_h,
        teacher.tf_batch_last_interval: batch_last_interval,
    }


def mk_scenario_from_opts(teacher_opts, seed):
    return Scenario(scenario_opts=teacher_opts.scenario_opts,
                    seed=seed,
                    Wh=teacher_opts.Wh,
                    Wm=teacher_opts.Wm,
                    Wr=teacher_opts.Wr,
                    Wt=teacher_opts.Wt,
                    Vy=teacher_opts.Vy,
                    Bh=teacher_opts.Bh,

                    vt=teacher_opts.vt,
                    wt=teacher_opts.wt,
                    bt=teacher_opts.bt,

                    init_h=np.zeros((teacher_opts.num_hidden_states, 1)))


def mk_scenario_from_teacher(teacher, seed):
    return Scenario(scenario_opts=teacher.scenario_opts,
                    seed=seed,
                    Wh=teacher.sess.run(teacher.tf_Wh),
                    Wm=teacher.sess.run(teacher.tf_Wm),
                    Wr=teacher.sess.run(teacher.tf_Wr),
                    Wt=teacher.sess.run(teacher.tf_Wt),
                    Vy=teacher.sess.run(teacher.tf_Vy),
                    Bh=teacher.sess.run(teacher.tf_Bh),

                    vt=teacher.sess.run(teacher.tf_vt),
                    wt=teacher.sess.run(teacher.tf_wt),
                    bt=teacher.sess.run(teacher.tf_bt),

                    init_h=np.zeros((teacher.num_hidden_states, 1)))


def get_test_feed_dicts(teacher, seeds, **kwargs):
    seeds = list(seeds)
    scenarios = [mk_scenario_from_teacher(teacher, seed).run(max_events=MAX_EVENTS)
                 for seed in seeds]
    return (get_feed_dict(teacher, scenarios, **kwargs), scenarios)


def run_scenario(teacher, seed):
    return mk_scenario_from_teacher(teacher, seed).run(max_events=MAX_EVENTS)


def _scenario_worker(scenario):
    scenario.run()
    return scenario


# Baselines

def uniform_baseline(scenario_opts, target_reviews, seed, verbose=True):
    """Distribute target_reviews uniformly over the study period."""
    student = mk_standard_student(scenario_opts, seed=seed * 2)
    T = scenario_opts['T']
    tau = scenario_opts['tau']

    num_items = scenario_opts['alphas'].shape[0]
    per_item_reviews = target_reviews / num_items
    interval = T / per_item_reviews

    reviews = 0
    review_timings = []
    for t in np.arange(0, T, interval):
        for item in np.arange(num_items):
            student.review(item, t)
            review_timings.append((item, t))
            reviews += 1

    if verbose:
        print('Total reviews = {}/{}'.format(reviews, target_reviews))

    return {
        'reward': -REWARD_SCALING * np.mean([student.recall(item, T + tau)
                                             for item in range(num_items)]),
        'student': student,
        'num_reviews': reviews,
        'review_timings': review_timings,
    }


def uniform_random_baseline(
        scenario_opts, target_reviews,
        seed, verbose=True
):
    """Distribute target_reviews uniformly over the study period."""
    student = mk_standard_student(scenario_opts, seed=seed * 2)
    num_items = len(scenario_opts['n_0s'])
    T, tau = scenario_opts['T'], scenario_opts['tau']

    RS = np.random.RandomState(seed)
    num_random_reviews = RS.poisson(target_reviews)

    items = RS.choice(np.arange(num_items),
                      size=num_random_reviews,
                      replace=True)
    reviews_times = RS.uniform(size=num_random_reviews) * T

    review_timings = []
    for (item, t) in zip(items, reviews_times):
        student.review(item, t)
        review_timings.append((item, t))

    if verbose:
        print('Total reviews = {}/{}'
              .format(num_random_reviews, target_reviews))

    return {
        'reward': -REWARD_SCALING * np.mean([student.recall(item, T + tau)
                                             for item in range(num_items)]),
        'student': student,
        'num_reviews': num_random_reviews,
        'review_timings': review_timings,
    }


# Memorize implementation

def sample_memorize(q_max, forgetting_rate, RS):
    dt = 0
    while True:
        dt += RS.exponential(scale=1.0 / q_max)
        if RS.uniform() < 1 - np.exp(-forgetting_rate * dt):
            return dt


def memorize_baseline(scenario_opts, q_max, seed, verbose=True):
    student = mk_standard_student(scenario_opts, seed=seed * 2)
    num_items = scenario_opts['alphas'].shape[0]
    T, tau = scenario_opts['T'], scenario_opts['tau']

    reviews = []
    RS = np.random.RandomState(seed=seed * 7)
    for item in range(num_items):
        next_t_delta = sample_memorize(
            q_max, student.ns[item], RS
        )
        heapq.heappush(reviews, (next_t_delta, item))

    num_reviews = 0
    review_timings = []
    while True:
        (next_t, item) = heapq.heappop(reviews)
        if next_t > T:
            break

        num_reviews += 1
        student.review(item, next_t)
        review_timings.append((item, next_t))
        next_t_delta = sample_memorize(q_max, student.ns[item], RS)
        heapq.heappush(reviews, (next_t + next_t_delta, item))

    return {
        'reward': - REWARD_SCALING * np.mean([
            student.recall(item, T + tau) for item in range(num_items)
        ]),
        'num_reviews': num_reviews,
        'student': student,
        'review_timings': review_timings,
        'm_2_reward': student.get_m1_sq(episode_end_time=T),
    }


# Sweeping q

def calc_q_capacity_iter_memorize(
        scenario_opts, q_suggested, verbose=False,
        seeds=None, parallel=True, max_events=None
):
    if seeds is None:
        seeds = range(250, 270)

    num_reviews = [
        memorize_baseline(scenario_opts, q_max=q_suggested,
                          seed=x, verbose=verbose)['num_reviews']
        for x in seeds
    ]

    return np.asarray(num_reviews)
    # capacities = np.zeros(len(seeds), dtype=float)
    # if not parallel:
    #     for idx, seed in enumerate(seeds):
    #         m = sim_opts.create_manager_with_opt(seed)
    #         if dynamic:
    #             m.run_dynamic(max_events=max_events)
    #         else:
    #             m.run()
    #         capacities[idx] = u_int_opt(m.state.get_dataframe(),
    #                                     sim_opts=sim_opts)
    # else:
    #     num_workers = min(len(seeds), mp.cpu_count())
    #     with mp.Pool(num_workers) as pool:
    #         for (idx, capacity) in \
    #             enumerate(pool.imap(q_int_worker, [(sim_opts, x, dynamic, max_events)
    #                                                for x in seeds])):
    #             capacities[idx] = capacity

    # return capacities


# There are so many ways this can go south. Particularly, if the user capacity
# is much higher than the average of the wall of other followees.
def sweep_memorize_q(scenario_opts, capacity_cap, q_init, tol=1e-2,
                     verbose=False, parallel=True, max_events=None,
                     max_iters=float('inf')):

    # We know that on average, the ∫u(t)dt or number of events increases with
    # increasing 'q'

    def terminate_cond(new_capacity):
        return abs(new_capacity - capacity_cap) / capacity_cap < tol or \
            np.ceil(capacity_cap - 1) <= new_capacity <= np.ceil(capacity_cap + 1)

    # Step 1: Find the upper/lower bound by exponential increase/decrease
    init_cap = calc_q_capacity_iter_memorize(
        scenario_opts,
        q_init,
        parallel=parallel,
        max_events=max_events
    ).mean()

    if terminate_cond(init_cap):
        return q_init

    if verbose:
        RU.logTime('Initial capacity = {}, target capacity = {}, q_init = {}'
                   .format(init_cap, capacity_cap, q_init))

    q = q_init
    if init_cap < capacity_cap:
        iters = 0
        while True:
            iters += 1
            q_hi = q
            q *= 2.0
            q_lo = q
            capacity = calc_q_capacity_iter_memorize(
                scenario_opts, q,
                parallel=parallel,
                max_events=max_events
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
            q /= 2.0
            q_hi = q
            capacity = calc_q_capacity_iter_memorize(
                scenario_opts, q,
                parallel=parallel,
                max_events=max_events
            ).mean()
            if verbose:
                RU.logTime('q = {}, capacity = {}'.format(q, capacity))
            # Will break if capacity_cap is too low ~ 1 event,
            # unless only_tol is True.
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
        new_capacity = calc_q_capacity_iter_memorize(
            scenario_opts, q,
            parallel=parallel,
            max_events=max_events
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

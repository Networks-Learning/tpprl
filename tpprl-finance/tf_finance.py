import tensorflow as tf
import pandas as pd

# def mk_def_trader_opts():

class ExpRecurrentTrader:
    def __init__(self, wt, W_t, Wb_alpha, Ws_alpha, Wn_b, Wn_s,
                 W_h, W_1, W_2, W_3, b_t, b_alpha, bn_b, bn_s, b_h,
                 Vt_h, Vt_v, b_lambda, Vh_alpha, Vv_alpha, Va_b, Va_s,
                 num_hidden_states,sess, scope, batch_size, max_events,
                 q, learning_rate, clip_norm, summary_dir, save_dir, decay_steps, decay_rate, momentum,
                device_cpu, device_gpu, only_cpu):
        print()
        self.summary_dir = summary_dir
        self.save_dir = save_dir
        self.tf_dtype = tf.float32
        self.np_dtype = np.float32

        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.clip_norm = clip_norm

        self.q = q
        self.batch_size = batch_size

        self.tf_batch_size = None

        self.tf_max_events = None
        self.num_hidden_states = num_hidden_states

        self.scope = scope or type(self).__name__
        var_device = device_cpu if only_cpu else device_gpu
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

                with tf.variable_scope('output'):
                    with tf.device(var_device):
                        self.tf_Vy = tf.get_variable(name='Vy', shape=Vy.shape,
                                                     initializer=tf.constant_initializer(Vy))
                        self.tf_bt = tf.get_variable(name='bt', shape=bt.shape,
                                                     initializer=tf.constant_initializer(bt))
                        self.tf_vt = tf.get_variable(name='vt', shape=vt.shape,
                                                     initializer=tf.constant_initializer(vt))

                        wt_init = 0.0 if set_wt_zero else wt
                        self.tf_wt = tf.get_variable(name='wt', shape=wt.shape,
                                                     initializer=tf.constant_initializer(wt_init))
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
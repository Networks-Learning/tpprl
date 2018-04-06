import tensorflow as tf
import numpy as np


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

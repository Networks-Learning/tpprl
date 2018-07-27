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

        c = tf.matmul(h_prev, self.tf_vt) + self.tf_bt

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

    def last_LL(self, last_h, last_interval):
        """Calculate the likelihood of the survival term."""
        raise NotImplementedError()

    def last_loss(self, last_h, last_interval):
        """Calculate the squared loss of the survival term."""
        raise NotImplementedError()

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

    def last_LL(self, last_h, last_interval):
        """Calculate the likelihood of the survival term."""
        inf_batch_size = tf.shape(last_interval)[0]
        t_0 = tf.zeros(name='zero_time_last', shape=(inf_batch_size, 1), dtype=self.tf_dtype)
        u_theta = self.u_theta(last_h, tf.reshape(last_interval, (-1, 1)), name='u_theta_LL_last')
        u_theta_0 = self.u_theta(last_h, t_0, name='u_theta_LL_last_0')
        return tf.squeeze(-(1 / self.tf_wt) * (u_theta - u_theta_0), axis=-1)

    def last_loss(self, last_h, last_interval):
        """Calculate the squared loss of the survival term."""
        inf_batch_size = tf.shape(last_interval)[0]
        t_0 = tf.zeros(name='zero_time_last', shape=(inf_batch_size, 1), dtype=self.tf_dtype)
        u_theta = self.u_theta(last_h, tf.reshape(last_interval, (-1, 1)), name='u_theta_loss_last')
        u_theta_0 = self.u_theta(last_h, t_0, name='u_theta_loss_last_0')
        return tf.squeeze(
            (1 / (2 * self.tf_wt)) * (
                tf.square(u_theta) - tf.square(u_theta_0)
            ),
            axis=-1
        )

    @property
    def output_size(self):
        return self._output_size

    @property
    def state_size(self):
        return self._hidden_state_size


class TPPRExpCellStacked(tf.contrib.rnn.RNNCell):
    """u(t) = exp(vt * ht + wt * dt + bt).
    Stacked version.
    """

    def __init__(self, hidden_state_size, output_size, src_id, tf_dtype,
                 Wm, Wr, Wh, Wt, Bh,
                 wt, vt, bt, assume_wt_zero=False):
        self._output_size = output_size
        self._hidden_state_size = hidden_state_size
        self.src_id = src_id
        self.tf_dtype = tf_dtype
        self.assume_wt_zero = assume_wt_zero

        # The embedding matrix is reshaped because we will need to lookup into
        # it.
        batch_size, num_cats, embed_size = Wm.get_shape()
        self.tf_Wm = tf.reshape(Wm, (batch_size * num_cats, embed_size))
        self.tf_Wr = Wr
        self.tf_Wh = Wh
        self.tf_Wt = Wt
        self.tf_Bh = Bh

        self.tf_wt = wt
        self.tf_vt = vt
        self.tf_bt = bt

        self.num_cats = tf.shape(Wm)[1]

    def u_theta(self, h, t_delta, name):
        return tf.exp(
            tf.einsum('aij,ai->aj', self.tf_vt, h) +
            tf.einsum('ai,ai->ai', self.tf_wt, t_delta) +
            self.tf_bt,
            name=name
        )

    def __call__(self, inp, h_prev):
        raw_broadcaster_idx, rank, t_delta = inp
        inf_batch_size = tf.shape(raw_broadcaster_idx)[0]

        broadcaster_idx = tf.squeeze(raw_broadcaster_idx, axis=-1)
        lookup_offset = self.num_cats * tf.range(inf_batch_size)

        h_next = tf.nn.tanh(
            tf.nn.embedding_lookup(self.tf_Wm, broadcaster_idx + lookup_offset) +
            tf.einsum('aij,aj->ai', self.tf_Wh, h_prev) +
            tf.einsum('aij,aj->ai', self.tf_Wr, rank) +
            tf.einsum('aij,aj->ai', self.tf_Wt, t_delta) +
            tf.squeeze(self.tf_Bh, axis=-1),
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
        if self.assume_wt_zero:
            LL_int = u_theta * t_delta
            loss = tf.square(u_theta) * t_delta
        else:
            LL_int = (u_theta - u_theta_0) / self.tf_wt
            # print('LL_int = ', LL_int)
            loss = (tf.square(u_theta) - tf.square(u_theta_0)) / (2 * self.tf_wt)
            # print('loss = ', loss)

        return ((h_next,
                 tf.expand_dims(LL_log, axis=-1, name='LL_log'),
                 LL_int,
                 loss),
                h_next)

    def last_LL(self, last_h, last_interval):
        """Calculate the likelihood of the survival term."""
        inf_batch_size = tf.shape(last_interval)[0]
        t_0 = tf.zeros(name='zero_time_last', shape=(inf_batch_size, 1), dtype=self.tf_dtype)
        u_theta_0 = self.u_theta(last_h, t_0, name='u_theta_LL_last_0')
        u_theta = self.u_theta(last_h, tf.reshape(last_interval, (-1, 1)), name='u_theta_LL_last')
        if self.assume_wt_zero:
            return tf.squeeze(- u_theta * tf.reshape(last_interval, (-1, 1)))
        else:
            return tf.squeeze(-(1 / self.tf_wt) * (u_theta - u_theta_0), axis=-1)

    def last_loss(self, last_h, last_interval):
        """Calculate the squared loss of the survival term."""
        inf_batch_size = tf.shape(last_interval)[0]
        t_0 = tf.zeros(name='zero_time_last', shape=(inf_batch_size, 1), dtype=self.tf_dtype)
        u_theta_0 = self.u_theta(last_h, t_0, name='u_theta_loss_last_0')
        u_theta = self.u_theta(last_h, tf.reshape(last_interval, (-1, 1)), name='u_theta_loss_last')
        if self.assume_wt_zero:
            return tf.squeeze(tf.square(u_theta) * tf.reshape(last_interval, (-1, 1)))
        else:
            return tf.squeeze(
                (1 / (2 * self.tf_wt)) * (
                    tf.square(u_theta) - tf.square(u_theta_0)
                ),
                axis=-1
            )

    @property
    def output_size(self):
        return self._output_size

    @property
    def state_size(self):
        return self._hidden_state_size


class TPPRExpMarkedCellStacked(tf.contrib.rnn.RNNCell):
    """u(t) = exp(vt * ht + wt * dt + bt).
    v(t) = softmax(Vy * ht)

    Stacked version.
    """

    def __init__(self, hidden_state_size, output_size, tf_dtype,
                 Wm, Wr, Wh, Wt, Bh,
                 wt, vt, bt, Vy, assume_wt_zero=False):
        self._output_size = output_size
        self._hidden_state_size = hidden_state_size
        self.tf_dtype = tf_dtype
        self.assume_wt_zero = assume_wt_zero

        # The embedding matrix is reshaped because we will need to lookup into
        # it.
        batch_size, num_cats, embed_size = Wm.get_shape()
        self.tf_Wm = tf.reshape(Wm, (batch_size * num_cats, embed_size))
        self.tf_Wr = Wr
        self.tf_Wh = Wh
        self.tf_Wt = Wt
        self.tf_Bh = Bh

        self.tf_wt = wt
        self.tf_vt = vt
        self.tf_bt = bt
        self.tf_Vy = Vy

        self.num_cats = tf.shape(Wm)[1]

    def u_theta(self, h, t_delta, name):
        return tf.exp(
            tf.einsum('aij,ai->aj', self.tf_vt, h) +
            tf.einsum('ai,ai->ai', self.tf_wt, t_delta) +
            self.tf_bt,
            name=name
        )

    def __call__(self, inp, h_prev):
        raw_b_idx, recall, t_delta = inp
        inf_batch_size = tf.shape(raw_b_idx)[0]

        b_idx = tf.squeeze(raw_b_idx, axis=-1)
        lookup_offset = self.num_cats * tf.range(inf_batch_size)

        h_next = tf.nn.tanh(
            tf.nn.embedding_lookup(self.tf_Wm, b_idx + lookup_offset) +
            tf.einsum('aij,aj->ai', self.tf_Wh, h_prev) +
            tf.einsum('aij,aj->ai', self.tf_Wr, recall) +
            tf.einsum('aij,aj->ai', self.tf_Wt, t_delta) +
            tf.squeeze(self.tf_Bh, axis=-1),
            name='h_next'
        )

        u_theta = self.u_theta(h_prev, t_delta, name='u_theta')
        # print('u_theta = ', u_theta)

        t_0 = tf.zeros(name='zero_time', shape=(inf_batch_size, 1), dtype=self.tf_dtype)
        u_theta_0 = self.u_theta(h_prev, t_0, name='u_theta_0')

        # Calculating entropy of the categorical distribution.
        v_logits = tf.einsum('aij,ai->aj', self.tf_Vy, h_prev, name='v_logits')
        v_logexpsum = tf.reduce_logsumexp(v_logits, axis=1, keepdims=True, name='v_logexpsum')
        v_probs = tf.nn.softmax(v_logits, axis=1, name='v_probs')

        v_entropy = tf.reduce_sum(
            tf.multiply(-v_probs, v_logits - v_logexpsum),
            axis=1,
            keepdims=True,
            name='v_entropy'
        )

        v_unrolled = tf.reshape(
            v_probs,
            shape=[-1],
            name='v_unrolled'
        )
        # print('v_logits', v_logits)
        # print('v_logexpsum', v_logexpsum)
        # print('v_probs', v_probs)
        # print('v_entropy', v_entropy)

        # LL calculation
        LL_log = (
            tf.squeeze(tf.log(u_theta), axis=-1) +
            tf.log(tf.gather(v_unrolled, b_idx + lookup_offset))
        )

        if self.assume_wt_zero:
            LL_int = u_theta * t_delta
            loss = tf.square(u_theta) * t_delta
        else:
            LL_int = (u_theta - u_theta_0) / self.tf_wt
            loss = (tf.square(u_theta) - tf.square(u_theta_0)) / (2 * self.tf_wt)

        return ((h_next,
                 tf.expand_dims(LL_log, axis=-1, name='LL_log'),
                 LL_int,
                 loss,
                 v_entropy),
                h_next)

    def last_LL(self, last_h, last_interval):
        """Calculate the likelihood of the survival term."""
        inf_batch_size = tf.shape(last_interval)[0]
        t_0 = tf.zeros(name='zero_time_last', shape=(inf_batch_size, 1), dtype=self.tf_dtype)
        u_theta_0 = self.u_theta(last_h, t_0, name='u_theta_LL_last_0')
        u_theta = self.u_theta(last_h, tf.reshape(last_interval, (-1, 1)), name='u_theta_LL_last')
        if self.assume_wt_zero:
            return tf.squeeze(- u_theta * tf.reshape(last_interval, (-1, 1)))
        else:
            return tf.squeeze(-(1 / self.tf_wt) * (u_theta - u_theta_0), axis=-1)

    def last_loss(self, last_h, last_interval):
        """Calculate the squared loss of the survival term."""
        inf_batch_size = tf.shape(last_interval)[0]
        t_0 = tf.zeros(name='zero_time_last', shape=(inf_batch_size, 1), dtype=self.tf_dtype)
        u_theta_0 = self.u_theta(last_h, t_0, name='u_theta_loss_last_0')
        u_theta = self.u_theta(last_h, tf.reshape(last_interval, (-1, 1)), name='u_theta_loss_last')
        if self.assume_wt_zero:
            return tf.squeeze(tf.square(u_theta) * tf.reshape(last_interval, (-1, 1)))
        else:
            return tf.squeeze(
                (1 / (2 * self.tf_wt)) * (
                    tf.square(u_theta) - tf.square(u_theta_0)
                ),
                axis=-1
            )

    @property
    def output_size(self):
        return self._output_size

    @property
    def state_size(self):
        return self._hidden_state_size

import tensorflow as tf
import numpy as np


class TPPRExpMarkedCellStacked_finance(tf.contrib.rnn.RNNCell):
    """u(t) = exp(vt * ht + wt * dt + bt).
    v(t) = softmax(Vy * ht)

    Stacked version.
    """

    def __init__(self, hidden_state_size, output_size, tf_dtype,
                 W_t,Wb_alpha,Ws_alpha,Wn_b,Wn_s,W_h,W_1,
                 W_2,W_3,b_t,b_alpha,bn_b,bn_s,b_h,wt,
                 Vt_h,Vt_v,b_lambda,Vh_alpha,Vv_alpha,Va_b,Va_s, assume_wt_zero=False):
        self._output_size = output_size
        self._hidden_state_size = hidden_state_size
        self.tf_dtype = tf_dtype
        self.assume_wt_zero = assume_wt_zero

        # The embedding matrix is reshaped because we will need to lookup into
        # it.
        # batch_size, num_cats, embed_size = Wm.get_shape()
        # self.tf_Wm = tf.reshape(Wm, (batch_size * num_cats, embed_size))
        self.wt = wt
        self.W_t = W_t
        self.Wb_alpha = Wb_alpha
        self.Ws_alpha = Ws_alpha
        self.Wn_b = Wn_b
        self.Wn_s = Wn_s
        self.W_h = W_h
        self.W_1 = W_1
        self.W_2 = W_2
        self.W_3 = W_3
        self.b_t = b_t
        self.b_alpha = b_alpha
        self.bn_b = bn_b
        self.bn_s = bn_s
        self.b_h = b_h
        self.Vt_h = Vt_h
        self.Vt_v = Vt_v
        self.b_lambda = b_lambda
        self.Vh_alpha = Vh_alpha
        self.Vv_alpha = Vv_alpha
        self.Va_b = Va_b
        self.Va_s = Va_s

        # self.num_cats = tf.shape(Wm)[1]

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

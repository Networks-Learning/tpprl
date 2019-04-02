import tensorflow as tf
# import numpy as np

MAX_AMT = 1000.0
MAX_SHARE = 100
BASE_CHARGES = 1.0
PERCENTAGE_CHARGES = 0.001


class TPPRExpMarkedCellStacked_finance(tf.contrib.rnn.RNNCell):
    """u(t) = exp(vt * ht + wt * dt + bt).
    v(t) = softmax(Vy * ht)

    Stacked version.
    """

    def __init__(self, hidden_state_size, output_size, tf_dtype,
                 W_t, Wb_alpha, Ws_alpha, Wn_b, Wn_s, W_h, W_1,
                 W_2, W_3, b_t, b_alpha, bn_b, bn_s, b_h, wt,
                 Vt_h, Vt_v, b_lambda, Vh_alpha, Vv_alpha, Va_b, Va_s):
        self._output_size = output_size
        self._hidden_state_size = hidden_state_size
        self.tf_dtype = tf_dtype

        # The embedding matrix is reshaped because we will need to lookup into it.
        # batch_size, num_cats, embed_size = Wm.get_shape()
        # self.tf_Wm = tf.reshape(Wm, (batch_size * num_cats, embed_size))
        self.tf_wt = wt
        self.tf_W_t = W_t
        self.tf_Wb_alpha = Wb_alpha
        self.tf_Ws_alpha = Ws_alpha
        self.tf_Wn_b = Wn_b
        self.tf_Wn_s = Wn_s
        self.tf_W_h = W_h
        self.tf_W_1 = W_1
        self.tf_W_2 = W_2
        self.tf_W_3 = W_3
        self.tf_b_t = b_t
        self.tf_b_alpha = b_alpha
        self.tf_bn_b = bn_b
        self.tf_bn_s = bn_s
        self.tf_b_h = b_h
        self.tf_Vt_h = Vt_h
        self.tf_Vt_v = Vt_v
        self.tf_b_lambda = b_lambda
        self.tf_Vh_alpha = Vh_alpha
        self.tf_Vv_alpha = Vv_alpha
        self.tf_Va_b = Va_b
        self.tf_Va_s = Va_s

    def u_theta(self, h, t_delta, name):
        return tf.exp(
            tf.einsum('aij,ai->aj', self.tf_Vt_h, h) +
            tf.einsum('ai,ai->ai', self.tf_wt, t_delta) +
            self.tf_b_lambda,
            name=name
        )

    def __call__(self, inp, h_prev):
        # TODO: Event type \in {TradeFB, ReadFB}
        t_delta, alpha_i, n_i, v_curr, is_trade_feedback, current_amt, portfolio = inp
        inf_batch_size = tf.shape(t_delta)[0]

        if is_trade_feedback:
            tau_i = tf.math.add(x=tf.einsum('aij,aj->ai', self.tf_W_t, t_delta, name='einsum_tau_i'), y=self.tf_b_t, name="tau_i")
            b_i = tf.math.add(tf.math.add(tf.einsum('aij,aj->ai', self.tf_Wb_alpha, (1-alpha_i)),
                                          tf.einsum('sij,aj->ai', self.tf_Ws_alpha, alpha_i), name="einsum_b_i"),
                              self.tf_b_alpha, name="b_i")
            if alpha_i == 0:
                eta_i = tf.math.add(tf.einsum('aij,aj->ai', self.tf_Wn_b, n_i, name="einsum_n_i"), self.tf_bn_b, name='n_i_buy')
            else:
                eta_i = tf.math.add(tf.einsum('aij,aj->ai', self.tf_Wn_s, n_i, name="einsum_n_i"), self.tf_bn_s, name='n_i_sell')

            h_next = tf.nn.tanh(
                tf.einsum('aij,aj->ai', self.tf_Wh, h_prev) +
                tf.einsum('aij,aj->ai', self.tf_W_1, tau_i) +
                tf.einsum('aij,aj->ai', self.tf_W_2, b_i) +
                tf.einsum('aij,aj->ai',self.tf_W_3, eta_i) +
                tf.squeeze(self.tf_b_h, axis=-1),
                name='h_next'
            )
        else:
            h_next = h_prev
        # TODO: LL calculation for alpha_i and n_i

        u_theta = self.u_theta(h_prev, t_delta, name='u_theta')
        t_0 = tf.zeros(name='zero_time', shape=(inf_batch_size, 1), dtype=self.tf_dtype)
        u_theta_0 = self.u_theta(h_prev, t_0, name='u_theta_0')

        # calculte LL for alpha with sigmoid
        prob_alpha_i = tf.nn.sigmoid(tf.math.add(tf.einsum('aij,ai->aj', self.tf_Vh_alpha, h_prev, name="einsum_alphai_hi"),
                                                 tf.einsum('aij,ai->aj', self.tf_Vv_alpha, t_delta, name="einsum_alphai_tdelta"),
                                                 name="add_prob_alphai"),
                                     name="prob_alpha_i")
        # LL of alpha_i
        LL_alpha_i = tf.squeeze(tf.log(prob_alpha_i[alpha_i]), axis=-1)

        # calculate LL for n_i
        # TODO: apply mask??
        # calculate mask
        if alpha_i == 0:
            A = tf.einsum('aij,ai->aj', self.tf_Va_b, h_prev, name='A_buy')
            max_share_buy = min(MAX_SHARE, int(tf.math.floor(
                self.current_amt / (v_curr + (v_curr * PERCENTAGE_CHARGES))))) + 1  # to allow buying zero shares
            mask = tf.expand_dims(tf.concat(tf.ones(max_share_buy),
                                            tf.zeros(MAX_SHARE + 1 - max_share_buy)), axis=-1)  # total size is 101
            masked_A =tf.multiply(mask, A)
            masked_A[:max_share_buy] = tf.exp(masked_A[:max_share_buy])
            prob_n = masked_A / tf.reduce_sum(masked_A[:max_share_buy], axis=-1)
            prob_n = tf.squeeze(prob_n)
        else:
            A = tf.einsum('aij,ai->aj', self.tf_Va_s, h_prev, name='A_sell')
            num_share_sell = int(
                (self.owned_shares * v_curr) / (v_curr + (v_curr * PERCENTAGE_CHARGES)))
            max_share_sell = min(MAX_SHARE, num_share_sell) + 1  # to allow buying zero shares
            mask = tf.expand_dims(tf.concat(tf.ones(max_share_sell), tf.zeros(MAX_SHARE + 1 - max_share_sell)), axis=1)
            masked_A = tf.multiply(mask, A)
            masked_A[:max_share_sell] = tf.exp(masked_A[:max_share_sell])
            prob_n = masked_A / tf.reduce_sum(masked_A[:max_share_sell], axis=-1)
            prob_n = tf.squeeze(prob_n)
        # LL of n_i
        LL_n_i = tf.squeeze(tf.log(prob_n[n_i]), axis=-1)

        # LL of t_i and delta calculation
        LL_log = tf.squeeze(tf.log(u_theta), axis=-1)
        LL_int = (u_theta - u_theta_0) / self.tf_wt
        loss = (tf.square(u_theta) - tf.square(u_theta_0)) / (2 * self.tf_wt)

        return ((h_next,
                 tf.expand_dims(LL_log, axis=-1, name='LL_log'),
                 tf.expand_dims(LL_alpha_i, axis=-1, name='LL_alpha_i'),
                 tf.expand_dims(LL_n_i, axis=-1, name='LL_n_i'),
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

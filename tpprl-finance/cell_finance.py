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
            tf.einsum('aji,ai->aj', self.tf_Vt_h, h) +
            tf.einsum('aij,ai->aj', self.tf_wt, t_delta) +  # TODO: v_curr-v_past i.e. v_delta
            self.tf_b_lambda,
            name=name
        )

    def __call__(self, inp, h_prev):
        # TODO: Event type \in {TradeFB, ReadFB}
        t_delta, alpha_i, n_i, v_curr, is_trade_feedback, current_amt, portfolio = inp
        inf_batch_size = tf.shape(t_delta)[0]

        def calculate_h_next():
            tau_i = tf.math.add(x=tf.einsum('aij,aj->ai', self.tf_W_t, t_delta, name='einsum_tau_i'),
                                y=tf.squeeze(self.tf_b_t, axis=-1),
                                name="tau_i")
            b_i = tf.math.add(
                tf.math.add(tf.einsum('aij,aj->ai', self.tf_Wb_alpha, tf.cast((1 - alpha_i), dtype=tf.float32)),
                            tf.einsum('aij,aj->ai', self.tf_Ws_alpha, tf.cast(alpha_i, dtype=tf.float32)),
                            name="einsum_b_i"),
                tf.squeeze(self.tf_b_alpha, axis=-1), name="b_i")

            def encode_buy_n_i():
                buy_n_i = tf.math.add(
                    tf.einsum('aij,aj->ai', self.tf_Wn_b, tf.cast(n_i, dtype=tf.float32), name="einsum_n_i_buy"),
                    tf.squeeze(self.tf_bn_b, axis=-1),
                    name='n_i_buy')
                return buy_n_i

            def encode_sell_n_i():
                sell_n_i = tf.math.add(
                    tf.einsum('aij,aj->ai', self.tf_Wn_s, tf.cast(n_i, dtype=tf.float32), name="einsum_n_i_sell"),
                    tf.squeeze(self.tf_bn_s, axis=-1),
                    name='n_i_sell')
                return sell_n_i

            val_alpha_i = tf.squeeze(tf.gather(alpha_i, 0, axis=-1), axis=-1)
            eta_i = tf.cond(
                pred=tf.equal(val_alpha_i, 0),
                true_fn=encode_buy_n_i,
                false_fn=encode_sell_n_i,
                name="encode_n_i_eta_i"
            )
            hnext = tf.nn.tanh(
                tf.einsum('aij,ai->aj', self.tf_W_h, h_prev) +
                tf.einsum('aij,ai->aj', self.tf_W_1, tau_i) +
                tf.einsum('aij,ai->aj', self.tf_W_2, b_i) +
                tf.einsum('aij,ai->aj', self.tf_W_3, eta_i) +
                tf.squeeze(self.tf_b_h, axis=-1),
                name='h_next'
            )
            return hnext

        # TODO: LL calculation for alpha_i and n_i
        val_is_trade_feedback = tf.squeeze(tf.gather(is_trade_feedback, 0, axis=-1), axis=-1)
        h_next = tf.cond(
            tf.equal(val_is_trade_feedback, 1),
            true_fn=calculate_h_next,
            false_fn=lambda: h_prev,
            name="is_h_next_updated"
        )
        u_theta = self.u_theta(h_prev, t_delta, name='u_theta')
        t_0 = tf.zeros(name='zero_time', shape=(inf_batch_size, 1), dtype=self.tf_dtype)
        u_theta_0 = self.u_theta(h_prev, t_0, name='u_theta_0')

        # calculate LL for alpha with sigmoid
        prob_alpha_i = tf.nn.sigmoid(
            tf.math.add(tf.einsum('aij,ai->aj', self.tf_Vh_alpha, h_prev, name="einsum_alphai_hi"),
                        tf.einsum('aij,ai->aj', self.tf_Vv_alpha, t_delta, name="einsum_alphai_tdelta"),
                        name="add_prob_alphai"),
            name="prob_alpha_i")
        # LL of alpha_i
        LL_alpha_i = tf.squeeze(tf.log(tf.gather(prob_alpha_i, alpha_i, axis=-1)), axis=-1)

        # calculate LL for n_i
        # TODO: apply mask
        def prob_n_buy():
            A = tf.einsum('aij,ai->aj', self.tf_Va_b, h_prev, name='A_buy')
            # TODO: v_curr=(?, portfolio,1)
            max_share_buy = tf.math.minimum(MAX_SHARE, tf.cast(
                tf.math.floor(
                    tf.squeeze(current_amt) / (v_curr + (tf.scalar_mul(scalar=PERCENTAGE_CHARGES, x=v_curr)))),
                dtype=tf.int32)) + 1  # to allow buying zero shares
            mask = tf.cast(tf.expand_dims(
                tf.concat(tf.ones(max_share_buy, dtype=tf.int32),
                          tf.zeros(MAX_SHARE + 1 - max_share_buy, dtype=tf.int32)), axis=-1), dtype=tf.float32)
            exp_A = tf.exp(A)
            masked_A = tf.multiply(mask, exp_A)
            prob_n = masked_A / tf.reduce_sum(masked_A, axis=-1)
            prob_n = tf.squeeze(prob_n)
            return prob_n

        def prob_n_sell():
            A = tf.einsum('aij,ai->aj', self.tf_Va_s, h_prev, name='A_sell')
            num_share_sell = tf.cast(
                tf.multiply(portfolio, v_curr) / (v_curr + tf.scalar_mul(scalar=PERCENTAGE_CHARGES, x=v_curr)),
                dtype=tf.int32)
            max_share_sell = tf.squeeze(tf.math.minimum(MAX_SHARE, num_share_sell)) + 1  # to allow buying zero shares
            mask = tf.cast(tf.expand_dims(
                tf.concat([tf.ones(max_share_sell, dtype=tf.int32),
                           tf.zeros(MAX_SHARE + 1 - max_share_sell, dtype=tf.int32)], axis=-1), axis=1),
                dtype=tf.float32)
            exp_A = tf.exp(A)
            masked_A = tf.multiply(mask, exp_A)
            prob_n = masked_A / tf.reduce_sum(masked_A, axis=-1)
            prob_n = tf.squeeze(prob_n)
            return prob_n

        # calculate mask as per the value of alpha_i=0=buy and 1=sell
        val_alpha_i = tf.squeeze(tf.gather(alpha_i, 0, axis=-1), axis=-1)
        prob_n = tf.cond(
            pred=tf.equal(val_alpha_i, 0),
            true_fn=prob_n_buy,
            false_fn=prob_n_sell,
            name="encode_n_i_eta_i"
        )

        # LL of n_i
        LL_n_i = tf.squeeze(tf.log(tf.gather(prob_n, n_i, axis=-1)), axis=-1)

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

import tensorflow as tf

# import numpy as np

MAX_AMT = 1000.0
MAX_SHARE = 100
BASE_CHARGES = 1.0
PERCENTAGE_CHARGES = 0.001
EPSILON = 1e-6

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

    def u_theta(self, h, t_delta, v_delta, name):
        return tf.exp(
            tf.einsum('aij,aj->ai', self.tf_Vt_h, h, name='Vt_h__h') +
            tf.einsum('aij,aj->ai', self.tf_Vt_v, v_delta, name="Vt_v__v_delta") +
            tf.einsum('aij,ai->aj', self.tf_wt, t_delta, name="wt__t_delta") +  # TODO
            self.tf_b_lambda,
            name=name
        )

    def __call__(self, inp, h_prev):
        # TODO: Event type \in {TradeFB, ReadFB}
        t_delta, alpha_i, n_i, v_curr, is_trade_feedback, current_amt, portfolio, v_delta = inp
        inf_batch_size = tf.shape(t_delta)[0]

        def calculate_h_next():
            print_ops = []
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
                true_fn=lambda: encode_buy_n_i(),
                false_fn=lambda: encode_sell_n_i(),
                name="encode_n_i_eta_i"
            )
            h_val = tf.einsum('aij,aj->ai', self.tf_W_h, h_prev)
            t_val = tf.einsum('aij,aj->ai', self.tf_W_1, tau_i)
            b_val = tf.einsum('aij,aj->ai', self.tf_W_2, b_i)
            n_val = tf.einsum('aij,aj->ai', self.tf_W_3, eta_i)
            bias = tf.squeeze(self.tf_b_h, axis=-1)
            # print_ops.append(tf.print('eta_i:', eta_i, '\n',
            #                           'tau_i:', tau_i,  '\n',
            #                           'val_alpha_i:', val_alpha_i,  '\n',
            #                           'b_i:', b_i,  '\n',
            #                           't_delta:', t_delta, '\n',
            #                           'v_delta:', v_delta, '\n',
            #                           'h_val:', h_val, '\n',
            #                           't_val:', t_val, '\n',
            #                           'b_val:', b_val, '\n',
            #                           'n_val:', n_val, '\n',
            #                           'bias:', bias, '\n'
            #                           '================================='
            #                           ))

            with tf.control_dependencies(print_ops):
                hnext = tf.nn.tanh(
                    h_val +
                    t_val +
                    b_val +
                    n_val +
                    bias,
                    name='h_next'
                )

            return hnext

        val_is_trade_feedback = tf.squeeze(tf.gather(is_trade_feedback, 0, axis=-1), axis=-1)
        cond1 = tf.equal(val_is_trade_feedback, 1)

        h_next = tf.cond(
            cond1,
            true_fn=lambda: calculate_h_next(),
            false_fn=lambda: h_prev,
            name="is_h_next_updated"
        )

        u_theta = tf.squeeze(self.u_theta(h=h_prev, t_delta=t_delta, v_delta=v_delta, name='u_theta'))
        t_0 = tf.zeros(name='zero_time', shape=(inf_batch_size, 1), dtype=self.tf_dtype)
        u_theta_0 = tf.squeeze(self.u_theta(h=h_prev, t_delta=t_0, v_delta=v_delta, name='u_theta_0'))

        # LL of t_i and delta calculation
        LL_log = tf.cond(pred=tf.equal(val_is_trade_feedback, 1),
                         true_fn=lambda: tf.reshape(tf.log(u_theta), shape=[1], name="LL_log_reshape"),
                         false_fn=lambda: tf.zeros(shape=[1]))
        # LL_log = tf.reshape(tf.log(u_theta), shape=[1], name="LL_log_reshape")
        LL_int = tf.reshape((u_theta - u_theta_0) / tf.squeeze(self.tf_wt), shape=[1], name="LL_int_reshape")
        loss = tf.reshape((tf.square(u_theta) - tf.square(u_theta_0)) / (2 * tf.squeeze(self.tf_wt)), shape=[1, 1],
                          name="loss_reshape")

        # calculate LL for alpha with sigmoid
        prob_0 = tf.nn.sigmoid(
            tf.math.add(tf.einsum('aij,aj->ai', self.tf_Vh_alpha, h_prev, name="einsum_alphai_hi"),
                        tf.einsum('aij,aj->ai', self.tf_Vv_alpha, v_delta, name="einsum_alphai_vdelta"),
                        name="add_prob_alphai"),
            name="prob_alpha_i")
        # prob_0 = tf.squeeze(prob_0)
        prob_1 = 1.0-prob_0
        prob_alpha_i = tf.concat([prob_0, prob_1], axis=-1)

        # LL of alpha_i
        LL_alpha_i = tf.squeeze(tf.log(tf.gather(prob_alpha_i, alpha_i, axis=-1)), axis=[-1,-2])

        # calculate LL for n_i
        # TODO: apply mask
        def prob_n_buy():
            A = tf.einsum('aij,aj->ai', self.tf_Va_b, h_prev, name='A_buy')
            # if all values are zero, assign 1.0 to prob of buying zero shares
            # is_all_zero = tf.reduce_sum(A)
            # A = tf.cond(pred=tf.equal(is_all_zero, 0.0),
            #             true_fn=lambda: tf.scatter_update(ref = A, indices=[0], updates=[1.0]),
            #             false_fn=lambda: A)

            max_share_buy = tf.math.maximum(1, tf.math.minimum(MAX_SHARE, tf.cast(
                tf.math.floor(
                    tf.squeeze(current_amt) / (v_curr + (tf.scalar_mul(scalar=PERCENTAGE_CHARGES, x=v_curr)))),
                dtype=tf.int32)))
            ones1 = tf.ones(shape=[max_share_buy])
            zeros1 = tf.zeros(shape=[MAX_SHARE - max_share_buy])
            append1 = tf.concat([ones1, zeros1], axis=-1)
            mask = tf.cast(append1, dtype=tf.float32)
            exp_A = tf.exp(A)
            masked_A = tf.multiply(mask, exp_A)
            # prob_n = masked_A / tf.math.maximum(tf.reduce_sum(masked_A, axis=-1), EPSILON)

            reduce_sum = tf.reduce_sum(masked_A, axis=-1)
            reduce_sum = tf.cond(pred=tf.less_equal(tf.squeeze(tf.abs(reduce_sum)), EPSILON, name="avoid_zero_division_buy"),
                                 true_fn=lambda: EPSILON, false_fn=lambda: reduce_sum)

            prob_n = masked_A / reduce_sum
            prob_n = tf.squeeze(prob_n)
            return prob_n

        def prob_n_sell():
            A = tf.einsum('aij,aj->ai', self.tf_Va_s, h_prev, name='A_sell')
            num_share_sell = tf.cast(
                tf.multiply(portfolio, v_curr) / (v_curr + tf.scalar_mul(scalar=PERCENTAGE_CHARGES, x=v_curr)),
                dtype=tf.int32)
            max_share_sell = tf.math.maximum(1, tf.squeeze(tf.math.minimum(MAX_SHARE, num_share_sell)))
            ones1 = tf.ones([max_share_sell], dtype=tf.int32)
            zeros1 = tf.zeros([MAX_SHARE - max_share_sell], dtype=tf.int32)
            append1 = tf.concat([ones1, zeros1], axis=-1)
            mask = tf.cast(append1, dtype=tf.float32)
            exp_A = tf.exp(A)
            masked_A = tf.multiply(mask, exp_A)
            reduce_sum = tf.reduce_sum(masked_A, axis=-1)
            reduce_sum = tf.cond(pred=tf.less_equal(tf.squeeze(tf.abs(reduce_sum)), EPSILON, name="avoid_zero_division_sell"),
                                 true_fn=lambda: EPSILON, false_fn=lambda: reduce_sum)
            prob_n = masked_A / reduce_sum
            prob_n = tf.squeeze(prob_n)
            return prob_n

        # calculate mask as per the value of alpha_i=0=buy and 1=sell
        val_alpha_i = tf.squeeze(tf.gather(alpha_i, 0, axis=-1), axis=-1)
        prob_n = tf.cond(
            pred=tf.equal(val_alpha_i, 0),
            true_fn=lambda: prob_n_buy(),
            false_fn=lambda: prob_n_sell(),
            name="encode_n_i_eta_i"
        )

        # LL of n_i
        val = tf.squeeze(tf.gather(params=tf.squeeze(prob_n), indices=tf.squeeze(n_i), axis=-1))
        LL_n_i = tf.reshape(tf.log(val), shape=[1], name="LL_n_i_reshape")

        a = tf.expand_dims(LL_log, axis=-1, name='LL_log')
        b = tf.expand_dims(LL_int, axis=-1, name='LL_int')
        c = tf.expand_dims(LL_alpha_i, axis=-1, name='LL_alpha_i')
        d = tf.expand_dims(LL_n_i, axis=-1, name='LL_n_i')
        # return ((h_next,
        #          tf.expand_dims(LL_log, axis=-1, name='LL_log'),
        #          tf.expand_dims(LL_int, axis=-1, name='LL_int'),
        #          tf.expand_dims(LL_alpha_i, axis=-1, name='LL_alpha_i'),
        #          tf.expand_dims(LL_n_i, axis=-1, name='LL_n_i'),
        #          loss),
        #         h_next)
        return ((h_next,
                 a, b, c, d,
                 loss),
                h_next)

    def last_LL(self, last_h, v_delta, last_interval):
        """Calculate the likelihood of the survival term."""
        inf_batch_size = tf.shape(last_interval)[0]
        t_0 = tf.zeros(name='zero_time_last', shape=(inf_batch_size, 1), dtype=self.tf_dtype)
        u_theta_0 = self.u_theta(h=last_h, t_delta=t_0, v_delta=v_delta, name='u_theta_LL_last_0')
        u_theta = self.u_theta(h=last_h, t_delta=tf.reshape(last_interval, (-1, 1)), v_delta=v_delta, name='u_theta_LL_last')
        return tf.squeeze(-(1 / self.tf_wt) * (u_theta - u_theta_0), axis=-1)

    def last_loss(self, last_h, v_delta, last_interval):
        """Calculate the squared loss of the survival term."""
        inf_batch_size = tf.shape(last_interval)[0]
        t_0 = tf.zeros(name='zero_time_last', shape=(inf_batch_size, 1), dtype=self.tf_dtype)
        u_theta_0 = self.u_theta(h=last_h, t_delta=t_0, v_delta=v_delta, name='u_theta_loss_last_0')
        u_theta = self.u_theta(h=last_h, t_delta=tf.reshape(last_interval, (-1, 1)), v_delta=v_delta, name='u_theta_loss_last')
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

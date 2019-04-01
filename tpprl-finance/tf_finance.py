import sys
import tensorflow as tf
import pandas as pd
import numpy as np
import os

from .util_finance import _now, variable_summaries
from .cell_finance import TPPRExpMarkedCellStacked_finance

SAVE_DIR = "/NL/tpprl-result/work/rl-finance/"
HIDDEN_LAYER_DIM = 8
MAX_AMT = 1000.0
MAX_SHARE = 100
BASE_CHARGES = 1.0
PERCENTAGE_CHARGES = 0.001


def reward_fn(events, v_last):
    reward = MAX_AMT
    owned_shares = 0
    print("calculating reward...")
    for (_idx, event) in events.iterrows():
        if event.alpha_i == 0:
            reward -= event.n_i * event.v_curr
            owned_shares += event.n_i
        elif event.alpha_i == 1:
            reward += event.n_i * event.v_curr
            owned_shares -= event.n_i
        reward -= BASE_CHARGES
        reward -= (event.n_i * event.v_curr * PERCENTAGE_CHARGES)
    reward += owned_shares * v_last
    return reward


class Action:
    def __init__(self, alpha, n):
        self.alpha = alpha
        self.n = n

    def __str__(self):
        return "<{}: {}>".format("Sell" if self.alpha > 0 else "Buy", self.n)


class Feedback:
    def __init__(self, t_i, v_curr, is_trade_feedback, event_curr_amt):
        self.t_i = t_i
        self.v_curr = v_curr
        self.is_trade_feedback = is_trade_feedback
        self.event_curr_amt = event_curr_amt

    def is_trade_event(self):
        return self.is_trade_feedback

    def is_tick_event(self):
        return not self.is_trade_feedback


class TradeFeedback(Feedback):
    def __init__(self, t_i, v_curr, alpha_i, n_i, event_curr_amt):
        super(TradeFeedback, self).__init__(t_i, v_curr, is_trade_feedback=True, event_curr_amt=event_curr_amt)
        self.alpha_i = alpha_i
        self.n_i = n_i


class TickFeedback(Feedback):
    def __init__(self, t_i, v_curr, event_curr_amt):
        super(TickFeedback, self).__init__(t_i, v_curr, is_trade_feedback=False, event_curr_amt=event_curr_amt)


class State:
    def __init__(self, curr_time):
        self.time = curr_time
        self.events = []

    def apply_event(self, event):
        self.events.append(event)
        self.time = event.t_i
        # if event.alpha_i == 0:
        #     print("* BUY {} shares at price of {} at time {}".format(event.n_i, event.v_curr, event.t_i))
        # else:
        #     print("* SELL {} shares at price of {} at time {}".format(event.n_i, event.v_curr, event.t_i))

    def get_dataframe(self, output_file):
        df = pd.DataFrame.from_records(
            [{"t_i": event.t_i,
              "alpha_i": event.alpha_i,
              "n_i": event.n_i,
              "v_curr": event.v_curr,
              "is_trade_feedback": event.is_trade_feedback,
              "event_curr_amt": event.event_curr_amt} for event in self.events])
        print("\n saving events:")
        print(df[:2].values)
        df.to_csv(SAVE_DIR + output_file, index=False)
        return df


class Strategy:
    def __init__(self):
        self.current_amt = MAX_AMT
        self.owned_shares = 0

    def get_next_action_time(self, event):
        return NotImplemented

    def get_next_action_item(self, event):
        return NotImplemented

    def update_owned_shares(self, event):
        prev_amt = self.current_amt
        if event.alpha_i == 0:
            self.owned_shares += event.n_i
            # self.current_amt -= event.n_i * event.v_curr
        elif event.alpha_i == 1:
            self.owned_shares -= event.n_i
            # self.current_amt += event.n_i * event.v_curr

        assert self.current_amt>0


class RLStrategy(Strategy):
    def __init__(self, wt, W_t, Wb_alpha, Ws_alpha, Wn_b, Wn_s,
                 W_h, W_1, W_2, W_3, b_t, b_alpha, bn_b, bn_s, b_h,
                 Vt_h, Vt_v, b_lambda, Vh_alpha, Vv_alpha, Va_b, Va_s, seed):
        super(RLStrategy, self).__init__()
        self.RS = np.random.RandomState(seed)
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

        self.tau_i = np.zeros((HIDDEN_LAYER_DIM,1))
        self.b_i = np.zeros((HIDDEN_LAYER_DIM, 1))
        self.eta_i = np.zeros((HIDDEN_LAYER_DIM, 1))
        self.h_i = np.zeros((HIDDEN_LAYER_DIM,1))
        self.u_theta_t = 0
        self.last_time = 0.0
        self.curr_time = None
        self.Q = 1.0
        self.c1 = 1.0
        self.u = self.RS.uniform()
        self.last_price = None
        self.curr_price = None
        self.loglikelihood = 0
        print("using RL strategy")

    def get_next_action_time(self, event):
        # if this method is called after buying/selling action, then sample new u
        if self.curr_price is None:
            # This is the first event
            self.curr_price = event.v_curr
            self.last_price = 0

        # prev_q = self.Q
        if event.is_trade_feedback or self.curr_time is None:
            self.u = self.RS.uniform()
            self.Q = 1
        else:
            self.Q *= (1 - self.cdf(event.t_i))

        # sample t_i
        self.c1 = np.exp(np.array(self.Vt_h).dot(self.h_i) + (self.Vt_v * (self.curr_price - self.last_price)) + self.b_lambda)
        D = 1 - (self.wt / np.exp(self.c1)) * np.log((1 - self.u) / self.Q)
        a = np.squeeze((1-self.u)/self.Q)
        assert a<1
        assert np.log(D) > 0

        self.last_time = event.t_i
        new_t_i = self.last_time + (1 / self.wt) * np.log(D)
        new_t_i = np.asarray(new_t_i).squeeze()
        self.curr_time = new_t_i

        assert self.curr_time >= self.last_time

        # update the log likelihood
        u_theta_0 = np.squeeze(np.exp((self.Vt_h.dot(self.h_i)) + self.b_lambda))
        self.u_theta_t = np.squeeze(np.exp((self.Vt_h.dot(self.h_i)) + (self.wt * (self.curr_time-self.last_time)) + self.b_lambda))

        # calculate log likelihood
        # TODO: save current amount, portfolio: num of share in possession, v_curr
        self.loglikelihood += np.squeeze((self.u_theta_t - u_theta_0) / self.wt)  # prob of no event happening
        return self.curr_time

    def get_next_action_item(self, event):
        self.last_price = self.curr_price
        self.curr_price = event.v_curr
        # update h_i
        self.h_i = np.tanh(np.array(self.W_h).dot(self.h_i) + np.array(self.W_1).dot(self.tau_i)
                           + np.array(self.W_2).dot(self.b_i) + np.array(self.W_3).dot(self.eta_i) + self.b_h)
        # sample alpha_i
        prob_alpha = 1 / (1 + np.exp(
            -np.array(self.Vh_alpha).dot(self.h_i) - np.array(self.Vv_alpha).dot((self.curr_price - self.last_price))))
        alpha_i = self.RS.choice(np.array([0, 1]), p=np.squeeze(prob_alpha))

        # return empty trade details when the balance is insufficient to make a trade
        if self.current_amt <= (BASE_CHARGES + event.v_curr * PERCENTAGE_CHARGES):
            n_i = 0
            return alpha_i, n_i

        # subtract the fixed transaction charges
        self.current_amt -= BASE_CHARGES
        assert self.current_amt > 0
        if alpha_i == 0:
            A = np.array(self.Va_b).dot(self.h_i)
            A = np.append(np.array([[1]]), A, axis=0)

            # calculate mask
            max_share_buy = min(MAX_SHARE, int(np.floor(self.current_amt / (event.v_curr+(event.v_curr*PERCENTAGE_CHARGES)))))+1  # to allow buying zero shares
            mask = np.expand_dims(np.append(np.ones(max_share_buy), np.zeros(MAX_SHARE+1 - max_share_buy)), axis=1)  # total size is 101

            # apply mask
            masked_A = np.multiply(mask, A)
            masked_A[:max_share_buy] = np.exp(masked_A[:max_share_buy])
            prob_n = masked_A / np.sum(masked_A[:max_share_buy])
            prob_n = np.squeeze(prob_n)

            # sample
            n_i = self.RS.choice(np.arange(MAX_SHARE+1), p=np.squeeze(prob_n))
            # self.owned_shares += n_i
            # a = event.v_curr * n_i
            # self.current_amt -= a
            # assert self.current_amt > 0
        else:
            A = np.array(self.Va_b).dot(self.h_i)
            A = np.append(np.array([[1]]), A, axis=0)
            num_share_sell = int((self.owned_shares*event.v_curr)/(event.v_curr+(event.v_curr*PERCENTAGE_CHARGES)))
            max_share_sell = min(MAX_SHARE, num_share_sell)+1  # to allow buying zero shares
            mask = np.expand_dims(np.append(np.ones(max_share_sell), np.zeros(MAX_SHARE+1-max_share_sell)), axis=1)  # total size is 101

            # apply mask
            masked_A = np.multiply(mask, A)
            masked_A[:max_share_sell] = np.exp(masked_A[:max_share_sell])
            prob_n = masked_A / np.sum(masked_A[:max_share_sell])
            prob_n = np.squeeze(prob_n)

            # sample
            n_i = self.RS.choice(np.arange(MAX_SHARE+1), p=np.squeeze(prob_n))
            # self.owned_shares -= n_i
            # self.current_amt += event.v_curr * n_i
            # assert self.current_amt > 0

        # encode event details
        self.tau_i = np.array(self.W_t).dot((self.curr_time - self.last_time)) + self.b_t
        self.b_i = np.array(self.Wb_alpha).dot(1 - alpha_i) + np.array(self.Ws_alpha).dot(alpha_i) + self.b_alpha
        if alpha_i == 0:
            self.eta_i = np.array(self.Wn_b).dot(n_i) + self.bn_b
        else:
            self.eta_i = np.array(self.Wn_s).dot(n_i) + self.bn_s

        # update current amt
        if alpha_i == 0:
            self.current_amt -= n_i * event.v_curr
        elif alpha_i == 1:
            self.current_amt += n_i * event.v_curr

        # if n_i=0 i.e. there was no trade, add the base charges, which was previously deducted
        if n_i == 0:
            self.current_amt += BASE_CHARGES
        # subtract the percentage transaction charges
        a = event.v_curr * n_i * PERCENTAGE_CHARGES
        assert self.current_amt>a
        self.current_amt -= a
        assert self.current_amt > 0
        # update log likelihood
        self.loglikelihood += np.squeeze(np.log(self.u_theta_t) + prob_alpha[alpha_i] + prob_n[n_i])

        return alpha_i, n_i

    def cdf(self, t):
        """Calculates the CDF assuming that the last event was at self.t_last"""
        if self.wt == 0:
            return 1 - np.exp(- np.exp(self.c1) * (t - self.last_time))
        else:
            return 1 - np.exp((np.exp(self.c1) / self.wt) * (1 - np.exp(self.wt * (t - self.last_time))))

    def get_LL(self):
        # TODO: simulation end time
        return self.loglikelihood


class Environment:
    def __init__(self, T, time_gap, raw_data, agent, start_time, seed):
        self.T = T
        self.state = State(curr_time=start_time)
        self.time_gap = time_gap
        self.raw_data = raw_data
        self.agent = agent
        self.RS = np.random.RandomState(seed)
        # for reading market value per minute
        if self.time_gap == "minute":
            # TODO need to find a way to group by minute using unix timestamp
            self.tick_data = self.raw_data.groupby(self.raw_data["datetime"], as_index=False).last()
        elif self.time_gap == "second":
            self.tick_data = self.raw_data.groupby(self.raw_data["datetime"], as_index=False).last()

            # print(self.tick_data.head())
        else:
            raise ValueError("Time gap value '{}' not understood.".format(self.time_gap))

    def get_state(self):
        return self.state

    def simulator(self):
        row_iterator = self.tick_data.iterrows()
        first_tick = next(row_iterator)[1]
        current_event = TickFeedback(t_i=first_tick.datetime, v_curr=first_tick.price, event_curr_amt = self.agent.current_amt)
        v_last = current_event.v_curr
        print("trading..")

        for (_idx, next_tick) in row_iterator:
            while self.state.time <= self.T:
                next_agent_action_time = self.agent.get_next_action_time(current_event)
                # check if there is enough amount to buy at least one share at current price
                if next_agent_action_time > next_tick.datetime:
                    current_event = TickFeedback(t_i=next_tick.datetime, v_curr=next_tick.price, event_curr_amt = self.agent.current_amt)
                    # print("reading market value at time {}".format(current_event.t_i))
                    break
                else:
                    # TODO update price: interpolate
                    trade_price = current_event.v_curr
                    alpha_i, n_i = self.agent.get_next_action_item(current_event)
                    current_event = TradeFeedback(t_i=next_agent_action_time, v_curr=trade_price,
                                                  alpha_i=alpha_i, n_i=n_i, event_curr_amt=self.agent.current_amt)
                    self.agent.update_owned_shares(current_event)

                self.state.apply_event(current_event)
                v_last = current_event.v_curr
        print("LL:",self.agent.get_LL())
        return v_last


def mk_def_trader_opts(seed):
    """Make default option set."""
    t_min = 1254130200
    scope = None
    decay_steps = 100
    decay_rate = 0.001
    num_hidden_states = HIDDEN_LAYER_DIM
    learning_rate = 0.01
    clip_norm = 1.0
    RS = np.random.RandomState(seed)
    wt = np.abs(RS.rand(1, 1)) * -1
    W_t = RS.randn(HIDDEN_LAYER_DIM, 1)
    Wb_alpha = RS.randn(HIDDEN_LAYER_DIM, 1)
    Ws_alpha = RS.randn(HIDDEN_LAYER_DIM, 1)
    Wn_b = RS.randn(HIDDEN_LAYER_DIM, 1)
    Wn_s = RS.randn(HIDDEN_LAYER_DIM, 1)
    W_h = RS.randn(HIDDEN_LAYER_DIM, HIDDEN_LAYER_DIM) * 0.1 + np.diag(
        np.ones(HIDDEN_LAYER_DIM))  # Careful initialization
    W_1 = RS.randn(HIDDEN_LAYER_DIM, HIDDEN_LAYER_DIM)
    W_2 = RS.randn(HIDDEN_LAYER_DIM, HIDDEN_LAYER_DIM)
    W_3 = RS.randn(HIDDEN_LAYER_DIM, HIDDEN_LAYER_DIM)
    b_t = RS.randn(HIDDEN_LAYER_DIM, 1)
    b_alpha = RS.randn(HIDDEN_LAYER_DIM, 1)
    bn_b = RS.randn(HIDDEN_LAYER_DIM, 1)
    bn_s = RS.randn(HIDDEN_LAYER_DIM, 1)
    b_h = RS.randn(HIDDEN_LAYER_DIM, 1)
    Vt_h = RS.randn(1, HIDDEN_LAYER_DIM)
    Vt_v = RS.randn(1, 1)
    b_lambda = RS.randn(1, 1)
    Vh_alpha = RS.randn(2, HIDDEN_LAYER_DIM)
    Vv_alpha = RS.randn(2, 1)
    Va_b = RS.randn(MAX_SHARE, HIDDEN_LAYER_DIM)
    Va_s = RS.randn(MAX_SHARE, HIDDEN_LAYER_DIM)

    # The graph execution time depends on this parameter even though each
    # trajectory may contain much fewer events. So it is wise to set
    # it such that it is just above the total number of events likely
    # to be seen.
    momentum = 0.9
    max_events = 5000
    batch_size = 16
    T = 1254133800

    device_cpu = '/cpu:0'
    device_gpu = '/gpu:0'
    only_cpu = False
    save_dir = SAVE_DIR
    # Expected: './tpprl.summary/train-{}/'.format(run)
    summary_dir = None
    q = 0.0005
    set_wt_zero = False


class ExpRecurrentTrader:

    def __init__(self, wt, W_t, Wb_alpha, Ws_alpha, Wn_b, Wn_s,
                 W_h, W_1, W_2, W_3, b_t, b_alpha, bn_b, bn_s, b_h,
                 Vt_h, Vt_v, b_lambda, Vh_alpha, Vv_alpha, Va_b, Va_s,
                 num_hidden_states, sess, scope, batch_size, q, learning_rate, clip_norm,
                 summary_dir, save_dir, decay_steps, decay_rate, momentum,
                 device_cpu, device_gpu, only_cpu):
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

        # self.tf_max_events = T
        self.num_hidden_states = num_hidden_states

        self.scope = scope or type(self).__name__
        var_device = device_cpu if only_cpu else device_gpu
        with tf.device(device_cpu):
            # Global step needs to be on the CPU
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

            with tf.variable_scope(self.scope):
                with tf.variable_scope('hidden_state'):
                    with tf.device(var_device):
                        self.tf_W_t = tf.get_variable(name='W_t', shape=W_t.shape,
                                                      initializer=tf.constant_initializer(W_t))
                        self.tf_Wb_alpha = tf.get_variable(name='Wb_alpha', shape=Wb_alpha.shape,
                                                           initializer=tf.constant_initializer(Wb_alpha))
                        self.tf_Ws_alpha = tf.get_variable(name='Ws_alpha', shape=Ws_alpha.shape,
                                                           initializer=tf.constant_initializer(Ws_alpha))
                        self.tf_Wn_b = tf.get_variable(name='Wn_b', shape=Wn_b.shape,
                                                       initializer=tf.constant_initializer(Wn_b))
                        self.tf_Wn_s = tf.get_variable(name='Wn_s', shape=Wn_s.shape,
                                                       initializer=tf.constant_initializer(Wn_s))
                        self.tf_W_h = tf.get_variable(name='W_h', shape=W_h.shape,
                                                      initializer=tf.constant_initializer(W_h))
                        self.tf_W_1 = tf.get_variable(name='W_1', shape=W_1.shape,
                                                      initializer=tf.constant_initializer(W_1))
                        self.tf_W_2 = tf.get_variable(name='W_2', shape=W_2.shape,
                                                      initializer=tf.constant_initializer(W_2))
                        self.tf_W_3 = tf.get_variable(name='W_3', shape=W_3.shape,
                                                      initializer=tf.constant_initializer(W_3))
                        self.tf_b_t = tf.get_variable(name='b_t', shape=b_t.shape,
                                                      initializer=tf.constant_initializer(b_t))
                        self.tf_b_alpha = tf.get_variable(name='b_alpha', shape=b_alpha.shape,
                                                          initializer=tf.constant_initializer(b_alpha))
                        self.tf_bn_b = tf.get_variable(name='bn_b', shape=bn_b.shape,
                                                       initializer=tf.constant_initializer(bn_b))
                        self.tf_bn_s = tf.get_variable(name='bn_s', shape=bn_s.shape,
                                                       initializer=tf.constant_initializer(bn_s))
                        self.tf_b_h = tf.get_variable(name='b_h', shape=b_h.shape,
                                                      initializer=tf.constant_initializer(b_h))

                        # Needed to calculate the hidden state for one step.
                        self.tf_h_i = tf.get_variable(name='h_i', initializer=tf.zeros((self.num_hidden_states, 1),
                                                                                       dtype=self.tf_dtype))
                        self.tf_tau_i = tf.get_variable(name='tau_i', initializer=tf.zeros((self.num_hidden_states, 1),
                                                                                           dtype=self.tf_dtype))
                        self.tf_b_i = tf.get_variable(name='b_i', initializer=tf.zeros((self.num_hidden_states, 1),
                                                                                       dtype=self.tf_dtype))
                        self.tf_eta_i = tf.get_variable(name='eta_i', initializer=tf.zeros((self.num_hidden_states, 1),
                                                                                           dtype=self.tf_dtype))

                with tf.variable_scope('output'):
                    with tf.device(var_device):
                        self.tf_wt = tf.get_variable(name='wt', shape=wt.shape,
                                                     initializer=tf.constant_initializer(wt))
                        self.tf_Vt_h = tf.get_variable(name='Vt_h', shape=Vt_h.shape,
                                                       initializer=tf.constant_initializer(Vt_h))
                        self.tf_Vt_v = tf.get_variable(name='Vt_v', shape=Vt_v.shape,
                                                       initializer=tf.constant_initializer(Vt_v))
                        self.tf_b_lambda = tf.get_variable(name='b_lambda', shape=b_lambda.shape,
                                                           initializer=tf.constant_initializer(b_lambda))
                        self.tf_Vh_alpha = tf.get_variable(name='Vh_alpha', shape=Vh_alpha.shape,
                                                           initializer=tf.constant_initializer(Vh_alpha))
                        self.tf_Vv_alpha = tf.get_variable(name='Vv_alpha', shape=Vv_alpha.shape,
                                                           initializer=tf.constant_initializer(Vv_alpha))
                        self.tf_Va_b = tf.get_variable(name='Va_b', shape=Va_b.shape,
                                                       initializer=tf.constant_initializer(Va_b))
                        self.tf_Va_s = tf.get_variable(name='Va_s', shape=Va_s.shape,
                                                       initializer=tf.constant_initializer(Va_s))

                # Create a large dynamic_rnn kind of network which can calculate
                # the gradients for a given batch of simulations.
                with tf.variable_scope('training'):
                    self.tf_batch_rewards = tf.placeholder(name='rewards',
                                                           shape=(self.tf_batch_size, 1),
                                                           dtype=self.tf_dtype)
                    self.tf_batch_t_deltas = tf.placeholder(name='t_deltas',
                                                            shape=(self.tf_batch_size, 1),
                                                            dtype=self.tf_dtype)
                    self.tf_batch_seq_len = tf.placeholder(name='seq_len',
                                                           shape=(self.tf_batch_size, 1),
                                                           dtype=tf.int32)
                    self.tf_batch_last_interval = tf.placeholder(name='last_interval',
                                                                 shape=self.tf_batch_size,
                                                                 dtype=self.tf_dtype)
                    self.tf_batch_alpha_i = tf.placeholder(name='alpha_i',
                                                           shape=(self.tf_batch_size, 1),
                                                           dtype=tf.int32)
                    self.tf_batch_n_i = tf.placeholder(name='n_i',
                                                         shape=(self.tf_batch_size, 1),
                                                         dtype=tf.int32)
                    # Inferred batch size
                    inf_batch_size = tf.shape(self.tf_batch_t_deltas)[0]

                    self.tf_batch_init_h = tf.zeros(
                        name='init_h',
                        shape=(inf_batch_size, self.num_hidden_states),
                        dtype=self.tf_dtype
                    )
                    # Stacked version (for performance)

                    with tf.name_scope('stacked'):
                        with tf.device(var_device):
                            (self.W_t_mini, self.Wb_alpha_mini, self.Ws_alpha_mini,
                             self.Wn_b_mini, self.Wn_s_mini, self.W_h_mini,
                             self.W_1_mini, self.W_2_mini, self.W_3_mini,
                             self.b_t_mini, self.b_alpha_mini, self.bn_b_mini,
                             self.bn_s_mini, self.b_h_mini, self.wt_mini,
                             self.Vt_h_mini, self.Vt_v_mini, self.b_lambda_mini,
                             self.Vh_alpha_mini, self.Vv_alpha_mini, self.Va_b_mini,
                             self.Va_s_mini) = [
                                tf.stack(x, name=name)
                                for x, name in zip(
                                    zip(*[
                                        (tf.identity(self.tf_W_t), tf.identity(self.tf_Wb_alpha),
                                         tf.identity(self.tf_Ws_alpha), tf.identity(self.tf_Wn_b),
                                         tf.identity(self.tf_Wn_s), tf.identity(self.tf_W_h),
                                         tf.identity(self.tf_W_1), tf.identity(self.tf_W_2),
                                         tf.identity(self.tf_W_3), tf.identity(self.tf_b_t),
                                         tf.identity(self.tf_b_alpha), tf.identity(self.tf_bn_b),
                                         tf.identity(self.tf_bn_s), tf.identity(self.tf_b_h),
                                         tf.identity(self.tf_wt), tf.identity(self.tf_Vt_h),
                                         tf.identity(self.tf_Vt_v), tf.identity(self.tf_b_lambda),
                                         tf.identity(self.tf_Vh_alpha), tf.identity(self.tf_Vv_alpha),
                                         tf.identity(self.tf_Va_b), tf.identity(self.tf_Va_s))
                                        for _ in range(self.batch_size)
                                    ]),
                                    ['W_t', 'Wb_alpha', 'Ws_alpha', 'Wn_b', 'Wn_s', 'W_h', 'W_1', 'W_2', 'W_3',
                                     'b_t', 'b_alpha', 'bn_b', 'bn_s', 'b_h', 'wt', 'Vt_h', 'Vt_v', 'b_lambda',
                                     'Vh_alpha', 'Vv_alpha', 'Va_b', 'Va_s']
                                )
                            ]

                            # TODO: [1]*4?? ouptut_size??
                            self.rnn_cell_stack = TPPRExpMarkedCellStacked_finance(
                                hidden_state_size=(None, self.num_hidden_states),
                                output_size=[1] * 5 + [self.num_hidden_states] + [1],
                                tf_dtype=self.tf_dtype,
                                W_t=self.W_t_mini, Wb_alpha=self.Wb_alpha_mini,
                                Ws_alpha=self.Ws_alpha_mini, Wn_b=self.Wn_b_mini,
                                Wn_s=self.Wn_s_mini, W_h=self.W_h_mini,
                                W_1=self.W_1_mini, W_2=self.W_2_mini,
                                W_3=self.W_3_mini, b_t=self.b_t_mini,
                                b_alpha=self.b_alpha_mini, bn_b=self.bn_b_mini,
                                bn_s=self.bn_s_mini, b_h=self.b_h_mini, wt=self.wt_mini,
                                Vt_h=self.Vt_h_mini, Vt_v=self.Vv_alpha_mini,
                                b_lambda=self.b_lambda_mini, Vh_alpha=self.Vh_alpha_mini,
                                Vv_alpha=self.Vv_alpha_mini, Va_b=self.Va_b_mini, Va_s=self.Va_s_mini
                            )

                            ((self.h_states_stack, LL_log_terms_stack, LL_int_terms_stack, loss_terms_stack),
                             tf_batch_h_t_mini) = tf.nn.dynamic_rnn(
                                self.rnn_cell_stack,
                                inputs=(tf.expand_dims(self.tf_batch_alpha_i, axis=-1),
                                        tf.expand_dims(self.tf_batch_n_i, axis=-1),
                                        tf.expand_dims(self.tf_batch_t_deltas, axis=-1)),
                                sequence_length=tf.squeeze(self.tf_batch_seq_len, axis=-1),
                                dtype=self.tf_dtype,
                                initial_state=self.tf_batch_init_h
                            )

                            self.LL_log_terms_stack = tf.squeeze(LL_log_terms_stack, axis=-1)
                            self.LL_int_terms_stack = tf.squeeze(LL_int_terms_stack, axis=-1)
                            self.loss_terms_stack = tf.squeeze(loss_terms_stack, axis=-1)

                            # LL_last_term_stack = rnn_cell.last_LL(tf_batch_h_t_mini, self.tf_batch_last_interval)
                            # loss_last_term_stack = rnn_cell.last_loss(tf_batch_h_t_mini, self.tf_batch_last_interval)

                            self.LL_last_term_stack = self.rnn_cell_stack.last_LL(tf_batch_h_t_mini,
                                                                                  self.tf_batch_last_interval)
                            self.loss_last_term_stack = self.rnn_cell_stack.last_loss(tf_batch_h_t_mini,
                                                                                      self.tf_batch_last_interval)

                            self.LL_stack = (tf.reduce_sum(self.LL_log_terms_stack, axis=1) - tf.reduce_sum(
                                self.LL_int_terms_stack, axis=1)) + self.LL_last_term_stack

                            tf_seq_len = tf.squeeze(self.tf_batch_seq_len, axis=-1)
                            self.loss_stack = (self.q / 2) * (tf.reduce_sum(self.loss_terms_stack, axis=1) +
                                                              self.loss_last_term_stack)

                # with tf.name_scope('calc_u'):
                #     with tf.device(var_device):
                #         # These are operations needed to calculate u(t) in post-processing.
                #         # These can be done entirely in numpy-space, but since we have a
                #         # version in tensorflow, they have been moved here to avoid
                #         # memory leaks.
                #         # Otherwise, new additions to the graph were made whenever the
                #         # function calc_u was called.
                #
                #         self.calc_u_h_states = tf.placeholder(
                #             name='calc_u_h_states',
                #             shape=(self.tf_batch_size, self.num_hidden_states),
                #             dtype=self.tf_dtype
                #         )
                #         self.calc_u_batch_size = tf.placeholder(
                #             name='calc_u_batch_size',
                #             shape=(None,),
                #             dtype=tf.int32
                #         )
                #
                #         # TODO: formulas ??
                #         self.calc_u_c_is_init = tf.matmul(self.tf_Vt_h, self.tf_batch_init_h) + self.tf_b_lambda
                #         self.calc_u_c_is_rest = tf.squeeze(
                #             tf.matmul(
                #                 self.calc_u_h_states,
                #                 tf.tile(
                #                     tf.expand_dims(self.tf_Vt_h, 0),
                #                     [self.calc_u_batch_size[0], 1, 1]
                #                 )
                #             ) + self.tf_b_lambda,
                #             axis=-1,
                #             name='calc_u_c_is_rest'
                #         )
                #
                #         self.calc_u_is_own_event = tf.equal(self.tf_batch_b_idxes, 0)

                self.all_tf_vars = [self.tf_W_t, self.tf_Wb_alpha, self.tf_Ws_alpha,
                                    self.tf_Wn_b, self.tf_Wn_s, self.tf_W_h,
                                    self.tf_W_1, self.tf_W_2, self.tf_W_3,
                                    self.tf_b_t, self.tf_b_alpha, self.tf_bn_b,
                                    self.tf_bn_s, self.tf_b_h, self.tf_wt,
                                    self.tf_Vt_h, self.tf_Vt_v, self.tf_b_lambda,
                                    self.tf_Vh_alpha, self.tf_Vv_alpha, self.tf_Va_b,
                                    self.tf_Va_s]

                self.all_mini_vars = [self.W_t_mini, self.Wb_alpha_mini, self.Ws_alpha_mini,
                                      self.Wn_b_mini, self.Wn_s_mini, self.W_h_mini,
                                      self.W_1_mini, self.W_2_mini, self.W_3_mini,
                                      self.b_t_mini, self.b_alpha_mini, self.bn_b_mini,
                                      self.bn_s_mini, self.b_h_mini, self.wt_mini,
                                      self.Vt_h_mini, self.Vt_v_mini, self.b_lambda_mini,
                                      self.Vh_alpha_mini, self.Vv_alpha_mini, self.Va_b_mini,
                                      self.Va_s_mini]

                with tf.name_scope('stack_grad'):
                    with tf.device(var_device):
                        self.LL_grad_stacked = {x: tf.gradients(self.LL_stack, x)
                                                for x in self.all_mini_vars}
                        self.loss_grad_stacked = {x: tf.gradients(self.loss_stack, x)
                                                  for x in self.all_mini_vars}
                        self.avg_gradient_stack = []
                        avg_baseline = 0.0
                        # Removing the average reward + loss is not optimal baseline,
                        # but still reduces variance significantly.
                        coef = tf.squeeze(self.tf_batch_rewards, axis=-1) + self.loss_stack - avg_baseline
                        for x, y in zip(self.all_mini_vars, self.all_tf_vars):
                            LL_grad = self.LL_grad_stacked[x][0]
                            loss_grad = self.loss_grad_stacked[x][0]
                            # if self.set_wt_zero and y == self.tf_wt:
                            #     self.avg_gradient_stack.append(([0.0], y))
                            #     continue
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

                self.opt = tf.train.AdamOptimizer(
                    learning_rate=self.tf_learning_rate,
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
                    variable_summaries(tf.cast(self.tf_batch_seq_len, self.tf_dtype), name='batch_seq_len')

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
                   save_every=25):
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
            for iter_idx in range(num_iters):
                seed_end = seed_start + self.batch_size

                seeds = range(seed_start, seed_end)

                scenarios = [run_scenario(self, seed) for seed in seeds]

                f_d = get_feed_dict(self, scenarios)

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

                print('{} Run {}, LL {:.2f}±{:.2f}, loss {:.2f}±{:.2f}, Rwd {:.3f}±{:.3f}'
                      ' seeds {}--{}, grad_norm {:.2f}, step = {}'
                      ', lr = {:.5f}, wt={:.5f}, b_lambda={:.5f}'
                      .format(_now(), iter_idx,
                              mean_LL, std_LL,
                              mean_loss, std_loss,
                              mean_reward, std_reward,
                              seed_start, seed_end - 1,
                              grad_norm, step, lr,
                              self.sess.run(self.tf_wt)[0], self.sess.run(self.tf_b_lambda)[0]))

                # Ready for the next iter_idx.
                seed_start = seed_end

                if iter_idx % save_every == 0:
                    print('Saving model!')
                    self.saver.save(self.sess,
                                    chkpt_file,
                                    global_step=self.global_step, )

        finally:
            if pool is not None:
                pool.close()

            if with_summaries:
                train_writer.flush()

            print('Saving model!')
            self.saver.save(self.sess, chkpt_file, global_step=self.global_step, )


def read_raw_data():
    """ read raw_data """
    print("reading raw data")
    # folder = "/home/psupriya/MY_HOME/tpprl_finance/dataset/"
    # folder = "/home/supriya/MY_HOME/MPI-SWS/dataset"
    # folder = "/NL/tpprl-result/work/rl-finance/"
    # raw = pd.read_csv(folder + "/hourly_data/0_hour.csv")  # header names=['datetime', 'price'])
    raw = pd.read_csv(SAVE_DIR + "/daily_data/0_day.csv")
    df = pd.DataFrame(raw)
    return df


def get_feed_dict(trader, scenarios):
    """Produce a feed_dict for the given list of scenarios."""
    batch_size = len(scenarios)
    full_shape = (batch_size, 1)
    batch_rewards = np.asarray([s.reward() for s in scenarios])[:, np.newaxis]
    batch_last_interval = np.asarray([
        s.get_last_interval() for s in scenarios
    ], dtype=float)

    batch_seq_len = np.asarray([
        s.get_num_events() for s in scenarios
    ], dtype=float)[:, np.newaxis]

    batch_t_deltas = np.zeros(shape=full_shape, dtype=float)
    batch_alpha_i = np.zeros(shape=full_shape, dtype=int)
    batch_n_i = np.zeros(shape=full_shape, dtype=float)
    batch_init_h = np.zeros(shape=(batch_size, trader.num_hidden_states), dtype=float)

    for idx, scen in enumerate(scenarios):
        # They are sorted by time already.
        batch_len = int(batch_seq_len[idx])
        batch_alpha_i[idx, 0:batch_len] = scen.alpha_i
        batch_t_deltas[idx, 0:batch_len] = scen.time_deltas
        batch_n_i[idx, 0:batch_len] = scen.n_i

    return {
        trader.tf_batch_alpha_i: batch_alpha_i,
        trader.tf_batch_n_i: batch_n_i,
        trader.tf_batch_rewards: batch_rewards,
        trader.tf_batch_seq_len: batch_seq_len,
        trader.tf_batch_t_deltas: batch_t_deltas,
        trader.tf_batch_init_h: batch_init_h,
        trader.tf_batch_last_interval: batch_last_interval
    }


def run_scenario(trader, seed):
    raw_data = read_raw_data()
    wt = trader.sess.run(trader.tf_wt)
    W_t = trader.sess.run(trader.tf_W_t)
    Wb_alpha = trader.sess.run(trader.tf_Wb_alpha)
    Ws_alpha = trader.sess.run(trader.tf_Ws_alpha)
    Wn_b = trader.sess.run(trader.tf_Wn_b)
    Wn_s = trader.sess.run(trader.tf_Wn_s)
    W_h = trader.sess.run(trader.tf_W_h)
    W_1 = trader.sess.run(trader.tf_W_1)
    W_2 = trader.sess.run(trader.tf_W_2)
    W_3 = trader.sess.run(trader.tf_W_3)
    b_t = trader.sess.run(trader.tf_b_t)
    b_alpha = trader.sess.run(trader.tf_b_alpha)
    bn_b = trader.sess.run(trader.tf_bn_b)
    bn_s = trader.sess.run(trader.tf_bn_s)
    b_h = trader.sess.run(trader.tf_b_h)
    Vt_h = trader.sess.run(trader.tf_Vt_h)
    Vt_v = trader.sess.run(trader.tf_Vt_v)
    b_lambda = trader.sess.run(trader.tf_b_lambda)
    Vh_alpha = trader.sess.run(trader.tf_Vh_alpha)
    Vv_alpha = trader.sess.run(trader.tf_Vv_alpha)
    Va_b = trader.sess.run(trader.tf_Va_b)
    Va_s = trader.sess.run(trader.tf_Va_s)
    # initiate agent/broadcaster
    # agent = SimpleStrategy(time_between_trades_secs=5)
    # agent = BollingerBandStrategy(window=20, num_std=2)
    agent = RLStrategy(wt, W_t, Wb_alpha, Ws_alpha, Wn_b, Wn_s, W_h, W_1, W_2, W_3, b_t, b_alpha, bn_b, bn_s, b_h, Vt_h,
                       Vt_v, b_lambda, Vh_alpha, Vv_alpha, Va_b, Va_s, seed)
    # start time is set to '2009-09-28 09:30:00' i.e. 9:30 am of 28sept2009: 1254130200
    # max time T is set to '2009-09-28 16:00:00' i.e. same day 4pm: 1254153600
    mgr = Environment(T=1254133800, time_gap="second", raw_data=raw_data, agent=agent, start_time=1254130200, seed=seed)
    # v_last = mgr.simulator()
    return mgr



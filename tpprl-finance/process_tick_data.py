"""
TODO: negative reward??
"""
import numpy as np
import pandas as pd
from collections import deque

HIDDEN_LAYER_DIM = 8
MAX_AMT = 1000.0
MAX_SHARE = 100
BASE_CHARGES = 1.0
PERCENTAGE_CHARGES = 0.001

folder = "/NL/tpprl-result/work/rl-finance/"
# folder = "/home/supriya/MY_HOME/MPI-SWS/dataset/"
# folder = "/NL/tpprl-result/work/rl-finance/"


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
        df.to_csv(folder + output_file, index=False)
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


class SimpleStrategy(Strategy):
    def __init__(self, time_between_trades_secs=5):
        super(SimpleStrategy, self).__init__()
        self.start_time = None
        self.own_events = 1
        self.time_between_trades_secs = time_between_trades_secs
        self.last_action = None
        print("Using Simple Strategy")

    def get_next_action_time(self, event):
        if self.current_amt <= (event.v_curr + BASE_CHARGES + event.v_curr * PERCENTAGE_CHARGES):
            return np.inf

        if self.start_time is None:
            # This is the first event
            self.start_time = event.t_i

        if event.is_trade_event():
            self.own_events += 1
            self.last_action = event.alpha_i

        return self.start_time + self.own_events * self.time_between_trades_secs

    def get_next_action_item(self, event):
        if self.last_action is None:
            alpha_i = 0
            n_i = 1
        else:
            alpha_i = int(self.last_action) ^ 1
            n_i = 1
        # update current amt
        if alpha_i == 0:
            self.current_amt -= n_i * event.v_curr
        elif alpha_i == 1:
            self.current_amt += n_i * event.v_curr
        # subtract the fixed transaction charges
        if n_i != 0:
            self.current_amt -= BASE_CHARGES
        # subtract the percentage transaction charges
        self.current_amt -= event.v_curr * n_i * PERCENTAGE_CHARGES
        assert self.current_amt > 0
        return alpha_i, n_i


class BollingerBandStrategy(Strategy):
    def __init__(self, window, num_std):
        super(BollingerBandStrategy, self).__init__()
        self.window = window
        self.num_std = num_std
        self.history = deque(maxlen=self.window)
        self.bollinger_band = None
        print("Using Bollinger Band Strategy")

    def get_next_action_time(self, event):
        if event.is_trade_event():
            return np.inf
        t_i = event.t_i
        self.history.append(event.v_curr)
        # even after appending current share value, if the size of history is not equal to window then return inf
        if len(self.history) < self.window:
            return np.inf

        if self.current_amt <= (event.v_curr + BASE_CHARGES + event.v_curr * PERCENTAGE_CHARGES):
            return np.inf

        # TODO: order one calculation
        self.bollinger_band = pd.DataFrame(list(self.history), columns=["price"])
        rolling_mean = self.bollinger_band["price"].rolling(window=self.window).mean()
        rolling_std = self.bollinger_band["price"].rolling(window=self.window).std()

        self.bollinger_band["Bollinger_High"] = rolling_mean + (rolling_std * self.num_std)
        self.bollinger_band["Bollinger_Low"] = rolling_mean - (rolling_std * self.num_std)

        if float(self.bollinger_band.Bollinger_Low.tail(1)) < float(self.bollinger_band.price.tail(1)) < float(
                self.bollinger_band.Bollinger_High.tail(1)):
            return np.inf
        return t_i

    def get_next_action_item(self, event):
        alpha_i = -1
        n_i = 1
        if float(self.bollinger_band.price.tail(1)) <= float(
                self.bollinger_band.Bollinger_Low.tail(1)) and self.current_amt >= float(
                self.bollinger_band.price.tail(1)):
            alpha_i = 0  # buy if current price is less than Bollinger Lower Band
        elif float(self.bollinger_band.price.tail(1)) >= float(
                self.bollinger_band.Bollinger_High.tail(1)) and self.owned_shares > 0:
            alpha_i = 1  # sell if current price is more than Bollinger Higher Band
        else:
            n_i = 0
        # update current amt
        if alpha_i == 0:
            self.current_amt -= n_i * event.v_curr
        elif alpha_i == 1:
            self.current_amt += n_i * event.v_curr
        # subtract the fixed transaction charges
        if n_i != 0:
            self.current_amt -= BASE_CHARGES
        # subtract the percentage transaction charges
        a = event.v_curr * n_i * PERCENTAGE_CHARGES
        assert self.current_amt>a
        self.current_amt -= event.v_curr * n_i * PERCENTAGE_CHARGES
        assert self.current_amt > 0
        return alpha_i, n_i


class RLStrategy(Strategy):
    def __init__(self, wt, W_t, Wb_alpha, Ws_alpha, Wn_b, Wn_s,
                 W_h, W_1, W_2, W_3, b_t, b_alpha, bn_b, bn_s, b_h,
                 Vt_h, Vt_v, b_lambda, Vh_alpha, Vv_alpha, Va_b, Va_s):
        super(RLStrategy, self).__init__()
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
        self.u = np.random.uniform()
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

        if self.current_amt <= (BASE_CHARGES + event.v_curr * PERCENTAGE_CHARGES):
            return np.inf

        prev_q = self.Q
        if event.is_trade_feedback or self.curr_time is None:
            self.u = np.random.uniform()
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
        alpha_i = np.random.choice(np.array([0, 1]), p=np.squeeze(prob_alpha))

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
            n_i = np.random.choice(np.arange(MAX_SHARE+1), p=np.squeeze(prob_n))
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
            n_i = np.random.choice(np.arange(MAX_SHARE+1), p=np.squeeze(prob_n))
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
        return self.loglikelihood


class Environment:
    def __init__(self, T, time_gap, raw_data, agent, start_time):
        self.T = T
        self.state = State(curr_time=start_time)
        self.time_gap = time_gap
        self.raw_data = raw_data
        self.agent = agent

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
        return v_last


def read_raw_data():
    """ read raw_data """
    print("reading raw data")
    # folder = "/home/psupriya/MY_HOME/tpprl_finance/dataset/"
    # folder = "/home/supriya/MY_HOME/MPI-SWS/dataset"
    # folder = "/NL/tpprl-result/work/rl-finance/"
    # raw = pd.read_csv(folder + "/hourly_data/0_hour.csv")  # header names=['datetime', 'price'])
    raw = pd.read_csv(folder + "/daily_data/0_day.csv")
    df = pd.DataFrame(raw)
    return df


if __name__ == '__main__':
    raw_data = read_raw_data()
    wt = np.random.uniform(size=(1, 1))
    W_t = np.zeros((8, 1))
    Wb_alpha = np.zeros((8, 1))
    Ws_alpha = np.zeros((8, 1))
    Wn_b = np.zeros((8, 1))
    Wn_s = np.zeros((8, 1))
    W_h = np.zeros((8, 8))
    W_1 = np.zeros((8, 8))
    W_2 = np.zeros((8, 8))
    W_3 = np.zeros((8, 8))
    b_t = np.zeros((8, 1))
    b_alpha = np.zeros((8, 1))
    bn_b = np.zeros((8, 1))
    bn_s = np.zeros((8, 1))
    b_h = np.zeros((8, 1))
    Vt_h = np.zeros((1, 8))
    Vt_v = np.zeros((1, 1))
    b_lambda = np.zeros((1, 1))
    Vh_alpha = np.zeros((2, 8))
    Vv_alpha = np.zeros((2, 1))
    Va_b = np.zeros((100, 8))
    Va_s = np.zeros((100, 8))
    # initiate agent/broadcaster
    # agent = SimpleStrategy(time_between_trades_secs=5)
    # agent = BollingerBandStrategy(window=20, num_std=2)
    agent = RLStrategy(wt, W_t, Wb_alpha, Ws_alpha, Wn_b, Wn_s, W_h, W_1, W_2, W_3, b_t, b_alpha, bn_b, bn_s, b_h, Vt_h, Vt_v, b_lambda, Vh_alpha, Vv_alpha, Va_b, Va_s)
    # start time is set to '2009-09-28 09:30:00' i.e. 9:30 am of 28sept2009: 1254130200
    # max time T is set to '2009-09-28 16:00:00' i.e. same day 4pm: 1254153600
    mgr = Environment(T=1254133800, time_gap="second", raw_data=raw_data, agent=agent, start_time=1254130200)
    v_last = mgr.simulator()
    method = "RL"
    output_file = "/results_{}_strategy/output_event_{}_0_day.csv".format(method, method)
    event_df = mgr.get_state().get_dataframe(output_file)
    reward = reward_fn(events=event_df, v_last=v_last)
    print("reward = ", reward)
    with open(folder+"/results_{}_strategy/reward_0_day.txt".format(method), "w") as rwd:
        rwd.write("reward:{}".format(reward))

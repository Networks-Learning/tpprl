"""
TODO: bollinger strategy takes current time and returns current time=> creating forever loop=> time is never updated
TODO: to accommodate data into memory, split interpolate.csv into each hour ticks and each day tick files
"""
import numpy as np
import pandas as pd
from collections import deque

HIDDEN_LAYER_DIM = 8
MAX_AMT = 1000.0
MAX_SHARE = 100


def reward_fn(events, v_last):
    reward = MAX_AMT
    owned_shares = 0
    print("calculating reward...")
    for (_idx, event) in events.iterrows():
        if event.alpha_i == 0:
            reward -= event.n_i*event.v_curr
            owned_shares += event.n_i
        elif event.alpha_i == 1:
            reward += event.n_i * event.v_curr
            owned_shares -= event.n_i
    reward += owned_shares*v_last
    return reward


class Action:
    def __init__(self, alpha, n):
        self.alpha = alpha
        self.n = n

    def __str__(self):
        return "<{}: {}>".format("Sell" if self.alpha > 0 else "Buy", self.n)


class Feedback:
    def __init__(self, t_i, v_curr, sample_u):
        self.t_i = t_i
        self.v_curr = v_curr
        self.sample_u = sample_u

    def is_trade_event(self):
        return self.sample_u

    def is_tick_event(self):
        return not self.sample_u


class TradeFeedback(Feedback):
    def __init__(self, t_i, v_curr, alpha_i, n_i):
        super(TradeFeedback, self).__init__(t_i, v_curr, sample_u=True)
        self.alpha_i = alpha_i
        self.n_i = n_i


class TickFeedback(Feedback):
    def __init__(self, t_i, v_curr):
        super(TickFeedback, self).__init__(t_i, v_curr, sample_u=False)


class State:
    def __init__(self, curr_time):
        self.time = curr_time
        self.events = []

    def apply_event(self, event):
        self.events.append(event)
        self.time = event.t_i
        if event.alpha_i == 0:
            print("* BUY {} shares at price of {} at time {}".format(event.n_i, event.v_curr, event.t_i))
        else:
            print("* SELL {} shares at price of {} at time {}".format(event.n_i, event.v_curr, event.t_i))

    def get_dataframe(self, output_file):
        df = pd.DataFrame.from_records(
            [{"t_i": event.t_i,
              "alpha_i": event.alpha_i,
              "n_i": event.n_i,
              "v_curr": event.v_curr,
              "sample_u": event.sample_u} for event in self.events])
        print("\n saving events:")
        print(df[:2].values)
        # folder = "/home/psupriya/MY_HOME/tpprl_finance/dataset/"
        folder = "/home/supriya/MY_HOME/MPI-SWS/dataset/"
        df.to_csv(folder + output_file, index=False)
        return df


class Strategy:
    def __init__(self):
        self.current_amt = MAX_AMT
        self.owned_shares = 0

    def get_next_action_time(self, event):
        return NotImplemented

    def get_next_action_item(self):
        return NotImplemented

    def update_owned_shares(self, event):
        if event.alpha_i == 0:
            self.owned_shares += event.n_i
            self.current_amt -= event.n_i * event.v_curr
        elif event.alpha_i == 1:
            self.owned_shares -= event.n_i
            self.current_amt += event.n_i * event.v_curr


class SimpleStrategy(Strategy):
    def __init__(self, time_between_trades_secs=5):
        super(SimpleStrategy, self).__init__()
        self.start_time = None
        self.own_events = 1
        self.time_between_trades_secs = time_between_trades_secs
        self.last_action = None
        print("Using Simple Strategy")

    def get_next_action_time(self, event):
        if self.start_time is None:
            # This is the first event
            self.start_time = event.t_i

        if event.is_trade_event():
            self.own_events += 1
            self.last_action = event.alpha_i

        return self.start_time + self.own_events * self.time_between_trades_secs

    def get_next_action_item(self):
        if self.last_action is None:
            alpha_i = 0
            n_i = 1
        else:
            alpha_i = int(self.last_action) ^ 1
            n_i = 1

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
        if event.t_i == np.inf:
            return np.inf
        t_i = event.t_i
        self.history.append(event.v_curr)
        # even after appending current share value, if the size of history is not equal to window then return inf
        if len(self.history) < self.window:
            return np.inf

        self.bollinger_band = pd.DataFrame(list(self.history), columns=["price"])
        rolling_mean = self.bollinger_band["price"].rolling(window=self.window).mean()
        rolling_std = self.bollinger_band["price"].rolling(window=self.window).std()

        self.bollinger_band["Bollinger_High"] = rolling_mean + (rolling_std * self.num_std)
        self.bollinger_band["Bollinger_Low"] = rolling_mean - (rolling_std * self.num_std)

        if float(self.bollinger_band.Bollinger_Low.tail(1)) < float(self.bollinger_band.price.tail(1)) < float(self.bollinger_band.Bollinger_High.tail(1)):
            return np.inf
        return t_i

    def get_next_action_item(self):
        alpha_i = -1
        n_i = 1
        if float(self.bollinger_band.price.tail(1)) <= float(self.bollinger_band.Bollinger_Low.tail(1)) and self.current_amt >= float(self.bollinger_band.price.tail(1)):
            alpha_i = 0  # buy if current price is less than Bollinger Lower Band
        elif float(self.bollinger_band.price.tail(1)) >= float(self.bollinger_band.Bollinger_High.tail(1)) and self.owned_shares > 0:
            alpha_i = 1  # sell if current price is more than Bollinger Higher Band
        else:
            n_i = 0
        return alpha_i, n_i


class RLStrategy:
    def __init__(self, wt, W_t, Wb_alpha, Ws_alpha, Wn_b, Wn_s,
                 W_h, W_1, W_2, W_3, b_t, b_alpha, bn_b, bn_s, b_h,
                 V_t, Vh_alpha, Vv_alpha, Va_b, Va_s):
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
        self.V_t = V_t
        self.Vh_alpha = Vh_alpha
        self.Vv_alpha = Vv_alpha
        self.Va_b = Va_b
        self.Va_s = Va_s

        self.h_i = np.zeros(HIDDEN_LAYER_DIM)
        self.t_0 = 0.0
        self.Q = 1.0
        self.c1 = 1.0
        self.current_amt = MAX_AMT
        self.owned_shares = 0
        self.u = np.random.uniform()

    def get_next_action_time(self, event):

        # if this method is called after buying/selling action, then sample new u
        # if event.sample_u:
        # encode event details
        # tau_i = np.array(self.W_t).dot((event.t_i - self.t_0)) + self.b_t
        # b_i = np.array(self.Wb_alpha).dot(1-event.alpha_i) + np.array(self.Ws_alpha).dot(event.alpha_i) + self.b_alpha
        # if event.alpha_i == 0:
        #     eta_i = np.array(self.Wn_b).dot(event.n_i) + self.bn_b
        # else:
        #     eta_i = np.array(self.Wn_s).dot(event.n_i) + self.bn_s
        #
        # # update h_i
        # self.h_i = np.tanh(np.array(self.W_h).dot(self.h_i) + np.array(self.W_1).dot(tau_i)
        #                    + np.array(self.W_2).dot(b_i) + np.array(self.W_3).dot(eta_i) + self.b_h)

        # sample new u, t_i, alpha_i
        # self.c1 = np.exp(np.array(self.V_t).dot(self.h_i))
        # self.u = np.random.uniform()
        # D = 1 - (self.wt / np.exp(self.c1)) * np.log((1 - self.u) / self.Q)
        # t_i = self.t0 + (1 / self.w) * np.log(D)
        # sample alpha_i
        # p_alpha = 1 / (1 + np.exp(-self.Vh_alpha * self.h_i))
        # alpha_i = np.random.choice(np.array([0, 1]), p=p_alpha)
        t_i = event.t_i + 60
        return t_i

    def get_number_of_share(self, event):
        if event.alpha_i == 0:
            A = np.array(self.Va_b).dot(self.h_i)
            max_share_buy = min(MAX_SHARE, np.floor(self.current_amt / event.v_curr)) + 1  # to allow buying zero shares
            mask = np.append(np.ones(max_share_buy), np.zeros(MAX_SHARE + 1 - max_share_buy))  # total size is 101
            masked_A = mask * A
            exp = np.exp(masked_A)
            prob = exp / np.sum(exp)
            event.n_i = np.random.choice(np.arange(MAX_SHARE), p=prob)
            self.owned_shares += event.n_i
            self.current_amt -= event.v_curr * event.n_i
        else:
            A = np.array(self.Va_b).dot(self.h_i)
            max_share_sell = min(MAX_SHARE, self.owned_shares) + 1  # to allow buying zero shares
            mask = np.append(np.ones(max_share_sell), np.zeros(MAX_SHARE + 1 - max_share_sell))  # total size is 101
            masked_A = mask * A
            exp = np.exp(masked_A)
            prob = exp / np.sum(exp)
            event.n_i = np.random.choice(np.arange(MAX_SHARE), p=prob)
            self.owned_shares -= event.n_i
            self.current_amt += event.v_curr * event.n_i
        return event


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
        elif self.time_gap == 'second':
            self.tick_data = self.raw_data.groupby(self.raw_data["datetime"], as_index=False).last()

            print(self.tick_data.head())
        else:
            raise ValueError("Time gap value '{}' not understood.".format(self.time_gap))

    def get_state(self):
        return self.state

    def simulator(self):
        row_iterator = self.tick_data.iterrows()
        first_tick = next(row_iterator)[1]
        current_event = TickFeedback(t_i=first_tick.datetime, v_curr=first_tick.price)
        v_last = current_event.v_curr
        print("trading..")

        for (_idx, next_tick) in row_iterator:
            while self.state.time <= self.T:
                next_agent_action_time = self.agent.get_next_action_time(current_event)
                if next_agent_action_time > next_tick.datetime:
                    current_event = TickFeedback(t_i=next_tick.datetime, v_curr=next_tick.price)
                    print("reading market value at time {}".format(current_event.t_i))
                    break
                else:
                    alpha_i, n_i = self.agent.get_next_action_item()
                    current_event = TradeFeedback(t_i=next_agent_action_time, v_curr=next_tick.price,
                                                  alpha_i=alpha_i, n_i=n_i)
                    self.agent.update_owned_shares(current_event)

                self.state.apply_event(current_event)
                v_last = current_event.v_curr
        return v_last


def read_raw_data():
    """ read raw_data """
    print("reading raw data")
    # folder = "/home/psupriya/MY_HOME/tpprl_finance/dataset/"
    folder = "/home/supriya/MY_HOME/MPI-SWS/dataset"
    raw = pd.read_csv(folder + "/first_hour_data.csv")  # header names=['datetime', 'price'])
    df = pd.DataFrame(raw)
    return df


if __name__ == '__main__':
    raw_data = read_raw_data()

    # initiate agent/broadcaster
    agent = SimpleStrategy(time_between_trades_secs=5)
    # agent = BollingerBandStrategy(window=20, num_std=2)

    # start time is set to '2009-09-28 09:30:00' i.e. 9:30 am of 28sept2009: 1254130200
    # max time T is set to '2009-09-28 16:00:00' i.e. same day 4pm: 1254153600
    mgr = Environment(T=1254133800, time_gap="second", raw_data=raw_data, agent=agent, start_time=1254130200)
    v_last = mgr.simulator()

    output_file = "output_event_simple_1hr_data.csv"
    event_df = mgr.get_state().get_dataframe(output_file)
    reward = reward_fn(events=event_df, v_last=v_last)
    print("reward = ", reward)

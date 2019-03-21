import numpy as np
import pandas as pd

HIDDEN_LAYER_DIM = 8
MAX_AMT = 1000
MAX_SHARE = 100


class ActionEvent:
    def __init__(self, t_i, v_curr, alpha_i, n_i):
        self.t_i = t_i
        self.v_curr = v_curr
        self.sample_u = True
        self.alpha_i = alpha_i
        self.n_i = n_i


class ReadEvent:
    def __init__(self, t_i, v_curr):
        self.t_i = t_i
        self.v_curr = v_curr
        self.sample_u = False


class State:
    def __init__(self, curr_time):
        self.time = curr_time
        self.events = []

    def apply_event(self, event):
        self.events.append(event)
        self.time = event.t_i

    def get_dataframe(self):
        return NotImplemented
        # folder = "/home/psupriya/MY_HOME/tpprl_finance/dataset/"
        #
        # raw = pd.read_csv(folder + "/IVE_tickbidask.csv", names=["date","time","price","bid","ask","size"])
        # df = pd.DataFrame()
        # df["datetime"] = pd.to_datetime(raw["date"]+" "+raw['time']).astype(int)//10**9 # store time in unix timestamp and store till seconds
        # df["price"] = raw["price"]
        # print(list(df)) # prints header
        # return df


class Manager:
    def __init__(self, T, time_gap, raw_data, agent, start_time):
        self.T = T
        self.state = State(curr_time=start_time)
        self.time_gap = time_gap
        self.raw_data = raw_data
        self.agent = agent

        # for reading market value per minute
        if self.time_gap == "minute":
            self.tick_data = self.raw_data.groupby(self.raw_data["datetime"].dt.minute).last()
        elif self.time_gap == 'second':
            # by default reading market value per second
            self.tick_data = self.raw_data.groupby(self.raw_data["datetime"].dt.second).last()
        else:
            raise ValueError("Time gap value '{}' not understood.".format(self.time_gap))

    def simulator(self):
        last_event = ReadEvent(t_i=1254130200, v_curr=50.79)

        for (_idx, row) in self.tick_data.iterrows():
            while True:
                next_agent_action = self.agent.get_next_action(last_event)
                if next_agent_action[0] > row.datetime:
                    last_event = ReadEvent(t_i=row.datetime, v_curr=row.price)
                    break
                else:
                    last_event = ActionEvent(t_i=row.datetime, v_curr=row.price, alpha_i=next_agent_action[1], n_i=next_agent_action[2])
                    # last_event = self.agent.get_number_of_share(last_event)
                self.state.apply_event(last_event)


class Policy:
    def __init__(self, wt, W_t, Wb_alpha, Ws_alpha, Wn_b, Wn_s,
                 W_h, W_1, W_2, W_3, b_t, b_alpha, bn_b, bn_s, b_h,
                 V_t, Vh_alpha, Vv_alpha, Va_b, Va_s):
        self.wt = wt
        self.W_t = W_t
        self.Wb_alpha = Wb_alpha
        self. Ws_alpha = Ws_alpha
        self. Wn_b = Wn_b
        self. Wn_s = Wn_s
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

    def get_next_action(self, event):

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
        t_i = event.t_i + 5
        alpha_i = int(event.alpha_i) ^ 1
        n_i = 1
        return t_i, alpha_i, n_i

    def get_number_of_share(self, event):
        if event.alpha_i == 0:
            A = np.array(self.Va_b).dot(self.h_i)
            max_share_buy = min(MAX_SHARE, np.floor(self.current_amt/event.v_curr)) + 1 # to allow buying zero shares
            mask = np.append(np.ones(max_share_buy), np.zeros(MAX_SHARE+1-max_share_buy)) # total size is 101
            masked_A = mask * A
            exp = np.exp(masked_A)
            prob = exp/np.sum(exp)
            event.n_i = np.random.choice(np.arange(MAX_SHARE), p=prob)
            self.owned_shares += event.n_i
            self.current_amt -= event.v_curr*event.n_i
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

# TODO: broadcaster code
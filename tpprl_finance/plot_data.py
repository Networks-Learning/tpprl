import plotly.plotly as py
import plotly.graph_objs as go

import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import datetime

SAVE_DIR = "/NL/tpprl-result/work/rl-finance"
RS = np.random.RandomState(seed=42)
start_num = 0  # 0
end_num = 8925


def plotly_code():
    vals = list(range(start_num, end_num))
    file_num = RS.choice(a=vals)
    raw = pd.read_csv(SAVE_DIR + "/per_minute_daily_data/{}_day.csv".format(file_num))
    print(file_num)
    data = [go.Scatter(x=raw.datetime, y=raw.price)]
    py.iplot(data, filename="plot_{}".format(file_num))


def matplotlib_code():
    # vals = list(range(start_num, end_num))
    # file_num = RS.choice(a=vals)
    # array_file_nums = [line.rstrip('\n') for line in open(SAVE_DIR + "/results_bollinger_strategy/files_selected.txt")]
    array_file_nums = [10552]
    for file_num in array_file_nums:
        day_data = pd.read_csv(SAVE_DIR + "/per_minute_daily_data/{}_day.csv".format(file_num))
        print(file_num)
        raw =pd.DataFrame()
        raw['datetime'] = pd.to_datetime(day_data["datetime"]*10**9)
        raw['market'] = day_data['price']

        print("bollinger_data:")
        bollinger_data = pd.read_csv(
            SAVE_DIR + "/results_bollinger_strategy/output_event_bollinger_{}_day.csv".format(file_num))
        print(bollinger_data.head(2))

        bollinger_data_buy = bollinger_data.loc[bollinger_data['alpha_i'] == 0]
        bollinger_data_sell = bollinger_data.loc[bollinger_data['alpha_i'] == 1]

        bollinger_df_buy = pd.DataFrame()
        bollinger_df_buy['datetime'] = pd.to_datetime(bollinger_data_buy.t_i*10**9)
        # bollinger_df_buy['int_datetime'] = bollinger_df_buy['datetime'].apply(lambda dt: datetime.datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute, 0))
        bollinger_df_buy['Bollinger_buy'] = bollinger_data_buy.v_curr

        bollinger_df_sell = pd.DataFrame()
        bollinger_df_sell['datetime'] = pd.to_datetime(bollinger_data_sell.t_i * 10 ** 9)
        bollinger_df_sell['Bollinger_sell'] = bollinger_data_sell.v_curr

        print("\nRL_data:")
        epoch = 0
        rl_data = pd.read_csv(SAVE_DIR + "/results_TF_RL/run_4May2019_12hr51min40sec/val_simulations/{}_epoch_{}_day.csv".format(epoch,file_num))
        print(rl_data.head(2))

        rl_data_buy = rl_data.loc[rl_data['alpha_i'] == 0]
        rl_data_sell = rl_data.loc[rl_data['alpha_i'] == 1]

        rl_df_buy = pd.DataFrame()
        rl_df_buy['datetime'] = pd.to_datetime(rl_data_buy.t_i*10**9)
        # rl_df_buy['int_datetime'] = rl_df_buy['datetime'].apply(lambda dt: datetime.datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute, 0))
        rl_df_buy['RL_buy'] = rl_data_buy.v_curr

        rl_df_sell = pd.DataFrame()
        rl_df_sell['datetime'] = pd.to_datetime(rl_data_sell.t_i * 10 ** 9)
        rl_df_sell['RL_sell'] = rl_data_sell.v_curr

        # common_buy = pd.merge(bollinger_df_buy, rl_df_buy, how='inner', on=['int_datetime'])

        # plot
        plt.plot_date(x=raw.datetime, y=raw.market, fmt='k-')

        plt.plot_date(x=bollinger_df_buy.datetime, y=bollinger_df_buy.Bollinger_buy, fmt='ro')
        plt.plot_date(x=bollinger_df_sell.datetime, y=bollinger_df_sell.Bollinger_sell, fmt='bo')

        plt.plot_date(x=rl_df_buy.datetime, y=rl_df_buy.RL_buy, fmt='rx')
        plt.plot_date(x=rl_df_sell.datetime, y=rl_df_sell.RL_sell, fmt='bx')

        # plt.plot_date(x=common_buy.datetime, y=common_buy.price, fmt='g^')
        plt.legend(loc='upper left')
        plt.xlabel("date")
        plt.ylabel("share price")
        plt.grid(True)
        import os
        if os.path.exists(SAVE_DIR+"/results_TF_RL/run_4May2019_12hr51min40sec/val_plots"):
            os.makedirs(SAVE_DIR+"/results_TF_RL/run_4May2019_12hr51min40sec/val_plots")
        plt.savefig(SAVE_DIR+"/results_TF_RL/run_4May2019_12hr51min40sec/val_plots/{}_epoch_{}_day.png".format(epoch, file_num))
        # plt.show()


if __name__ == '__main__':
    matplotlib_code()

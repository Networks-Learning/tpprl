import plotly.plotly as py
import plotly.graph_objs as go

import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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
    file_num = 10552
    day_data = pd.read_csv(SAVE_DIR + "/per_minute_daily_data/{}_day.csv".format(file_num))
    print(file_num)
    raw =pd.DataFrame()
    raw['datetime'] = pd.to_datetime(day_data["datetime"]*10**9)
    raw['price'] = day_data['price']

    bollinger_data = pd.read_csv(
        SAVE_DIR + "/results_bollinger_strategy/output_event_bollinger_{}_day.csv".format(file_num))
    bollinger_df = pd.DataFrame()
    bollinger_df['datetime'] = pd.to_datetime(bollinger_data.t_i*10**9)
    bollinger_df['price'] = bollinger_data.v_curr

    plt.plot_date(x=raw.datetime, y=raw.price, fmt='k-')
    plt.plot_date(x=bollinger_df.datetime, y=bollinger_df.price, fmt='ro')
    plt.xlabel("date")
    plt.ylabel("share price")
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    matplotlib_code()

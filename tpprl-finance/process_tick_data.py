import pandas as pd


def read_data():
    folder = "/home/psupriya/MY_HOME/tpprl_finance/dataset/"

    data = pd.read_csv(folder + "/sample_tick_data.csv", header=None)
    df_raw = pd.DataFrame(data)
    df = pd.DataFrame()
    df["datetime"] = pd.to_datetime(df_raw["date"] + " " + df_raw["time"])
    df["price"] = df_raw["price"]
    print(list(df))
    return df


def simulator(df, time_gap):
    # by default reading market value per second
    tick_data = df.groupby(df["datetime"].dt.second).last()

    # for reading market value per minute
    if time_gap == "minute":
        tick_data = df.groupby(df["datetime"].dt.minute).last()
    # else read market value only after change of delta_v
    # call get_next_action from policy class to get sampled time step
    # if sampled t is earlier than next time step to read market value then call synchronous feedback from policy to take action
    # else call asynch feedback to read market value and re-sample time
    return tick_data

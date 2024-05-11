import requests
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from gp import *
from lib import *

NUM_FOLDS=5




def main():

    # TR timezone is UTC+3 since 2016
    trtz=dt.timezone(dt.timedelta(hours=3))
    # 1 April 2019
    start_date=dt.datetime(2019,4,1,tzinfo=trtz)
    # 1 April 2024
    end_date=dt.datetime(2024,4,1,tzinfo=trtz)

    time_series=fetch_epias_exit_nomination(start_date,end_date)

    print(time_series.head())
    print(f"Dataset size: {len(time_series)}")

    # Train-test split is from 1 April 2023 onwards
    thres_date=dt.datetime(2023,4,1,tzinfo=trtz)

    train_df=time_series[time_series['date']<thres_date]
    test_df=time_series[time_series['date']>thres_date]

    print(f"Training set size: {len(train_df)}")
    print(f"Testing set size: {len(test_df)}")

if __name__ == "__main__":
    main()

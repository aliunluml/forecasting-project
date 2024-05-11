import requests
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from gp import *
from lib import *

NUM_FOLDS=5




def figure1(train,test):
    fig1,ax1=plt.subplots()
    # We unpack a list containing a single line object
    line11,=ax1.plot(train['date'],train['exitNominationAmount'],color='b',label='Train')
    line12,=ax1.plot(test['date'],test['exitNominationAmount'],color='r',label='Test')
    ax1.legend(handles=[line11, line12])
    ax1.set_title('Natural Gas Time-series')
    ax1.set_xlabel('Days')
    # Gas is measured in standard cubic meters.
    ax1.set_ylabel('Transmission Output Volume ($Sm^3$)')
    plt.show()


def plot_components(dates, data, trend, seasonal, residual):
    plt.figure(figsize=(14, 8))

    # Original Data
    plt.subplot(411)
    plt.plot(dates, data, label='Original')
    plt.legend(loc='upper left')
    plt.title('Original Data')

    # Trend
    plt.subplot(412)
    plt.plot(dates, trend, label='Trend')
    plt.legend(loc='upper left')
    plt.title('Trend')

    # Seasonal
    plt.subplot(413)
    plt.plot(dates, seasonal, label='Seasonal')
    plt.legend(loc='upper left')
    plt.title('Seasonal')

    # Residual
    plt.subplot(414)
    plt.plot(dates, residual, label='Residual')
    plt.legend(loc='upper left')
    plt.title('Residual')

    plt.tight_layout()
    plt.show()


def figure3(autocorrelations,max_lags):
    lags = np.arange(max_lags + 1)
    plt.figure(figsize=(10, 5))
    plt.stem(lags, autocorrelations[:max_lags + 1], use_line_collection=True)
    plt.title('Autocorrelation Function')
    plt.xlabel('Lags')
    plt.ylabel('Autocorrelation')
    plt.ylim(-1, 1)
    plt.show()


def main():

    # ===========================DOWNLOADING THE DATA===========================

    # TR timezone is UTC+3 since 2016
    trtz=dt.timezone(dt.timedelta(hours=3))
    # 1 April 2019
    start_date=dt.datetime(2019,4,1,tzinfo=trtz)
    # 1 April 2024
    end_date=dt.datetime(2024,4,1,tzinfo=trtz)
    # Using the EPIAS API to fetch the Turkish natural gas transmission dataset
    df=fetch_epias_exit_nomination(start_date,end_date)

    print(df.head())
    print(f"Dataset size: {len(df)}")

    # ===========================DATASET SPLIT===========================

    # Train-test split is from 1 April 2023 onwards
    thres_date=dt.datetime(2023,4,1,tzinfo=trtz)

    train_df=df[df['date']<thres_date]
    test_df=df[df['date']>thres_date]

    print(f"Training set size: {len(train_df)}")
    print(f"Testing set size: {len(test_df)}")

    # Plot the train-test split
    figure1(train_df,test_df)

    # Crossvalidation folds
    fold_length=len(train_df)//(NUM_FOLDS+1)
    folds=[(train_df[0:(i+1)*fold_length],train_df[(i+1)*fold_length:(i+2)*fold_length]) for i in range(0,NUM_FOLDS)]

    # Plot the 3rd crossvalidation fold
    figure1(*folds[2])

    # ===========================DATA DECOMPOSITION===========================

    ys = df['exitNominationAmount'].to_numpy()
    yearly_trend, yearly_seasonal, yearly_residual = decompose_timeseries(ys, 365)
    plot_components(df['date'], ys, yearly_trend, yearly_seasonal, yearly_residual)

    monthly_trend, monthly_seasonal, monthly_residual = decompose_timeseries(ys, 31)
    plot_components(df['date'], ys, monthly_trend, monthly_seasonal, monthly_residual)

    # ===========================DATA AUTOCORRELATION===========================

    autocorrelation_values = autocorr(ys,max_lags=365)
    figure3(autocorrelation_values,max_lags=365)

    # =============================GP REGRESSION=============================

    # Initial GP hyperparameters before tuning
    variance=1e-2
    a_year,a_quarter,a_month,a_week=(1e-3,5e-2,1e-1,0.2)
    b=0.5

    gp_hyperparams=np.array([variance,a_year,a_quarter,a_month,a_week,b])

    # Sum of kernels for yearly, quarterly, monthly, weekly periodicities + trend + residual
    kernel=lambda x_1,x_2: a_year*periodic_kernel(x_1,x_2,sigma=variance,timescale=90,period=365)+a_quarter*periodic_kernel(x_1,x_2,sigma=variance,timescale=30,period=90)+a_month*periodic_kernel(x_1,x_2,sigma=variance,timescale=7,period=30)+a_week*periodic_kernel(x_1,x_2,sigma=variance,timescale=3,period=7)+b*exp_quadratic_kernel(x_1,x_2,sigma=variance,timescale=90)+white_noise_kernel(x_1,x_2,sigma=variance)
    model=GaussianProcess(kernel)

    # Treating each day as an integer value. We can do this because of the equally distanced daily measurements in the dataset.
    xs=train_df.index.to_numpy()
    ys=train_df['exitNominationAmount'].to_numpy()

    # model.update_joint(xs,ys)

    # print(test_df.index.head())

    # model.infer()

if __name__ == "__main__":
    main()

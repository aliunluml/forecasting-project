import requests
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from gp import *
from arima import *
from lib import *

NUM_FOLDS=5



def preprocessing(ys):
    standardized=(ys-np.mean(ys))/np.std(ys)
    return standardized

def destandardize(ys, mean, std):
    result=ys*std+mean
    return result


def ridge_autoregression(model,ys,a):

    starting_i=model.order-1
    targets=ys[starting_i:]

    X=np.lib.stride_tricks.sliding_window_view(ys,window_shape=model.order)
    # preds dim is less than ys dim.

    new_params=np.linalg.inv(X.T@X+a*np.eye(model.order))@X.T@targets
    model.params=new_params
    return model


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
    # figure1(train_df,test_df)

    # Crossvalidation folds
    fold_length=len(train_df)//(NUM_FOLDS+1)
    folds=[(train_df[0:(i+1)*fold_length],train_df[(i+1)*fold_length:(i+2)*fold_length]) for i in range(0,NUM_FOLDS)]

    # Plot the 3rd crossvalidation fold
    # figure1(*folds[2])

    # ===========================DATA DECOMPOSITION===========================

    ys = df['exitNominationAmount'].to_numpy()
    yearly_trend, yearly_seasonal, yearly_residual = decompose_timeseries(ys, 365)
    # plot_components(df['date'], ys, yearly_trend, yearly_seasonal, yearly_residual)

    monthly_trend, monthly_seasonal, monthly_residual = decompose_timeseries(ys, 31)
    # plot_components(df['date'], ys, monthly_trend, monthly_seasonal, monthly_residual)

    # ===========================DATA AUTOCORRELATION===========================

    autocorrelation_values = autocorr(ys,max_lags=365)
    # figure3(autocorrelation_values,max_lags=365)

    # # =============================GP REGRESSION=============================

    # # Initial GP hyperparameters before tuning
    # variance=1e4
    # a_year,a_quarter,a_month,a_week=(1e-3,5e-2,1e-1,0.2)
    # b=0.5
    #
    # gp_hyperparams=np.array([variance,a_year,a_quarter,a_month,a_week,b])
    #
    # # Sum of kernels for yearly, quarterly, monthly, weekly periodicities + trend + residual
    # kernel=lambda x_1,x_2: a_year*periodic_kernel(x_1,x_2,sigma=variance,timescale=90,period=365)+a_quarter*periodic_kernel(x_1,x_2,sigma=variance,timescale=30,period=90)+a_month*periodic_kernel(x_1,x_2,sigma=variance,timescale=7,period=30)+a_week*periodic_kernel(x_1,x_2,sigma=variance,timescale=3,period=7)+b*exp_quadratic_kernel(x_1,x_2,sigma=variance,timescale=90)+white_noise_kernel(x_1,x_2,sigma=variance)
    # model=GaussianProcess(kernel)
    #
    # # Treating each day as an integer value. We can do this because of the equally distanced daily measurements in the dataset.
    # xs=train_df.index.to_numpy()
    # ys=train_df['exitNominationAmount'].to_numpy()
    #
    # model.update_joint(xs,ys)
    # # Respective pandas indices from the original df rows.
    # xs_star=test_df.index.to_numpy()
    # # print(xs[:10])
    # # print(ys[:10])
    # # print(xs_star[:3])
    #
    # # Looked at how they do the visualization in the following. Otherwise, implemented from maths.
    # # https://github.com/peterroelants/peterroelants.github.io/blob/main/notebooks/gaussian_process/gaussian-process-tutorial.ipynb
    # predictive=model.infer(xs_star)
    # # fs_star=predictive.sample(6)
    # # fs=model.joint_dist.sample(6)
    # # for i in range(6):
    # #     f=fs[i,:]
    # #     f_star=fs_star[i,:]
    # #     plt.plot(train_df['date'],f)
    # #     plt.plot(test_df['date'],f_star)
    #
    # print("MASE")
    # print(mean_absolute_scaled_err(test_df['exitNominationAmount'].to_numpy(),predictive.mean))
    #
    # plt.plot(df['date'],df['exitNominationAmount'],linestyle='dashed',color='k',zorder=1)
    # plt.plot(test_df['date'],predictive.mean,color='r',zorder=2)
    # plt.plot(train_df['date'], model.joint_dist.mean,color='b',zorder=3)
    # plt.legend(["actual","prediction", "train"], loc="lower right")
    #
    # pred_std_dev=np.sqrt(np.diag(predictive.covariance))
    # plt.fill_between(test_df['date'], predictive.mean-2*pred_std_dev, predictive.mean+2*pred_std_dev, color='red',alpha=0.15, label='$2 \sigma_{2|1}$')
    #
    # joint_std_dev=np.sqrt(np.diag(model.joint_dist.covariance))
    # plt.fill_between(train_df['date'], model.joint_dist.mean-2*joint_std_dev, model.joint_dist.mean+2*joint_std_dev, color='red',alpha=0.15, label='$2 \sigma_{2|1}$')
    #
    # plt.title('GP Regression')
    # plt.xlabel('Days')
    # # Gas is measured in standard cubic meters.
    # plt.ylabel('Transmission Output Volume ($Sm^3$)')
    #
    # # plt.show()
    # plt.savefig("gp.png")
    #

    # =============================LINEAR AUTOREGRESSION=============================

    k0=np.zeros(len(df))
    k1=np.zeros(len(df))
    k2=np.zeros(len(df))
    k3=np.zeros(len(df))
    scales=[3,7,30,90]

    for i in range(4):

        train_xs=train_df.index.to_numpy()
        train_ys=train_df['exitNominationAmount'].to_numpy()
        standard_train_ys=preprocessing(train_ys)

        order=scales[i]
        monthly_ar=NonstationaryAR(order)
        a=1e-1
        monthly_ar=ridge_autoregression(monthly_ar,standard_train_ys,a)
        preds=monthly_ar.infer(standard_train_ys)
        train_ys_hat=destandardize(preds,np.mean(train_ys),np.std(train_ys))

        plt.plot(train_df['date'][order-1:],train_ys_hat,color='b')
        plt.plot(train_df['date'],train_ys,color='k')

        test_xs=test_df.index.to_numpy()
        test_ys=test_df['exitNominationAmount'].to_numpy()
        standard_test_ys=preprocessing(test_ys)
        test_ys_hat=destandardize(monthly_ar.infer(standard_test_ys),np.mean(train_ys),np.std(train_ys))


        plt.plot(test_df['date'][order-1:],test_ys_hat,color='r')
        plt.plot(test_df['date'],test_ys,color='k')

        plt.title(f"Linear Autoregression (AR({order}))")
        plt.xlabel('Days')
        # # Gas is measured in standard cubic meters.
        plt.ylabel('Transmission Output Volume ($Sm^3$)')
        plt.savefig("ar.png")
        plt.show()

        print("MASE")
        print(mean_absolute_scaled_err(test_df['exitNominationAmount'].to_numpy()[order-1:],test_ys_hat))

        exec(f"np.put(k{i},test_xs[order-1:],test_ys_hat)")
        exec(f"np.put(k{i},train_xs[order-1:],train_ys_hat)")

    ensemble_preds=(k0+k1+k2+k3)/4

    plt.plot(train_df['date'],ensemble_preds[:1460],color='b')
    plt.plot(test_df['date'],ensemble_preds[1460:],color='r')
    plt.plot(df['date'],df['exitNominationAmount'],color='k')

    plt.title(f"Ensemble Linear Autoregression of ARs")
    plt.xlabel('Days')
    # # Gas is measured in standard cubic meters.
    plt.ylabel('Transmission Output Volume ($Sm^3$)')
    plt.savefig("ar.png")
    plt.show()

    print("MASE")
    print(mean_absolute_scaled_err(test_df['exitNominationAmount'].to_numpy(),ensemble_preds[1460:]))


if __name__ == "__main__":
    main()

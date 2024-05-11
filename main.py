import requests
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from gp import *
from lib import *

NUM_FOLDS=5


# TR timezone is UTC+3 since 2016
trtz=dt.timezone(dt.timedelta(hours=3))

# 1 April 2019
start_date=dt.datetime(2019,4,1,tzinfo=trtz)
# 1 April 2024
end_date=dt.datetime(2024,4,1,tzinfo=trtz)

# EPIAS API expects dates to be in the ISO 8601 format.
# JSON has a different syntax than Python. It needs double quotes, not single quotes!
payload={
  "endDate":end_date.isoformat(),
  "startDate":start_date.isoformat(),
  "page": {
    "sort": {
      "direction": "ASC",
      "field": "date"
    }
  }
}

# HTTP POST request with the data
# Disconnect from the current Colab instance and connect to another one if you get HTTPConnection erros.
response = requests.post('https://seffaflik.epias.com.tr/natural-gas-service/v1/transmission/data/exit-nomination',json=payload)

if response.status_code!=200:
    exit
else:
    pass


data=response.json()
df=pd.DataFrame.from_dict(data['items'])
# Revert back from the ISO 8601 format to the datetime objects
df['date']=df['date'].map(lambda x: dt.datetime.fromisoformat(x))

print(df.head())
print(f"Dataset size: {len(df)}")


# Train-test split is from 1 April 2023 onwards
thres_date=dt.datetime(2023,4,1,tzinfo=trtz)

train_df=df[df['date']<thres_date]
test_df=df[df['date']>thres_date]

print(f"Training set size: {len(train_df)}")
print(f"Testing set size: {len(test_df)}")

fig1,ax1=plt.subplots()
# We unpack a list containing a single line object
line11,=ax1.plot(train_df['date'],train_df['exitNominationAmount'],color='b',label='Train')
line12,=ax1.plot(test_df['date'],test_df['exitNominationAmount'],color='r',label='Test')
ax1.legend(handles=[line11, line12])
ax1.set_title('Natural Gas Time-series')
ax1.set_xlabel('Days')
# Gas is measured in standard cubic meters.
ax1.set_ylabel('Transmission Output Volume ($Sm^3$)')
plt.show()


# Because we are planning to compare the performances of different ML algorithms, we have a test set that is not used until the very end.
# We are also using cross-validation because we want to monitor the performance of each individual algorithm and ensure convergence before the final benchmarking.

# Because our training set length is 1460, We are OK.
# If it were otherwise, there might have been training data leftovers.

fold_length=len(train_df)//(NUM_FOLDS+1)
folds=[(train_df[0:(i+1)*fold_length],train_df[(i+1)*fold_length:(i+2)*fold_length]) for i in range(0,NUM_FOLDS)]

# Get the 2nd fold
train_split,val_split=folds[4]

# assert len(train_split)+len(val_split)==len(train_df)

fig2,ax2=plt.subplots()
# We unpack a list containing a single line object
line21,=ax2.plot(train_split['date'],train_split['exitNominationAmount'],color='b',label='Train')
line22,=ax2.plot(val_split['date'],val_split['exitNominationAmount'],color='c',label='Validation')
line23,=ax2.plot(test_df['date'],test_df['exitNominationAmount'],color='r',label='Test')
ax2.legend(handles=[line21, line22,line23])
ax2.set_title('Natural Gas Time-series')
ax2.set_xlabel('Days')
# Gas is measured in standard cubic meters.
ax2.set_ylabel('Transmission Output Volume ($Sm^3$)')
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

data_series = df['exitNominationAmount']
yearly_trend, yearly_seasonal, yearly_residual = decompose_timeseries(data_series.values, 365)
plot_components(df['date'], data_series, yearly_trend, yearly_seasonal, yearly_residual)

monthly_trend, monthly_seasonal, monthly_residual = decompose_timeseries(data_series.values, 31)
plot_components(df['date'], data_series, monthly_trend, monthly_seasonal, monthly_residual)


max_lags=365
data_series = df['exitNominationAmount'].values
autocorrelation_values = autocorr(data_series,max_lags)

lags = np.arange(max_lags + 1)
plt.figure(figsize=(10, 5))
plt.stem(lags, autocorrelation_values[:max_lags + 1], use_line_collection=True)
plt.title('Autocorrelation Function')
plt.xlabel('Lags')
plt.ylabel('Autocorrelation')
plt.ylim(-1, 1)
plt.show()

# autocorr(data_series)

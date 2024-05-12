import numpy as np
import pandas as pd
import datetime as dt
import requests


def mean_squared_err(targets,preds):
    mse=np.mean((targets-preds)**2)
    return mse

# MASE
def mean_absolute_scaled_err(targets,preds):
    naive_preds_mae=np.abs(targets[1:]-targets[:-1]).mean()
    errors=(targets-preds)
    scaled_errors=errors/naive_preds_mae
    result=np.abs(scaled_errors).mean()
    return result


def fetch_epias_exit_nomination(start,end):
    # datetime arguments should have TR time tzinfo
    assert (start.tzinfo is not None) and (end.tzinfo is not None)

    # EPIAS API expects dates to be in the ISO 8601 format.
    # JSON has a different syntax than Python. It needs double quotes, not single quotes!
    payload={
      "endDate":end.isoformat(),
      "startDate":start.isoformat(),
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
        raise requests.RequestException()
    else:
        pass

    data=response.json()
    df=pd.DataFrame.from_dict(data['items'])
    # Revert back from the ISO 8601 format to the datetime objects
    df['date']=df['date'].map(lambda x: dt.datetime.fromisoformat(x))

    return df

# https://waterprogramming.wordpress.com/2018/09/04/implementation-of-the-moving-average-filter-using-convolution/
def moving_average(data, window_size):
    window_size = min(window_size, len(data))
    # boxcar smoothing via convolution with a rectangle of width 2M+1
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


# https://otexts.com/fpp3/classical-decomposition.html
def decompose_timeseries(data, period):
    n = len(data)

    trend_window = period
    trend = moving_average(data, trend_window)
    trend = np.pad(trend, (trend_window // 2, n - len(trend) - trend_window // 2), mode='edge')

    detrended = data - trend
    seasonal = np.array([np.mean(detrended[i::period]) for i in range(period)])
    seasonal = np.tile(seasonal, n // period + 1)[:n]

    residual = data - trend - seasonal

    return trend, seasonal, residual


#ACF
def autocorr(x, max_lags=365):
    n = len(x)
    mean_x = np.mean(x)
    var_x = np.var(x)
    x = x - mean_x
    autocorr_full = np.correlate(x, x, mode='full')[-n:]
    autocorr = autocorr_full/(var_x*np.arange(n, 0, -1))

    return autocorr

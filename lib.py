import numpy as np

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

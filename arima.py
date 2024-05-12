import numpy as np

# Normally, for the AR model parameters of high orders, there are constraints on model parameters. https://otexts.com/fpp3/AR.html
# By placing NO constraints on the parameters of the AR model, we allow the violation of its Weak-Sense-Stationarity.
class NonstationaryAR():
    def __init__(self,order, params=None):
        super(NonstationaryAR,self).__init__()
        self.order=order
        self.params=np.random.randn(order) if params is None else params

    def infer_next(self,window):
        next=self.params@window+np.random.randn(len(window))
        return next

    def infer(self,data):
        # Assuming dataset length is greater than the order
        preds=np.convolve(self.params,data,mode='valid')+np.random.randn(len(data)-self.order+1)
        return preds


# class MA():
#     def __init__(self):
#         super(MA,self).__init__()
#
#
#
# class ARIMA():
#     def __init__(self):
#         super(ARIMA,self).__init__()

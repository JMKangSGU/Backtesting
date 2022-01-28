import numpy as np
import pandas as pd
from typing import Union
import statsmodels.api as sm
from scipy.stats import t as T

class Ornstein_Uhlenbeck:

    def __init__(self, series: Union[pd.Series, np.array],
                 period: Union[str, int, float],
                 method: str='least_square'):

        if type(series) == pd.Series:
            self.time_series = np.array(series)
        else: self.time_series = series
        self.mu = None
        self.sigma = None
        self.theta = None
        self.t, self.i = None, None
        self.confidence_interval_ = None
        self.method = method

        if type(period) == str:
            if period == 'daily':
                self.dt = 1/252
            elif period == 'monthly':
                self.dt = 1/12
            elif period == 'year':
                self.dt = 1
        elif type(period) == int or float:
            self.dt = period
        else:
            raise KeyError('daily, monthly, year, int or float')


    def fit(self):

        if self.method == 'least_square':
            X = self.time_series[1:]
            X = sm.add_constant(X)
            Y = self.time_series[:-1].reshape(-1, 1)
            model = sm.OLS(Y, X)
            res = model.fit()
            alpha, phi = res.params[0], res.params[1]
            epsilon = res.mse_resid
            self.theta = -np.log(phi)/self.dt
            self.mu = alpha/(1-phi)
            self.sigma = np.sqrt(epsilon/self.dt)

        elif self.method == 'max_likelihood':
            None

    def predict(self, t: int, i: int=10000, cl: float=.95
                ):

        self.t, self.i = t, i
        S = self.time_series[-1]*np.exp(-self.theta*t*self.dt) + \
            self.mu*(1 - np.exp(-self.theta*t*self.dt)) + \
            self.sigma*np.sqrt((1 - np.exp(-2*self.theta*t*self.dt))/(2*self.theta))*np.random.normal(loc=.0, scale=1.0, size=i)
        mean = np.mean(S)
        std = np.std(S)
        dof = len(S)-1
        self.confidence_interval_ = T.interval(cl, dof, loc=mean, scale=std)
        return S

if __name__ == '__main__':
    import yfinance as yf
    import matplotlib.pyplot as plt
    from OU_Process import Ornstein_Uhlenbeck

    df = yf.download('^GSPC', '2020-01-01', '2022-01-01').Close
    model = Ornstein_Uhlenbeck(df, period='daily')
    model.fit()
    pred = model.predict(20)
    print(model.confidence_interval_)
    plt.hist(pred)
    plt.show()
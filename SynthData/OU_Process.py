import numpy as np
import pandas as pd
from typing import Union
import statsmodels.api as sm
from scipy.stats import t as T
from scipy.optimize import minimize

class Ornstein_Uhlenbeck:

    def __init__(
            self,
            series: Union[pd.Series, np.array],
            period: Union[str, int, float],
            method: str='least_square'):

        if type(series) == pd.Series:
            self.time_series = np.array(series)
        elif type(series) == np.array:
            self.time_series = series
        else: raise TypeError('pd.Series, np.array or None(for simulation)')

        self.mu = None
        self.sigma = None
        self.theta = None
        self.t, self.i = None, None
        self.method = method
        self.confidence_interval_ = None
        self.half_life_ = None

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
            raise TypeError('daily, monthly, year, int or float')


    def fit(self):
        Xtm1 = self.time_series[:-1].reshape(-1, 1)
        Xt = self.time_series[1:]

        if self.method == 'least_square':
            Xtm1 = sm.add_constant(Xtm1)
            model = sm.OLS(Xt, Xtm1)
            res = model.fit()
            alpha, phi = res.params[0], res.params[1]
            epsilon = res.mse_resid
            self.theta = -np.log(phi)/self.dt
            self.mu = alpha/(1-phi)
            self.sigma = np.sqrt(epsilon/self.dt)
            self.half_life_ = np.log(2)/self.theta

        elif self.method == 'max_likelihood':
            x_init = np.array([.5, 1])
            res = minimize(fun=self.mleObject,
                           x0=x_init,
                           args=[Xt, Xtm1, self.dt],
                           method='L-BFGS-B',
                           tol=1e-100,
                           options={'disp': False})
            params = res.x
            self.theta = params[0]
            self.sigma = params[1]
            self.mu    = 0
            self.half_life_ = np.log(2) / self.theta


    def predict(
            self,
            t: int,
            i: int=10000,
            cl: float=.95
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

    @staticmethod
    def mleObject(params, args):
        Xt, Xtm1, dt = args[0], args[1], args[2]
        theta, sigma = params[0], params[1]
        N = Xt.shape[0]
        func = (-N * np.log(2 * N) / 2) \
               - (N * np.log(sigma)) \
               - (np.sum(np.square(Xt - (Xtm1 * np.exp(-theta * dt))))) / (2 * sigma ** 2)
        return func

    @staticmethod
    def simulation(mu, sigma, theta, n, dt):
        X = np.zeros(n)
        for i in range(1, n):
            X[i] = X[i-1] + theta*(mu - X[i-1])*dt + sigma*np.random.normal(loc=0.0, scale=1.0)
        return X

if __name__ == '__main__':
    import FinanceDataReader as fdr
    import matplotlib.pyplot as plt


    df = fdr.DataReader('KO', '2017-01-01', '2022-01-01').Close
    model = Ornstein_Uhlenbeck(df, period='daily')
    model.fit()
    paths = []
    for i in range(20):
        paths.append(model.predict(t=i))
    paths = np.array(paths)
    plt.plot(paths[:, :20])
    plt.show()
    pred = model.predict(20)
    plt.hist(pred)
    plt.show()
    print(model.confidence_interval_)
    print(model.half_life_)
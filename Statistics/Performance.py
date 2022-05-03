import numpy as np
import pandas as pd
from typing import Union
from scipy.stats import skew, kurtosis, norm
from utils.types import toNp

def getSR(PnL: Union[pd.DataFrame, pd.Series, np.array]):
    if   PnL.shape[1] == 1: axis=0
    else                  : axis=1
    SR = PnL.mean(axis=axis) / PnL.std(axis=axis)
    return SR

def getPSR(ret: np.array, bmrk: Union[str, int, float]):
    """
    Probabilistic Sharpe Ratio (Lopez de Prado 2018b) computation method
    :param ret: N trials of Pnl (N>=1)
    :param bmrk: The benchmark sharpe ratio
    :return: Probabilistic sharpe ratio of each or single trial
    """
    rs, idx, col = toNp(ret)
    srStar       = float(bmrk)
    sr, T        = getSR(rs), len(rs)
    Eskew, Ekurt = skew(rs) , kurtosis(rs)

    x   = ((sr - srStar) * (np.sqrt(T-1))) / \
          np.sqrt(1 - Eskew * sr + (Ekurt - 1) / 4 * (sr ** 2))
    psr = norm.cdf(x)
    return psr

def getDSR(ret: np.array):
    """
    Deflated Sharpe Ratio (Lopez de Prado 2018b) computation method
    :param ret: N trials of PnL (N>1)
    :return: Deflated sharpe ratio for each trial
    """
    rs, idx, col = toNp(ret)
    n, std, eM   = rs.shape[0], rs.std(axis=1), np.euler_gamma

    srStar = std * ((1 - eM) * norm.ppf(1 - 1/n) + eM * norm.ppf(1 - 1 / (n * np.e)))

    DSR    = []
    for i in range(n): DSR.append(getPSR(rs[i]), srStar)

    DSR    = pd.DataFrame(DSR, index=['Deflated SR'], columns=col)
    return DSR


if __name__ == "__main__":
    import os

    if os.path.exists("D:/BTtest.csv"):
        df = pd.read_csv("D:/BTtest.csv", index_col=0, parse_dates=True)
    else:
        import yfinance as yf
        df = yf.download(['AAPL', 'AMD', 'MSFT'], start='2018-01-01')['Adj Close'].pct_change().dropna()
        df.to_csv("D:/BTtest.csv")
        npdf = df.to_numpy().T





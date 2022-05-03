import numpy as np
import pandas as pd
import matplotlib.pyplot as mpl
import seaborn as sns
from typing import Union
from itertools import combinations
from Performance import getSR, getPSR, getDSR
from scipy import stats
from tqdm import tqdm

sns.set()

def getPBO(partitioned_ret: Union[np.array, pd.DataFrame], num_splits: Union[int, float, str], type: str, plot: bool, ):
    """
    Derivation of Probability of backtest overfitting [Bailey et al.; 2017a]
    :param partitioned_ret: Array of partitioned backtest paths. Required dimensions: (P, N, T)
                            where i) P is the number of partitions.
                                  ii) N is the number of backtest trials.
                                  iii) T is the number of rows per partition.
    :param num_splits: Number of splits for the performance sequence. Used only when given by DataFrame
    :parm type : Type of Sharpe ratio to consider as performance metric.
                 Either i) Standard : Standard Sharpe Ratio
                        ii) PSR     : Probabilistic Sharpe Ratio
                        iii) DSR    : Deflated Sharpe Ratio
    :param plot: When True, two plots are rendered
                 i) Best Sharpe ratio in-sample vs Sharpe ratio out-of-sample
                 ii) Histogram of rank logits
    :return: Probability of Overfitting
    """
    if isinstance(partitioned_ret, pd.DataFrame):
        pr = np.array_split(partitioned_ret.to_numpy(), int(num_splits))
        frame = []
        for i in pr: frame.append(i.T)
        if frame[0].shape[1] != frame[1].shape[1]:
            frame[0] = frame[0].T[frame[0].shape[1] - frame[1].shape[1]:].T
        pr = frame
        N, S = partitioned_ret.shape[1], partitioned_ret.shape[0]

    if isinstance(partitioned_ret, np.array):
        pr, N, S     = partitioned_ret, partitioned_ret.shape[1], partitioned_ret.shape[0]

    c , s     = [i for i in range(S)], int(S / 2)

    w, lambdas = [], []
    isSR, osSR = [], []
    for com in tqdm(combinations(c, s)):
        anti_com = [i for i, j in enumerate(np.isin(c, com)) if not j]

        train  , test   = np.concatenate(pr[list(com)], axis=1), np.concatenate(pr[anti_com], axis=1)

        # 차후 하나의 getSR로 수정예정
        if type == 'Standard':
            trainSR, testSR = getSR(train), getSR(test)
        elif type == 'PSR':
            trainSR, testSR = getPSR(train), getPSR(test)
        elif type == 'DSR':
            trainSR, testSR = getDSR(train), getDSR(test)

        maxidx          = np.where(trainSR == trainSR.max())[0][0]

        rank    = np.where(testSR.argsort()[::-1] == maxidx)[0][0]
        relrank = (np.array(range(N)) + 1) / (N + 1)
        relrank = relrank[rank]
        lambda_ = np.log(relrank / (1 - relrank))

        isSR.append(trainSR[maxidx]); osSR.append(testSR[maxidx])
        w.append(relrank); lambdas.append(lambda_)

    lambdas    = np.array(lambdas)
    PBO        = len(lambdas[lambdas <= 0]) / len(lambdas)
    isSR, osSR = np.array(isSR).reshape(-1), np.array(osSR).reshape(-1)

    if plot:
        a, b = np.polyfit(isSR, osSR, 1)

        f, ax   = mpl.subplots(2, 1, figsize=(10, 20))
        n, x, _ = ax[0].hist(lambdas, bins=50, density=True)
        density = stats.gaussian_kde(lambdas)
        ax[0].plot(x, density(x), linestyle='-')
        ax[0].set_title("Hist. of Rank Logits")
        ax[0].set_ylabel("Frequency"); ax[0].set_xlabel("Logits")

        ax[1].scatter(isSR, osSR)
        ax[1].set_title("OOS Perf. Degradation")
        ax[1].set_ylabel("OOS SR"); ax[1].set_xlabel("IS SR")
        ax[1].plot(isSR, a*isSR + b, color='crimson')

        mpl.show()

    return PBO

if __name__ == '__main__':
    dt = np.random.randn(15, 100, 100)
    PBO = getPBO(dt)
    print("PBO : ", PBO)

import numpy as np
import pandas as pd
from typing import Union

def resampleRet():
    pass

def toNp(ret: Union[np.array, pd.Series, pd.DataFrame]):
    if isinstance(ret, np.array)    : pass
    if isinstance(ret, pd.Series)   : idx, col = ret.index, None
    if isinstance(ret, pd.DataFrame): idx, col = ret.index, ret.columns
    ret_ = ret.to_numpy().T
    return ret_, idx, col


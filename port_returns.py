# returns.py
import pandas as pd
import numpy as np

def portfolio_returns(
    returns_df: pd.DataFrame, 
    w: np.ndarray
) -> pd.Series:
    return returns_df.dot(w)
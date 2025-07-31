# metrics.py
import numpy as np
import pandas as pd
from typing import Optional
def compute_zscore(
    portfolio_returns: pd.Series,
    portfolio_vol_pred: pd.Series,
    horizon: int=1
) -> pd.Series:
    """
    Compute z-score for a given horizon:
    z_t = realized cumulative return over [t, t+horizon-1] / predicted volatility at t

    Args:
        portfolio_returns (pd.Series): daily portfolio returns indexed by date.
        portfolio_vol_pred (pd.Series): predicted volatilities indexed by date.
        horizon (int): holding period (days).

    Returns:
        pd.Series: z-score indexed by date corresponding to the start of the holding period.
    """
    cum_returns = portfolio_returns.rolling(window=horizon).sum().shift(-(horizon-1))
    
    # the beginning-of-horizon volatility forecast
    vol_shifted = portfolio_vol_pred

    # Align dates
    common_dates = cum_returns.index.intersection(vol_shifted.index)
    returns_aligned = cum_returns.loc[common_dates]
    vol_aligned = vol_shifted.loc[common_dates]

    # Compute z-score
    zscore = returns_aligned / vol_aligned

    # Clean z-score
    zscore = zscore.replace([np.inf, -np.inf], np.nan).dropna()

    return zscore

def compute_bias_stat(
    zscore: pd.Series,
    ddof: int=0
) -> float:
    return float(zscore.std(ddof=ddof))

def compute_q_stat(
    zscore: pd.Series
) -> float:
    squared = zscore.pow(2)
    eps = np.finfo(float).eps
    Q_t = squared - np.log(squared + eps)
    Q_stat = Q_t.mean()
    return float(Q_stat)
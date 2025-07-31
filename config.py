# config.py
import numpy as np
import pandas as pd
from typing import List, Dict, Callable

# Data paths
RETURNS_CSV = "synthetic_data/data/returns.csv"
OMEGA_DIRS = {
    "modelA": "synthetic_data/models/modelA/omega",
    "modelB": "synthetic_data/models/modelB/omega",
}

# Prediction horizons
HORIZONS = [1, 5, 21, 63]

# Weights
def equal_weights(
    n: int
) -> np.ndarray:
    """ Equal-weighted portfolio """
    return np.ones(n) / n

def inverse_hist_vol_weights(
    returns_df: pd.DataFrame,
    window: int = 21
) -> np.ndarray:
    """
    Compute inverse-hist-vol weights on a rolling basis.

    Returns a DataFrame of weights indexed by date,
    where each row t uses returns_df.loc[t-window:t-1].
    """
    # pre-allocate
    dates = returns_df.index
    wts = pd.DataFrame(index=dates, columns=returns_df.columns)

    for t in range(window, len(dates)):
        hist_slice = returns_df.iloc[t-window:t]
        vols = hist_slice.std(ddof=1)
        inv_vol = 1 / vols
        wts.iloc[t] = inv_vol / inv_vol.sum()

    return wts.dropna()

LOOKBACK_WINDOW = 21
# def value_weights(market_caps: np.ndarray) -> np.ndarray: ...
# def minvar_weights(cov_matrix: np.ndarray) -> np.ndarray: ...

WEIGHTS: Dict[str, Callable[..., np.ndarray]] = {
    "equal": equal_weights,
    "inv_hist_vol": inverse_hist_vol_weights,
    # "value": get_value_weights, "minvar": get_minvar_weights, ...
}
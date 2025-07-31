# vol_pred.py
import os
import numpy as np
import pandas as pd
from data_utils import list_omega_dates, load_omega_matrix
from typing import List

# portfolio predicted volatility
def portfolio_vol_pred(
    omega_dir: str,
    assets: List[str],
    w: np.ndarray,
    horizon: int=1
) -> pd.Series:
    """
    Compute predicted portfolio volatility for each date using
    cumulative horizon-day covariance matrices.

    Returns:
        pd.Series: Predicted portfolio volatilities indexed by date.
    """
     # Retrieve and sort available dates
    dates = list_omega_dates(omega_dir)
    vols = {}

    for i, dt in enumerate(dates):
        if i+horizon > len(dates):
            break
        omega_sum = None
        for j in range(horizon):
            date = dates[i+j]
            file_path = os.path.join(
                omega_dir,
                f"omega_{date.strftime('%Y%m%d')}.csv"
            )
            omega = load_omega_matrix(file_path, assets)
            omega_sum = omega if omega_sum is None else omega_sum + omega
        vols[dt] = np.sqrt(w.T @ omega_sum @ w)
    
    return pd.Series(vols).sort_index()
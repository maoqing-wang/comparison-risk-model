# config.py
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Callable

# ===============================
# Paths (change DATA_ROOT only)
# ===============================
# e.g. "synthetic_data" or "synthetic_pca_compare"
DATA_ROOT = Path("synthetic_ewma_compare")

RETURNS_CSV = DATA_ROOT / "data" / "returns.csv"

OMEGA_DIRS: Dict[str, str] = {
    "modelA": str(DATA_ROOT / "models" / "modelA" / "omega"),
    "modelB": str(DATA_ROOT / "models" / "modelB" / "omega"),
}

# ===============================
# Horizons & Lookback
# ===============================
HORIZONS = [1, 5, 21, 63]
# History window for weight calculation (e.g. inv-hist-vol)
LOOKBACK_WINDOW = 21

# ===============================
# Portfolio Weights (unified API)
# ===============================
def equal_weights(
    returns_df: pd.DataFrame,
    window: int | None = None
) -> np.ndarray:
    """
    Equal-weighted portfolio; ignore window.
    """
    n = returns_df.shape[1]
    return np.ones(n) / n


def inverse_hist_vol_weights(
    returns_df: pd.DataFrame,
    window: int = 21
) -> np.ndarray:
    """
    Inverse historical volatility weights (single point, using "last window days").
      w_i ∝ 1 / vol_i, where vol_i is the sample standard deviation of the asset's daily returns over the last window days.
    Returns: length-N weight vector, sum(w)=1.
    """
    # Last window days
    if len(returns_df) < window:
        raise ValueError(f"Not enough rows for window={window}: got {len(returns_df)}")
    hist = returns_df.tail(window)

    # Sample standard deviation (avoid 0 volatility)
    vols = hist.std(ddof=1).replace(0.0, np.nan)
    vols = vols.fillna(vols.mean() if not np.isnan(vols.mean()) else 1.0)

    inv_vol = 1.0 / vols.values
    w = inv_vol / inv_vol.sum()
    return w


# unified entry: main loop iterates WEIGHTS, directly calls w_func(returns_df, LOOKBACK_WINDOW)
WEIGHTS: Dict[str, Callable[[pd.DataFrame, int], np.ndarray]] = {
    "equal": equal_weights,
}
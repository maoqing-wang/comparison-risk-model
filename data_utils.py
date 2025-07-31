# data_utils.py
import os
import pandas as pd
import numpy as np
from typing import List, Optional

def load_returns(
    path: str
) -> pd.DataFrame:
    """
    Load and sort the returns DataFrame.
    """
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.sort_index()

def list_omega_dates(
    omega_dir: str
) -> List[pd.Timestamp]:
    """
    List and sort dates for all files named 'omega_YYYYMMDD.csv' in the folder.
    """
    files = [f for f in os.listdir(omega_dir)
             if f.startswith('omega_') and f.endswith('.csv')]
    dates = [pd.to_datetime(f.split('_')[1].split('.')[0], format='%Y%m%d')
             for f in files]
    return sorted(dates)

def load_omega_matrix(
    file_path: str, 
    assets: List[str]
) -> np.ndarray:
    """
    Read one omega CSV (covariance matrix), reindex to given assets, 
    and return a NumPy array.
    """
    df = pd.read_csv(file_path, index_col=0)
    df = df.reindex(index=assets, columns=assets)
    return df.values
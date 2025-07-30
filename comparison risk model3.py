#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import statsmodels.api as sm

# Reload the cleaned return data
file_path = 'final_cleaned_financial_data.csv'
df = pd.read_csv(file_path)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df = df.apply(pd.to_numeric, errors='coerce')
df.dropna(axis=1, how='all', inplace=True)
returns = df.pct_change().dropna()

# Define proxy factor list based on paper approximation
proxy_factors = [
    'VB US EQUITY', 'IWM US EQUITY',              # Equity (Size)
    'LD12TRUU INDEX', 'LD19TRUU INDEX',           # Momentum/Volatility
    'GSG US EQUITY', 'SPGCCITR INDEX',            # Commodities/Value
    'VNQ US EQUITY', 'RMZ INDEX', 'RMSG INDEX',   # Real Estate
    'EMB US EQUITY', 'TIP US EQUITY',             # Bonds
    'EFA US EQUITY', 'VWO US EQUITY', 'NDUEEGF INDEX'  # Global Equity
]

# Split the dataset
factor_returns = returns[proxy_factors].dropna()
common_dates = factor_returns.index

# Only use assets that have data for the same dates as the factor returns
asset_returns = returns.loc[common_dates].drop(columns=proxy_factors, errors='ignore')
result_summary = []

# For each asset, regress on the factors
for asset in asset_returns.columns:
    y = asset_returns[asset]
    X = factor_returns.loc[y.index]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    # Variance decomposition
    y_hat = model.fittedvalues
    residuals = model.resid

    total_var = np.var(y)
    factor_var = np.var(y_hat)
    idio_var = np.var(residuals)
    r_squared = model.rsquared

    result_summary.append({
        'Asset': asset,
        'Total Variance': total_var,
        'Factor Variance': factor_var,
        'Idiosyncratic Variance': idio_var,
        'R-squared': r_squared
    })

# Create results dataframe
results_df = pd.DataFrame(result_summary)


# In[9]:


# Re-run the regression analysis after connection reset
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Reload the cleaned return data
file_path = 'final_cleaned_financial_data.csv'
df = pd.read_csv(file_path)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df = df.apply(pd.to_numeric, errors='coerce')
df.dropna(axis=1, how='all', inplace=True)
returns = df.pct_change().dropna()

# Define proxy factor list based on paper approximation
proxy_factors = [
    'VB US EQUITY', 'IWM US EQUITY',
    'LD12TRUU INDEX', 'LD19TRUU INDEX',
    'GSG US EQUITY', 'SPGCCITR INDEX',
    'VNQ US EQUITY', 'RMZ INDEX', 'RMSG INDEX',
    'EMB US EQUITY', 'TIP US EQUITY',
    'EFA US EQUITY', 'VWO US EQUITY', 'NDUEEGF INDEX'
]

# Isolate factor returns
factor_returns = returns[proxy_factors].dropna()
common_dates = factor_returns.index
asset_returns = returns.loc[common_dates].drop(columns=proxy_factors, errors='ignore')

# Perform regression for each asset
result_summary = []

for asset in asset_returns.columns:
    y = asset_returns[asset]
    X = factor_returns.loc[y.index]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    # Decompose variance
    y_hat = model.fittedvalues
    residuals = model.resid
    total_var = np.var(y)
    factor_var = np.var(y_hat)
    idio_var = np.var(residuals)
    r_squared = model.rsquared

    result_summary.append({
        'Asset': asset,
        'Total Variance': total_var,
        'Factor Variance': factor_var,
        'Idiosyncratic Variance': idio_var,
        'R-squared': r_squared
    })

# Convert to DataFrame and display
results_df = pd.DataFrame(result_summary)

# Display the decomposition results using pandas (no ace_tools)
results_df.head(20)


# In[ ]:





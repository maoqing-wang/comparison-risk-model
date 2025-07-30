#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA


# In[43]:


file_path = 'final_fully_cleaned_data.csv'
df = pd.read_csv(file_path,index_col=0, parse_dates=True)


df = df.sort_index(ascending=False)
df.head()


# In[44]:


# Drop columns with too many missing values and forward fill the rest
df = df.dropna(axis=1, thresh=int(0.9 * len(df)))
df = df.ffill().dropna()

# Calculate daily returns
returns = df.pct_change().dropna()


# In[45]:


# Model A
# Define proxy factors and target assets (MAC3) - Mimic the MAC3-style Model from Miya

factors = ['SPGCCITR INDEX', 'LUACTRUU INDEX']
assets = ['RMS G INDEX', 'LUMSTRUU INDEX']

factor_returns = returns[factors]
asset_returns = returns[assets]

mac3_results = []

# Run regression for each asset
for asset in asset_returns.columns:
    y = asset_returns[asset]
    X = sm.add_constant(factor_returns.loc[y.index])
    model = sm.OLS(y, X).fit()
    
    y_hat = model.fittedvalues
    residuals = model.resid

    mac3_results.append({
        'Asset': asset,
        'Total Variance': y.var(),
        'Factor Variance': y_hat.var(),
        'Idiosyncratic Variance': residuals.var(),
        'R-squared': model.rsquared
    })

# Create summary DataFrame
mac3_df = pd.DataFrame(mac3_results)
print(mac3_df)


# In[46]:


'''
Model A: MAC3-style regression
- Asset RMS G INDEX, R-squared = 0.0018 → Regression model explains almost nothing — likely due to poor factor match or noisy asset
- Asset LUMSTRUU INDEX, R-squared = 0.6253 → Regression explains ~63% of return variance — reasonably strong factor fit
'''


# In[53]:


# Model B
# Ledoit-Wolf Shrinkage + PCA

import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.covariance import LedoitWolf


# Compute log returns
returns = np.log(df / df.shift(1)).dropna()

# Select the same 4 assets for ICA
assets_ica = ['SPGCCITR INDEX', 'LUACTRUU INDEX', 'RMS G INDEX', 'LUMSTRUU INDEX']
asset_returns = returns[assets_ica]

# Standardize the returns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(asset_returns.values)

# Apply FastICA for 2 independent components
ica = FastICA(n_components=2, random_state=42, max_iter=1000)
S = ica.fit_transform(X_scaled)          # Independent components
A = ica.mixing_                         # Mixing matrix
reconstructed = np.dot(S, A.T)          # Reconstructed signals

# Compute residuals
residuals_ica = X_scaled - reconstructed

# Build results DataFrame
ica_results = []
for i, asset in enumerate(assets_ica):
    realized = X_scaled[:, i]
    factor = reconstructed[:, i]
    idio = residuals_ica[:, i]
    
    corr_factor = np.corrcoef(realized, factor)[0, 1]
    corr_idio = np.corrcoef(realized, idio)[0, 1]
    total_var = np.var(realized)
    factor_var = np.var(factor)
    idio_var = np.var(idio)
    explained_pct = 100 * factor_var / total_var
    
    ica_results.append({
        'Asset': asset,
        'Total Variance': total_var,
        'Factor Variance': factor_var,
        'Idiosyncratic Variance': idio_var,
        'Explained Variance %': explained_pct,
        'ICA Factor Corr': corr_factor,
        'ICA Idio Corr': corr_idio
    })

ica_df = pd.DataFrame(ica_results)
ica_df


# In[ ]:





# In[ ]:





# In[56]:


# Model A vs. Model B
# Compare MAC3 vs ICA Decompositions

comparison = mac3_df[['Asset', 'R-squared']].merge(
    pca_df[['Asset', 'Explained Variance %']], on='Asset'
)

print(comparison)


# In[ ]:





# In[ ]:





# In[ ]:





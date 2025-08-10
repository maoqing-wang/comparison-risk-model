# generate_synthetic_ewma_compare.py

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import argparse

np.random.seed(42)

OUTPUT_ROOT = "synthetic_ewma_compare"

nAssets   = 50
nDays     = 252 * 5
nFactors  = 6
lookback  = 60
horizons  = [1, 5, 21, 63]
hlabels   = ['h1d', 'h1w', 'h1m', 'h1q']

baseDate  = datetime(2020, 1, 1)


def ensure_dirs(root: Path):
    data_dir   = root / "data"
    shared_dir = root / "shared"
    models_dir = root / "models"
    for d in [data_dir, shared_dir, models_dir]:
        d.mkdir(parents=True, exist_ok=True)
    return data_dir, shared_dir, models_dir


def main(out_root: str):
    root = Path(out_root)
    data_dir, shared_dir, models_dir = ensure_dirs(root)

    # Dates & IDs
    dates    = pd.date_range(baseDate, periods=nDays, freq='D')
    assetIDs = [f"Asset{i+1}" for i in range(nAssets)]

    # Step 1: Simulate returns
    dailySigma = 0.02
    returns = dailySigma * np.random.randn(nDays, nAssets)
    returns_df = pd.DataFrame(returns, index=dates, columns=assetIDs)

    returns_df.to_csv(data_dir / "returns.csv")
    pd.DataFrame({'dates': dates}).to_csv(data_dir / "dates.csv", index=False)
    pd.DataFrame({'assetIDs': assetIDs}).to_csv(data_dir / "assetIDs.csv", index=False)

    # Step 2: Shared artefacts
    alpha = 0.05 * np.random.randn(nAssets)
    pd.Series(alpha, index=assetIDs).to_csv(shared_dir / "alpha_vector.csv", header=['alpha'])

    constraints = pd.DataFrame({
        'Rebalance': ['Monthly'],
        'MaxGrossExposure': [1.0],
        'LongOnly': [True],
        'TurnoverLimit': [0.25]
    })
    constraints.to_csv(shared_dir / "constraints.csv", index=False)

    horizon_map = pd.DataFrame({'horizonLabels': ['1d','1w','1m','1q'], 'horizons': horizons})
    horizon_map.to_csv(shared_dir / "horizon_map.csv", index=False)

    # Step 3: Build two models
    for model_name in ['modelA', 'modelB']:
        mdlDir = models_dir / model_name
        mdlDir.mkdir(parents=True, exist_ok=True)

        if model_name == 'modelA':
            # Rolling window structure factors
            X = np.random.randn(nAssets, nFactors)
            X = (X - X.mean()) / X.std()
            pd.DataFrame(X, index=assetIDs, columns=[f"Factor{j+1}" for j in range(nFactors)])\
              .to_csv(mdlDir / "exposures_X.csv")

            pinvXX     = np.linalg.pinv(X.T @ X) @ X.T
            factorRets = (pinvXX @ returns.T).T
            residuals  = returns - (X @ factorRets.T).T

            sigma_dict = {h: [] for h in hlabels}
            F_list, Delta_list = [], []
            omega_dir = mdlDir / "omega"
            omega_dir.mkdir(parents=True, exist_ok=True)

            for t in range(lookback, nDays):
                idx = np.arange(t - lookback, t)
                F_win     = np.cov(factorRets[idx, :], rowvar=False)
                Delta_win = np.var(residuals[idx, :], axis=0)
                F_list.append(F_win.flatten());  Delta_list.append(Delta_win)

                omega_t = X @ F_win @ X.T + np.diag(Delta_win)
                pd.DataFrame(omega_t, index=assetIDs, columns=assetIDs)\
                  .to_csv(omega_dir / f"omega_{dates[t].strftime('%Y%m%d')}.csv")

                assetVars = np.diag(omega_t)
                for h, hlabel in zip(horizons, hlabels):
                    sigma_dict[hlabel].append(np.sqrt(h * assetVars))

            for hlabel in hlabels:
                sigmas_array = np.array(sigma_dict[hlabel])
                pd.DataFrame(sigmas_array, index=dates[lookback:], columns=assetIDs)\
                  .to_csv(mdlDir / f"sigma_forecast_{hlabel}.csv")

            pd.DataFrame(F_list, index=dates[lookback:]).to_csv(mdlDir / "F_rolling.csv")
            pd.DataFrame(Delta_list, index=dates[lookback:], columns=assetIDs)\
              .to_csv(mdlDir / "Delta_rolling.csv")

        else:
            # EWMA PCA
            K_pca = 3
            lambda_ = 2 / (lookback + 1)  # EWMA decay
            omega_dir = mdlDir / "omega"
            omega_dir.mkdir(parents=True, exist_ok=True)

            sigma_dict = {h: [] for h in hlabels}

            # Initialize covariance matrix
            S_ewma = np.cov(returns[:lookback, :], rowvar=False)

            for t in range(lookback, nDays):
                r_t = returns[t, :].reshape(1, -1)
                demeaned = r_t - r_t.mean()
                S_ewma = lambda_ * (demeaned.T @ demeaned) + (1 - lambda_) * S_ewma

                vals, vecs = np.linalg.eigh(S_ewma)
                top = np.argsort(vals)[-K_pca:]
                lam = vals[top][::-1]
                V   = vecs[:, top][:, ::-1]

                S_factor  = (V * lam) @ V.T
                spec_diag = np.clip(np.diag(S_ewma - S_factor), 1e-10, None)
                Omega_t   = S_factor + np.diag(spec_diag)

                pd.DataFrame(Omega_t, index=assetIDs, columns=assetIDs)\
                  .to_csv(omega_dir / f"omega_{dates[t].strftime('%Y%m%d')}.csv")

                assetVars = np.diag(Omega_t)
                for h, hlabel in zip(horizons, hlabels):
                    sigma_dict[hlabel].append(np.sqrt(h * assetVars))

            for hlabel in hlabels:
                sigmas_array = np.array(sigma_dict[hlabel])
                pd.DataFrame(sigmas_array, index=dates[lookback:], columns=assetIDs)\
                  .to_csv(mdlDir / f"sigma_forecast_{hlabel}.csv")

    print(f"Done. All files saved under: {root.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic data (ModelA rolling vs ModelB EWMA)")
    parser.add_argument("--out", type=str, default=OUTPUT_ROOT)
    args = parser.parse_args()
    main(args.out)
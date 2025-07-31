# run_pipeline.py
import numpy as np
import pandas as pd

from config import RETURNS_CSV, OMEGA_DIRS, HORIZONS, WEIGHTS, LOOKBACK_WINDOW
from data_utils import load_returns
from vol_pred import portfolio_vol_pred
from port_returns import portfolio_returns
from metrics import compute_zscore, compute_bias_stat, compute_q_stat
from plotting import plot_term_structure, plot_bias_q

def main():

    # Load Data
    returns_df = load_returns(RETURNS_CSV)
    assets = returns_df.columns.tolist()
    n = len(assets)

    results = []
    for w_name, w_func in WEIGHTS.items():
        if w_name == "equal":
            w = w_func(n)
        elif w_name == "inv_hist_vol":
            wts_df = w_func(returns_df, LOOKBACK_WINDOW)
            w = wts_df.iloc[-1].values
        else:
            w = w_func(n)

        for model_name, omega_dir in OMEGA_DIRS.items():
            for H in HORIZONS:
                # Predict Volatility & Portfolio Returns
                vol_pred = portfolio_vol_pred(omega_dir, assets, w, H)
                port_ret = portfolio_returns(returns_df, w)

                # Compute z-score, bias-stat, Q-stat
                z = compute_zscore(port_ret, vol_pred, H)
                bias = compute_bias_stat(z)
                q = compute_q_stat(z)

                results.append({
                    "Weight": w_name,
                    "Model": model_name,
                    "Horizon": H,
                    "Bias Statistic": bias,
                    "Q Statistic": q
                })
                print(f"{w_name:>6} | {model_name:>6} | H={H:2d} "
                      f"=> Bias={bias:.4f}, Q={q:.4f}")

    # Save Results
    results_df = pd.DataFrame(results)
    results_df.to_csv("vol_forecast_stats.csv", index=False)
    print("\nAll results saved to vol_forecast_stats.csv")

    plot_term_structure(results_df, stat_col="Bias Statistic")
    plot_term_structure(results_df, stat_col="Q Statistic")

    # (Optional) Plot comparison of Bias and Q
    plot_bias_q(results_df)

if __name__ == "__main__":
    main()
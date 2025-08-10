from config_new import RETURNS_CSV, OMEGA_DIRS, HORIZONS, WEIGHTS, LOOKBACK_WINDOW
from data_utils import load_returns
from vol_pred import portfolio_vol_pred
from port_returns import portfolio_returns
from metrics import compute_zscore, compute_bias_stat, compute_q_stat
from plotting import plot_term_structure, plot_bias_q

def main():
    # Load returns
    returns_df = load_returns(RETURNS_CSV)
    assets = returns_df.columns.tolist()

    results = []
    for w_name, w_func in WEIGHTS.items():
        w = w_func(returns_df, LOOKBACK_WINDOW)

        for model_name, omega_dir in OMEGA_DIRS.items():
            for H in HORIZONS:
                vol_pred = portfolio_vol_pred(omega_dir, assets, w, H)
                port_ret = portfolio_returns(returns_df, w)

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
                print(f"{w_name:>12} | {model_name:>6} | H={H:2d} "
                      f"=> Bias={bias:.4f}, Q={q:.4f}")

    # Save
    import pandas as pd
    results_df = pd.DataFrame(results)
    results_df.to_csv("vol_forecast_stats.csv", index=False)

    plot_term_structure(results_df, stat_col="Bias Statistic")
    plot_term_structure(results_df, stat_col="Q Statistic")
    plot_bias_q(results_df)

if __name__ == "__main__":
    main()
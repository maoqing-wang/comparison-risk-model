# plotting.py
import matplotlib.pyplot as plt
from typing import List, Optional
import pandas as pd

def plot_term_structure(
    df: pd.DataFrame,
    stat_col: str,
    models: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Plot term-structure (statistic vs. horizon) for one statistic.
    """
    if models is None:
        models = df['Model'].unique().tolist()

    plt.figure()
    for model in models:
        sub = df[df['Model'] == model]
        plt.plot(
            sub['Horizon'],
            sub[stat_col],
            marker='o',
            label=model
        )

    plt.xlabel("Horizon (days)")
    plt.ylabel(stat_col)
    plt.title(f"{stat_col} vs. Horizon")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

def plot_bias_q(
    df: pd.DataFrame,
    models: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Plot bias and Q statistics across horizons for multiple models.
    """
    if models is None:
        models = df['Model'].unique().tolist()

    plt.figure()
    for model in models:
        sub = df[df['Model'] == model].sort_values('Horizon')
        plt.plot(
            sub['Horizon'],
            sub['Bias Statistic'],
            marker='o',
            linestyle='-',
            label=f"{model} Bias"
        )
        plt.plot(
            sub['Horizon'],
            sub['Q Statistic'],
            marker='s',
            linestyle='--',
            label=f"{model} Q"
        )

    plt.xlabel("Horizon (days)")
    plt.ylabel("Statistic Value")
    plt.title("Bias vs. Q Statistics Across Horizons")
    plt.legend(loc="best")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()
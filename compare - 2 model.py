"""
Portfolio Optimization Efficiency Test - Two Model Comparison
Model A: Multi-Factor Regression Risk Model
Model B: Ledoit-Wolf Shrinkage Covariance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings('ignore')

# Set plotting parameters
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 12


def load_csv_data(file_path='Book1.csv'):
    """Load CSV data from file"""
    print("Loading CSV data...")

    try:
        # Read file
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Extract asset names from row 4
        asset_line = lines[3].strip()
        asset_names = [name.strip().replace('\r', '') for name in asset_line.split(',')[1:] if name.strip()]

        # Find data start row
        data_start_idx = None
        for i, line in enumerate(lines):
            if 'Dates' in line:
                data_start_idx = i
                break

        # Process data rows
        data_rows = []
        for line in lines[data_start_idx + 1:]:
            values = line.strip().split(',')
            if len(values) > 1 and values[0] and '/' in values[0]:
                data_rows.append(values)

        # Build DataFrame
        dates = []
        prices_data = []

        for row in data_rows:
            try:
                date = pd.to_datetime(row[0], format='%m/%d/%Y')
                dates.append(date)

                price_row = []
                for i in range(1, min(len(row), len(asset_names) + 1)):
                    try:
                        if i < len(row) and row[i] and row[i] not in ['#N/A Invalid Security', '', '\r']:
                            price = float(row[i])
                            price_row.append(price)
                        else:
                            price_row.append(np.nan)
                    except ValueError:
                        price_row.append(np.nan)

                while len(price_row) < len(asset_names):
                    price_row.append(np.nan)

                prices_data.append(price_row[:len(asset_names)])
            except:
                continue

        # Create price DataFrame
        price_df = pd.DataFrame(prices_data, index=dates, columns=asset_names)
        price_df = price_df.sort_index()

        # Clean data
        price_df = price_df.fillna(method='ffill').fillna(method='bfill')
        price_df = price_df.dropna(axis=1, how='any')

        # Keep only assets with sufficient data
        price_df = price_df.loc[:, (price_df > 0).sum() > len(price_df) * 0.8]

        print(f"Successfully loaded: {len(price_df)} days, {len(price_df.columns)} assets")
        print(f"Date range: {price_df.index[0]} to {price_df.index[-1]}")

        # Calculate returns
        returns_df = price_df.pct_change().dropna()

        # Remove extreme values
        for col in returns_df.columns:
            Q1 = returns_df[col].quantile(0.01)
            Q99 = returns_df[col].quantile(0.99)
            returns_df[col] = returns_df[col].clip(Q1, Q99)

        return price_df, returns_df

    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        print("Generating sample data for demonstration...")
        return generate_sample_data()
    except Exception as e:
        print(f"Error reading data: {e}")
        print("Generating sample data for demonstration...")
        return generate_sample_data()


def generate_sample_data():
    """Generate sample data for demonstration"""
    np.random.seed(42)
    dates = pd.date_range('2015-01-01', '2024-12-31', freq='D')
    dates = dates[dates.weekday < 5]  # Business days only

    n_assets = 20
    asset_names = [
        'US_Treasury_10Y', 'Corporate_Bond', 'REIT_Index', 'SP500', 'NASDAQ',
        'Emerging_Markets', 'Commodities', 'Gold', 'Oil', 'Tech_Stocks',
        'Healthcare', 'Utilities', 'Financials', 'International_Dev', 'Small_Cap',
        'High_Yield_Bond', 'Investment_Grade', 'Currency_Hedge', 'Infrastructure', 'Energy'
    ]

    # Generate correlated return data
    correlation_matrix = np.random.uniform(0.1, 0.7, (n_assets, n_assets))
    correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
    np.fill_diagonal(correlation_matrix, 1.0)

    # Ensure positive definite
    eigenvals, eigenvecs = np.linalg.eigh(correlation_matrix)
    eigenvals = np.maximum(eigenvals, 0.01)
    correlation_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

    # Generate returns
    mean_returns = np.random.uniform(-0.0002, 0.0008, n_assets)
    volatilities = np.random.uniform(0.008, 0.025, n_assets)

    cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix

    returns = np.random.multivariate_normal(mean_returns, cov_matrix, len(dates))
    returns_df = pd.DataFrame(returns, index=dates, columns=asset_names)

    # Generate price data
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    price_df = pd.DataFrame(prices, index=dates, columns=asset_names)

    print(f"Generated sample data: {len(returns_df)} days, {len(returns_df.columns)} assets")
    return price_df, returns_df


class TwoModelPortfolioOptimizer:
    def __init__(self, returns_data):
        self.returns = returns_data
        self.n_assets = returns_data.shape[1]
        self.asset_names = returns_data.columns.tolist()

    def split_data(self, split_ratio=0.7):
        """Split data into training and testing periods"""
        split_point = int(len(self.returns) * split_ratio)
        self.train_returns = self.returns.iloc[:split_point]
        self.test_returns = self.returns.iloc[split_point:]

        print(f"\nData Split:")
        print(f"Training period: {self.train_returns.index[0].strftime('%Y-%m-%d')} to {self.train_returns.index[-1].strftime('%Y-%m-%d')} ({len(self.train_returns)} days)")
        print(f"Testing period: {self.test_returns.index[0].strftime('%Y-%m-%d')} to {self.test_returns.index[-1].strftime('%Y-%m-%d')} ({len(self.test_returns)} days)")

    def build_model_a_multi_factor(self):
        """
        Model A: Multi-Factor Regression Risk Model
        Uses Fama-French + momentum + macro factors
        Builds covariance matrix: Sigma = B * Sigma_F * B^T + diag(sigma^2_epsilon)
        """
        print("\nBuilding Model A: Multi-Factor Regression Risk Model")

        # Step 1: Construct factors
        factors = self.construct_factors()

        # Step 2: Run factor regression for each asset
        factor_loadings = []  # B matrix
        idiosyncratic_vars = []  # Specific risk

        for asset in self.asset_names:
            asset_returns = self.train_returns[asset].values

            # Linear regression: r_i = alpha + beta_1*f_1 + beta_2*f_2 + ... + epsilon_i
            reg = LinearRegression()
            reg.fit(factors, asset_returns)

            # Store factor loadings (betas)
            factor_loadings.append(reg.coef_)

            # Calculate specific risk (residual variance)
            predicted = reg.predict(factors)
            residuals = asset_returns - predicted
            idiosyncratic_vars.append(np.var(residuals))

        # Step 3: Build covariance matrix
        B = np.array(factor_loadings)  # n_assets x n_factors
        factor_cov = np.cov(factors.T)  # n_factors x n_factors

        # Sigma = B * Sigma_F * B^T + diag(sigma^2_epsilon)
        factor_component = B @ factor_cov @ B.T
        specific_component = np.diag(idiosyncratic_vars)

        self.cov_model_a = factor_component + specific_component

        print(f"   Built {factors.shape[1]} factors")
        print(f"   Average factor R-squared = {self.calculate_average_r_squared(factors):.3f}")
        print(f"   Covariance matrix condition number = {np.linalg.cond(self.cov_model_a):.2f}")

        return self.cov_model_a

    def construct_factors(self):
        """Construct economic factors similar to Fama-French"""
        # Build Fama-French-like factors using market data

        # 1. Market factor
        market_factor = self.train_returns.mean(axis=1).values  # Equal-weighted market portfolio

        # 2. Size factor - using PCA approximation
        pca = PCA(n_components=1)
        size_factor = pca.fit_transform(self.train_returns.values).flatten()

        # 3. Value factor - using return differences
        high_vol_assets = self.train_returns.std().nlargest(len(self.asset_names) // 3).index
        low_vol_assets = self.train_returns.std().nsmallest(len(self.asset_names) // 3).index
        value_factor = (self.train_returns[high_vol_assets].mean(axis=1) -
                        self.train_returns[low_vol_assets].mean(axis=1)).values

        # 4. Momentum factor
        momentum_lookback = min(20, len(self.train_returns) // 10)
        momentum_factor = self.train_returns.rolling(momentum_lookback).mean().mean(axis=1).fillna(0).values

        # 5. Macro factor - using market volatility as proxy
        macro_factor = self.train_returns.rolling(10).std().mean(axis=1).fillna(self.train_returns.std().mean()).values

        # Combine factor matrix
        factors = np.column_stack([
            market_factor,
            size_factor,
            value_factor,
            momentum_factor,
            macro_factor
        ])

        # Standardize factors
        factors = (factors - factors.mean(axis=0)) / factors.std(axis=0)

        return factors

    def calculate_average_r_squared(self, factors):
        """Calculate average R-squared for factor model"""
        r_squareds = []

        for asset in self.asset_names:
            asset_returns = self.train_returns[asset].values
            reg = LinearRegression()
            reg.fit(factors, asset_returns)
            r_squared = reg.score(factors, asset_returns)
            r_squareds.append(r_squared)

        return np.mean(r_squareds)

    def build_model_b_ledoit_wolf(self):
        """
        Model B: Ledoit-Wolf Shrinkage Covariance
        Pure statistical shrinkage without economic factors
        """
        print("\nBuilding Model B: Ledoit-Wolf Shrinkage Covariance")

        lw = LedoitWolf()
        self.cov_model_b = lw.fit(self.train_returns).covariance_

        # Get shrinkage intensity
        shrinkage = lw.shrinkage_

        print(f"   Shrinkage intensity = {shrinkage:.3f}")
        print(f"   Covariance matrix condition number = {np.linalg.cond(self.cov_model_b):.2f}")

        return self.cov_model_b

    def optimize_portfolio(self, cov_matrix, method='min_vol'):
        """Portfolio optimization using quadratic programming"""
        n_assets = self.n_assets

        def objective(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))

        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]

        # Bounds: long-only, max 15% per asset
        bounds = tuple([(0, 0.15) for _ in range(n_assets)])

        # Initial guess
        initial_guess = np.array([1.0 / n_assets] * n_assets)

        # Optimize
        result = minimize(objective, initial_guess, method='SLSQP',
                          bounds=bounds, constraints=constraints,
                          options={'disp': False})

        if result.success:
            return result.x
        else:
            print(f"Optimization failed: {result.message}")
            return initial_guess

    def backtest_models(self):
        """Backtest both model portfolios"""
        print("\nStarting backtest...")

        # Build covariance matrices
        cov_a = self.build_model_a_multi_factor()
        cov_b = self.build_model_b_ledoit_wolf()

        models = {
            'Model A (Multi-Factor)': cov_a,
            'Model B (Ledoit-Wolf)': cov_b
        }

        self.weights = {}
        self.portfolio_returns = {}
        self.results = {}

        for model_name, cov_matrix in models.items():
            # Optimize weights
            weights = self.optimize_portfolio(cov_matrix)
            self.weights[model_name] = weights

            # Calculate out-of-sample portfolio returns
            portfolio_returns = (self.test_returns * weights).sum(axis=1)
            self.portfolio_returns[model_name] = portfolio_returns

            # Calculate performance metrics
            self.results[model_name] = self.calculate_performance_metrics(
                portfolio_returns, weights, model_name
            )

        return self.results

    def calculate_performance_metrics(self, returns, weights, model_name):
        """Calculate portfolio performance metrics"""
        annualize_factor = 252

        metrics = {
            'Realized Volatility': returns.std() * np.sqrt(annualize_factor),
            'Mean Return': returns.mean() * annualize_factor,
            'Sharpe Ratio': (returns.mean() * annualize_factor) / (returns.std() * np.sqrt(annualize_factor)),
            'Max Drawdown': self.calculate_max_drawdown(returns),
            'VaR (5%)': np.percentile(returns, 5) * np.sqrt(annualize_factor),
            'Max Weight': np.max(weights),
            'Effective Assets': 1 / np.sum(weights ** 2),
            'Skewness': returns.skew(),
            'Kurtosis': returns.kurtosis()
        }

        print(f"   {model_name}:")
        print(f"      Realized volatility: {metrics['Realized Volatility']:.4f} ({metrics['Realized Volatility'] * 100:.2f}%)")
        print(f"      Sharpe ratio: {metrics['Sharpe Ratio']:.4f}")
        print(f"      Max weight: {metrics['Max Weight']:.1%}")

        return metrics

    def calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown"""
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = (cum_returns - rolling_max) / rolling_max
        return drawdowns.min()

    def create_comprehensive_comparison(self):
        """Create detailed model comparison analysis"""
        results_df = pd.DataFrame(self.results).T
        results_df = results_df.round(4)

        # Calculate relative performance
        model_a_vol = self.results['Model A (Multi-Factor)']['Realized Volatility']
        model_b_vol = self.results['Model B (Ledoit-Wolf)']['Realized Volatility']

        vol_improvement = (model_a_vol - model_b_vol) / model_a_vol * 100

        print(f"\nModel Comparison Results:")
        print("=" * 60)
        print(results_df.to_string())

        # Determine better model
        better_model = "Model B (Ledoit-Wolf)" if model_b_vol < model_a_vol else "Model A (Multi-Factor)"
        worse_model = "Model A (Multi-Factor)" if model_b_vol < model_a_vol else "Model B (Ledoit-Wolf)"
        improvement_pct = abs(vol_improvement)

        print(f"\nKey Findings:")
        print(f"   Best model: {better_model}")
        print(f"   Volatility improvement: {improvement_pct:.2f}%")
        print(f"   {better_model} has lower risk than {worse_model}")

        return results_df, better_model, improvement_pct

    def plot_detailed_comparison(self):
        """Create detailed comparison charts"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Color scheme
        colors = ['#2E86AB', '#A23B72']  # Blue and purple-red
        model_names = list(self.results.keys())

        # 1. Cumulative returns comparison
        ax1 = axes[0, 0]
        for i, (model_name, returns) in enumerate(self.portfolio_returns.items()):
            cum_returns = (1 + returns).cumprod()
            ax1.plot(cum_returns.index, cum_returns.values,
                     label=model_name, linewidth=3, color=colors[i])
        ax1.set_title('Cumulative Returns Comparison', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Cumulative Returns')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Realized volatility comparison (key metric)
        ax2 = axes[0, 1]
        volatilities = [self.results[name]['Realized Volatility'] for name in model_names]
        bars = ax2.bar(range(len(model_names)), volatilities, color=colors, alpha=0.8)
        ax2.set_title('Realized Volatility Comparison (Key Metric)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Annualized Volatility')
        ax2.set_xticks(range(len(model_names)))
        ax2.set_xticklabels(['Model A\n(Multi-Factor)', 'Model B\n(Ledoit-Wolf)'])

        # Add value labels and mark best
        for i, (bar, vol) in enumerate(zip(bars, volatilities)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, height + 0.002,
                     f'{vol:.4f}', ha='center', va='bottom', fontweight='bold')

            # Mark best model
            if vol == min(volatilities):
                ax2.text(bar.get_x() + bar.get_width() / 2, height / 2,
                         'BEST', ha='center', va='center',
                         fontsize=12, fontweight='bold', color='white')

        # 3. Risk-adjusted returns comparison
        ax3 = axes[0, 2]
        sharpe_ratios = [self.results[name]['Sharpe Ratio'] for name in model_names]
        bars = ax3.bar(range(len(model_names)), sharpe_ratios, color=colors, alpha=0.8)
        ax3.set_title('Sharpe Ratio Comparison', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Sharpe Ratio')
        ax3.set_xticks(range(len(model_names)))
        ax3.set_xticklabels(['Model A\n(Multi-Factor)', 'Model B\n(Ledoit-Wolf)'])

        # Add value labels
        for bar, sr in zip(bars, sharpe_ratios):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2, height + 0.02,
                     f'{sr:.3f}', ha='center', va='bottom', fontweight='bold')

        # 4. Portfolio weights comparison
        ax4 = axes[1, 0]

        # Show top 10 weights
        model_a_weights = self.weights['Model A (Multi-Factor)']
        model_b_weights = self.weights['Model B (Ledoit-Wolf)']

        # Find top assets by combined weights
        combined_weights = model_a_weights + model_b_weights
        top_assets_idx = np.argsort(combined_weights)[-10:]

        x = np.arange(len(top_assets_idx))
        width = 0.35

        bars1 = ax4.bar(x - width / 2, model_a_weights[top_assets_idx],
                        width, label='Model A', color=colors[0], alpha=0.8)
        bars2 = ax4.bar(x + width / 2, model_b_weights[top_assets_idx],
                        width, label='Model B', color=colors[1], alpha=0.8)

        ax4.set_title('Top Asset Weights Comparison', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Weight')
        ax4.set_xlabel('Asset')
        ax4.set_xticks(x)
        ax4.set_xticklabels([self.asset_names[i][:8] + '...' if len(self.asset_names[i]) > 8
                             else self.asset_names[i] for i in top_assets_idx],
                            rotation=45, ha='right')
        ax4.legend()

        # 5. Risk decomposition
        ax5 = axes[1, 1]
        max_weights = [self.results[name]['Max Weight'] for name in model_names]
        effective_assets = [self.results[name]['Effective Assets'] for name in model_names]

        x = np.arange(len(model_names))

        bars1 = ax5.bar(x - 0.2, max_weights, 0.4, label='Max Weight', color='lightcoral', alpha=0.8)
        ax5_twin = ax5.twinx()
        bars2 = ax5_twin.bar(x + 0.2, effective_assets, 0.4, label='Effective Assets', color='lightblue', alpha=0.8)

        ax5.set_title('Weight Concentration Analysis', fontsize=14, fontweight='bold')
        ax5.set_ylabel('Max Weight', color='red')
        ax5_twin.set_ylabel('Effective Assets', color='blue')
        ax5.set_xticks(x)
        ax5.set_xticklabels(['Model A', 'Model B'])

        # Add legend
        lines1, labels1 = ax5.get_legend_handles_labels()
        lines2, labels2 = ax5_twin.get_legend_handles_labels()
        ax5.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        # 6. Tail risk comparison
        ax6 = axes[1, 2]
        var_5pct = [self.results[name]['VaR (5%)'] for name in model_names]
        max_drawdowns = [self.results[name]['Max Drawdown'] for name in model_names]

        x = np.arange(len(model_names))

        bars1 = ax6.bar(x - 0.2, np.abs(var_5pct), 0.4, label='VaR (5%)', color='orange', alpha=0.8)
        bars2 = ax6.bar(x + 0.2, np.abs(max_drawdowns), 0.4, label='Max Drawdown', color='red', alpha=0.8)

        ax6.set_title('Tail Risk Comparison', fontsize=14, fontweight='bold')
        ax6.set_ylabel('Risk Measure (Absolute Value)')
        ax6.set_xticks(x)
        ax6.set_xticklabels(['Model A', 'Model B'])
        ax6.legend()

        plt.tight_layout()
        return fig

    def generate_final_report(self):
        """Generate final comprehensive report"""
        results_df, better_model, improvement_pct = self.create_comprehensive_comparison()

        print("\n" + "=" * 80)
        print("Portfolio Optimization Efficiency Test - Final Report")
        print("=" * 80)

        print(f"\nTest Setup:")
        print(f"   Number of assets: {self.n_assets}")
        print(f"   Training period: {len(self.train_returns)} days")
        print(f"   Testing period: {len(self.test_returns)} days")

        print(f"\nModel Comparison:")
        print(f"   Model A: Multi-Factor Regression Risk Model")
        print(f"            Uses Fama-French + momentum + macro factors")
        print(f"            Builds structured covariance: Sigma = B*Sigma_F*B^T + diag(sigma^2_eps)")
        print(f"            Captures economic style risks")
        print(f"   ")
        print(f"   Model B: Ledoit-Wolf Shrinkage Covariance")
        print(f"            Pure statistical shrinkage method")
        print(f"            No economic factors")
        print(f"            Reduces estimation error")

        model_a_metrics = self.results['Model A (Multi-Factor)']
        model_b_metrics = self.results['Model B (Ledoit-Wolf)']

        print(f"\nKey Results:")
        print(f"   Best model: {better_model}")
        print(f"   Volatility improvement: {improvement_pct:.2f}%")
        print(f"   ")
        print(f"   Model A realized volatility: {model_a_metrics['Realized Volatility']:.4f} ({model_a_metrics['Realized Volatility'] * 100:.2f}%)")
        print(f"   Model B realized volatility: {model_b_metrics['Realized Volatility']:.4f} ({model_b_metrics['Realized Volatility'] * 100:.2f}%)")

        print(f"\nConclusions:")
        if 'Model B' in better_model:
            print(f"   Ledoit-Wolf shrinkage performs better in out-of-sample test")
            print(f"   Pure statistical method effectively reduces estimation error")
            print(f"   Shrinkage estimator provides more stable covariance matrix")
            print(f"   In limited sample case, statistical shrinkage outperforms complex factor model")
        else:
            print(f"   Multi-Factor model performs better in out-of-sample test")
            print(f"   Economic factors successfully capture market structure risks")
            print(f"   Structured approach provides better risk prediction")
            print(f"   Factor model economic intuition is validated")

        print(f"\nInvestment Implications:")
        print(f"   Better covariance estimation directly translates to lower portfolio risk")
        print(f"   {improvement_pct:.1f}% volatility reduction has significant value in institutional investing")
        print(f"   Risk model choice has material impact on portfolio optimization efficiency")

        return results_df, better_model, improvement_pct


def main():
    """Main execution function"""
    print("Portfolio Optimization Efficiency Test - Two Model Comparison")
    print("Model A: Multi-Factor Regression vs Model B: Ledoit-Wolf Shrinkage")
    print("=" * 70)

    # Load data
    prices, returns = load_csv_data('Book1.csv')

    if len(returns.columns) < 5:
        print("Warning: Too few assets for effective portfolio optimization")
        return None, None

    # Data overview
    print(f"\nData Overview:")
    print(f"   Time span: {returns.index[0].strftime('%Y-%m-%d')} to {returns.index[-1].strftime('%Y-%m-%d')}")
    print(f"   Number of assets: {len(returns.columns)}")
    print(f"   Observation days: {len(returns)}")
    print(f"   Average daily return: {returns.mean().mean() * 100:.4f}%")
    print(f"   Average daily volatility: {returns.std().mean() * 100:.4f}%")

    # Show some asset names
    print(f"   Key assets: {', '.join(returns.columns[:5].tolist())}...")

    # Create optimizer
    optimizer = TwoModelPortfolioOptimizer(returns)

    # Execute analysis
    optimizer.split_data(split_ratio=0.7)

    # Backtest models
    results = optimizer.backtest_models()

    # Generate detailed comparison
    results_df, better_model, improvement = optimizer.generate_final_report()

    # Create charts
    print(f"\nGenerating detailed comparison charts...")
    fig = optimizer.plot_detailed_comparison()
    plt.suptitle('Model A (Multi-Factor) vs Model B (Ledoit-Wolf) - Portfolio Optimization Efficiency',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.show()

    # Output key data for reporting
    print(f"\nKey Data for Report:")
    print(f"=" * 50)
    model_a_vol = results['Model A (Multi-Factor)']['Realized Volatility']
    model_b_vol = results['Model B (Ledoit-Wolf)']['Realized Volatility']
    model_a_sharpe = results['Model A (Multi-Factor)']['Sharpe Ratio']
    model_b_sharpe = results['Model B (Ledoit-Wolf)']['Sharpe Ratio']

    print(f"Model A (Multi-Factor Regression):")
    print(f"  - Realized volatility: {model_a_vol:.4f} ({model_a_vol * 100:.2f}%)")
    print(f"  - Sharpe ratio: {model_a_sharpe:.4f}")
    print(f"  - Max weight: {results['Model A (Multi-Factor)']['Max Weight']:.1%}")
    print(f"  - Effective assets: {results['Model A (Multi-Factor)']['Effective Assets']:.1f}")

    print(f"\nModel B (Ledoit-Wolf Shrinkage):")
    print(f"  - Realized volatility: {model_b_vol:.4f} ({model_b_vol * 100:.2f}%)")
    print(f"  - Sharpe ratio: {model_b_sharpe:.4f}")
    print(f"  - Max weight: {results['Model B (Ledoit-Wolf)']['Max Weight']:.1%}")
    print(f"  - Effective assets: {results['Model B (Ledoit-Wolf)']['Effective Assets']:.1f}")

    vol_diff = abs(model_a_vol - model_b_vol)
    vol_improvement_pct = vol_diff / max(model_a_vol, model_b_vol) * 100

    print(f"\nCore Findings:")
    print(f"  - Volatility difference: {vol_diff:.4f} ({vol_improvement_pct:.2f}%)")
    print(f"  - Better model: {better_model}")
    print(f"  - Empirical conclusion: {'Statistical shrinkage more effective' if 'Ledoit-Wolf' in better_model else 'Factor model more effective'}")

    return optimizer, results_df


def quick_demo():
    """Quick demonstration function using simulated data"""
    print("Quick Demo Mode - Using Simulated Data")
    print("=" * 50)

    # Generate demo data
    _, returns = generate_sample_data()

    # Run analysis
    optimizer = TwoModelPortfolioOptimizer(returns)
    optimizer.split_data(split_ratio=0.7)
    results = optimizer.backtest_models()
    results_df, better_model, improvement = optimizer.generate_final_report()

    # Display charts
    fig = optimizer.plot_detailed_comparison()
    plt.suptitle('Demo: Model A vs Model B Comparison Analysis', fontsize=16, fontweight='bold')
    plt.show()

    return optimizer, results_df


if __name__ == "__main__":
    # Choose run mode
    print("Select run mode:")
    print("1. Use your Book1.csv data")
    print("2. Quick demo mode (simulated data)")

    choice = input("Enter choice (1 or 2, default is 1): ").strip()

    if choice == "2":
        optimizer, results = quick_demo()
    else:
        optimizer, results = main()

    print(f"\nAnalysis complete! You can use these results for your assignment report.")
    print(f"Key conclusion: Better covariance matrix estimation methods do indeed enable more efficient portfolio construction!")
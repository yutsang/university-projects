import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sco
from typing import List


def get_stock(path: str) -> List[str]:
    """Get list of stock symbols from CSV files in the given directory."""
    result = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith('.csv'):
                result.append(file[:-4])
    return result


def get_price(stock: List[str], path: str) -> pd.DataFrame:
    """Read closing prices for each stock and return as a DataFrame indexed by Date."""
    result = {}
    for i in stock:
        data = pd.read_csv(os.path.join(path, f'{i}.csv'))
        result[i] = data["Close"].values.tolist()
        result["Date"] = data["Date"].values.tolist()
    result = pd.DataFrame(result, index=result["Date"])
    result.drop(columns=["Date"], inplace=True)
    return result


def statistics(weights, returns, cov_matrix):
    """Return portfolio return, volatility, and Sharpe ratio."""
    weights = np.array(weights)
    port_return = np.sum(returns.mean() * weights) * 252
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    sharpe = port_return / port_vol
    return np.array([port_return, port_vol, sharpe])


def run_analysis(stock_path: str, results_dir: str = "results"):
    os.makedirs(results_dir, exist_ok=True)
    stock = get_stock(stock_path)
    data = get_price(stock, stock_path)

    # Plot normalized prices
    plt.figure(figsize=(15, 6))
    (data / data.iloc[0] * 100).plot(ax=plt.gca())
    plt.legend(loc="best")
    plt.title("Normalized Stock Prices")
    plt.savefig(os.path.join(results_dir, "normalized_prices.png"))
    plt.close()

    # Calculate returns and covariance
    returns = np.log(data / data.shift(1))
    cov_matrix = returns.cov() * 252
    noa = len(stock)

    # Monte Carlo simulation
    port_returns = []
    port_vols = []
    port_weights = []
    for _ in range(4000):
        weights = np.random.random(noa)
        weights /= np.sum(weights)
        port_weights.append(weights)
        port_returns.append(np.sum(returns.mean() * 252 * weights))
        port_vols.append(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))))
    port_returns = np.array(port_returns)
    port_vols = np.array(port_vols)

    # Plot random portfolios
    risk_free = 0.04
    plt.figure(figsize=(8, 4))
    plt.scatter(port_vols, port_returns, c=(port_returns - risk_free) / port_vols, marker='o')
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.title('Random Portfolios')
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, "random_portfolios.png"))
    plt.close()

    # Portfolio optimization: Max Sharpe
    def min_sharpe(weights):
        return -statistics(weights, returns, cov_matrix)[2]
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},)
    bnds = tuple((0, 1) for _ in range(noa))
    opts = sco.minimize(min_sharpe, noa * [1. / noa, ], method='SLSQP', bounds=bnds, constraints=cons)
    max_sharpe_weights = opts['x']
    max_sharpe_stats = statistics(max_sharpe_weights, returns, cov_matrix)

    # Portfolio optimization: Min variance
    def min_variance(weights):
        return statistics(weights, returns, cov_matrix)[1]
    optv = sco.minimize(min_variance, noa * [1. / noa, ], method='SLSQP', bounds=bnds, constraints=cons)
    min_var_weights = optv['x']
    min_var_stats = statistics(min_var_weights, returns, cov_matrix)

    # Efficient frontier
    target_returns = np.linspace(0.0, 0.5, 50)
    target_vols = []
    for tar in target_returns:
        cons2 = ({'type': 'eq', 'fun': lambda x: statistics(x, returns, cov_matrix)[0] - tar},
                 {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        res = sco.minimize(min_variance, noa * [1. / noa, ], method='SLSQP', bounds=bnds, constraints=cons2)
        target_vols.append(res['fun'])
    target_vols = np.array(target_vols)

    # Plot efficient frontier
    plt.figure(figsize=(8, 4))
    plt.scatter(port_vols, port_returns, c=port_returns / port_vols, marker='o', alpha=0.3, label='Random Portfolios')
    plt.scatter(target_vols, target_returns, c=target_returns / target_vols, marker='x', label='Efficient Frontier')
    plt.plot(max_sharpe_stats[1], max_sharpe_stats[0], 'r*', markersize=15, label='Max Sharpe')
    plt.plot(min_var_stats[1], min_var_stats[0], 'y*', markersize=15, label='Min Variance')
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.title('Efficient Frontier and Optimal Portfolios')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, "efficient_frontier.png"))
    plt.close()

    # Print results
    print("Max Sharpe Portfolio Weights:")
    for s, w in zip(stock, max_sharpe_weights.round(3)):
        print(f"  {s}: {w}")
    print("Stats (Return, Volatility, Sharpe):", max_sharpe_stats.round(3))
    print("\nMin Variance Portfolio Weights:")
    for s, w in zip(stock, min_var_weights.round(3)):
        print(f"  {s}: {w}")
    print("Stats (Return, Volatility, Sharpe):", min_var_stats.round(3))
    print(f"\nPlots saved in '{results_dir}' directory.")


def main():
    parser = argparse.ArgumentParser(description="Stock Portfolio Analysis (HSI Constituents)")
    parser.add_argument('--data', type=str, default='Data/Stock', help='Path to stock CSV files')
    parser.add_argument('--results', type=str, default='results', help='Directory to save results')
    args = parser.parse_args()
    run_analysis(args.data, args.results)


if __name__ == "__main__":
    main() 
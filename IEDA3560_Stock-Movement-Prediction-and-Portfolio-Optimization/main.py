import os
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from math import exp
from datetime import datetime, timedelta
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt import black_litterman, objective_functions
from pypfopt.black_litterman import BlackLittermanModel
from pypfopt.efficient_frontier import EfficientFrontier

REQUIRED_FILES = {
    'S&P500.csv': 'https://drive.google.com/uc?export=download&id=1qLoKEZHEjqvjgBB1CFX62oh7IGJptAk7',
    'market_cap.csv': 'https://drive.google.com/uc?export=download&id=1YZmaQNzgpkj-DHbyFUhzzSDXKajabViF',
    'risk_free.csv': 'https://drive.google.com/uc?export=download&id=1KRYPKo4ZEZdbq0RCmIutu6n93FYAjQDN',
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def check_and_warn_files():
    for fname, url in REQUIRED_FILES.items():
        if not os.path.exists(fname):
            logging.warning(f"Required file '{fname}' not found. Please download it from: {url}")

def find_std(data, period, start_date):
    df = data.copy()
    end = df.index.get_loc(start_date)
    start = end - period
    if start >= 0:
        log_return = df['Log Return'].iloc[start:end]
        return np.std(log_return, ddof=1)
    return np.NaN

def find_beta(data, period, start_date):
    df = data.copy()
    end = df.index.get_loc(start_date)
    start = end - period
    if start >= 0:
        log_return = df['Log Return'].iloc[start:end].to_numpy()
        SnP_log_return = df['SnP Log Return'].iloc[start:end].to_numpy().reshape(-1, 1)
        regr = LinearRegression()
        beta = regr.fit(SnP_log_return, log_return).coef_[0]
        return beta
    return np.NaN

def ShiftNum(var):
    if var[1] == 'D':
        return int(var[0])
    elif var[1] == 'W':
        return int(var[0]) * 5
    elif var[1] == 'M':
        return int(var[0]) * 20
    elif var[1] == 'Y':
        return int(var[0]) * 250
    else:
        raise ValueError("Invalid lag format. Use xD, xW, xM, xY.")

def feature_engineering(data_processed, SnP, tickers):
    stocks = {}
    return_lag = ['1D', '3D', '1W', '2W', '3W', '1M', '6W', '2M', '3M']
    for stock in tickers:
        stocks[stock] = data_processed[stock][['Close']].copy()
        stocks[stock]['Stock'] = [stock] * stocks[stock].shape[0]
        stocks[stock]['Log Return'] = np.log(stocks[stock]['Close']) - np.log(stocks[stock]['Close']).shift(1)
        stocks[stock] = stocks[stock].join(SnP[['SnP Log Return']], how='left')
        stocks[stock]['SnP Log Return_1D'] = stocks[stock]['SnP Log Return'].shift(1)
        stocks[stock].dropna(inplace=True)
        stocks[stock]['Volatility'] = [find_std(stocks[stock], 30, date) for date in stocks[stock].index]
        stocks[stock]['Beta'] = [find_beta(stocks[stock], 30, date) for date in stocks[stock].index]
        for var in return_lag:
            name = 'Return_' + var
            stocks[stock][name] = stocks[stock]['Log Return'].shift(ShiftNum(var))
    cleaned = pd.concat(stocks.values())
    cleaned.dropna(inplace=True)
    return cleaned

def main(args):
    check_and_warn_files()
    # Load tickers
    tickers_csv = pd.read_csv('S&P500.csv')
    tickers = list(set(tickers_csv['Symbol']) - set(["BRK.B","BF.B","EMBC","CEG","OGN","CARR","OTIS","CTVA","MRNA","FOX","FOXA","DOW","CDAY","IR"]))
    # Download price data
    yf.pdr_override()
    end = pd.to_datetime(args.end_date) if args.end_date else pd.to_datetime("2022-04-30")
    start = pd.to_datetime(args.start_date) if args.start_date else end - timedelta(days=5000)
    data = yf.download(tickers, start=start, end=end, group_by="ticker", interval="1d")
    data_processed = data.dropna(axis=0)
    # S&P 500 for comparison
    SnP = yf.download('^GSPC', start=data_processed.index[0].date(), end=datetime.today(), interval='1d')
    SnP['SnP Log Return'] = np.log(SnP['Close']) - np.log(SnP['Close']).shift(1)
    # Feature engineering
    cleaned = feature_engineering(data_processed, SnP, tickers)
    cleaned.index = pd.to_datetime(cleaned.index)
    SnP_Return = cleaned[["SnP Log Return"]]
    cleaned = cleaned.drop(columns=["SnP Log Return","Close"], axis=1)
    cleaned = cleaned.sort_values(by=["Stock","Date"])
    # Market cap and risk-free
    market_cap = pd.read_csv('market_cap.csv').astype('int64').iloc[0].to_dict()
    riskfree = pd.read_csv('risk_free.csv')
    # Prepare lists
    date_list = cleaned.index.drop_duplicates()
    stock_list = cleaned["Stock"].drop_duplicates()
    SnP_Return = SnP_Return.drop_duplicates().reset_index().drop_duplicates().set_index("Date")
    log_return_df = cleaned[['Stock','Log Return']].reset_index().set_index(['Stock','Date']).unstack(level=[0])["Log Return"]
    SPY = yf.download('SPY', start=date_list[0], end=date_list[-1])
    # Main loop
    n_days = args.n_days if args.n_days else 252
    training_len = args.training_len if args.training_len else 100
    trade_days = date_list[-n_days:]
    daily_return, ms_daily_return, mv_daily_return = [], [], []
    seed = 0
    for trade_day in trade_days:
        trade_day_index = date_list.get_loc(trade_day)
        first_training_day_index = trade_day_index - training_len
        train_valid_dates = date_list[first_training_day_index:trade_day_index]
        train_days, eval_days = train_test_split(train_valid_dates, test_size=0.7, random_state=seed)
        seed += 1
        train_data = cleaned.loc[train_days, :]
        eval_data = cleaned.loc[eval_days, :]
        trade_day_data = cleaned.loc[trade_day]
        train_x = train_data.drop(['Log Return'], axis=1)
        train_y = train_data['Log Return']
        eval_x = eval_data.drop(['Log Return'], axis=1)
        eval_y = eval_data['Log Return']
        trade_day_x = trade_day_data.drop(['Log Return'], axis=1)
        trade_day_y = trade_day_data['Log Return']
        model = CatBoostRegressor(iterations=100, task_type="CPU", learning_rate=0.1, depth=8, l2_leaf_reg=1e-7, allow_writing_files=False, eval_metric='MAPE', random_seed=0, thread_count=-1, verbose=0)
        eval_set = Pool(eval_x, eval_y, cat_features=[0])
        catboost_train_data = Pool(data=train_x, label=train_y, cat_features=[0])
        model.fit(catboost_train_data, eval_set=eval_set, early_stopping_rounds=10)
        preds_log_return = model.predict(trade_day_x)
        temp_df = pd.DataFrame(preds_log_return).transpose()
        temp_df.columns = stock_list
        log_return_opt_df = log_return_df[first_training_day_index:trade_day_index]
        log_return_opt_df = log_return_opt_df.append(temp_df)
        log_return_opt_df.index = date_list[first_training_day_index:trade_day_index+1]
        portfolio = log_return_opt_df.applymap(lambda x: exp(x))
        cs_actual = CovarianceShrinkage(portfolio, frequency=len(log_return_opt_df))
        e_cov = cs_actual.ledoit_wolf()
        market_prices = SPY.loc[log_return_opt_df.index[0]:log_return_opt_df.index[-2]]
        annual_risk_free = riskfree.loc[trade_day_index]['Price']/100
        daily_risk_free = (1+annual_risk_free)**(1/252)-1
        delta = black_litterman.market_implied_risk_aversion(market_prices['Close'], risk_free_rate=daily_risk_free)
        prior = black_litterman.market_implied_prior_returns(market_cap, delta, e_cov)
        viewdict = {trade_day_x['Stock'].values[i]: exp(preds_log_return[i])-1 for i in range(len(preds_log_return))}
        bl = BlackLittermanModel(e_cov, pi=prior, absolute_views=viewdict)
        ret_bl = bl.bl_returns()
        S_bl = bl.bl_cov()
        ms_ef = EfficientFrontier(ret_bl, S_bl, verbose=False)
        ms_ef.add_objective(objective_functions.L2_reg)
        try:
            ms_ef.max_sharpe()
            ms_weights = ms_ef.clean_weights()
        except Exception:
            ms_weights = ms_ef.nonconvex_objective(objective_functions.sharpe_ratio, objective_args=(ms_ef.expected_returns, ms_ef.cov_matrix), weights_sum_to_one=True,)
        ms_weights = list(ms_weights.values())
        ms_expected_return = np.dot(ms_weights, temp_df.iloc[0].to_numpy().T)
        ms_return = np.dot(ms_weights, trade_day_y.to_numpy().T)
        ms_daily_return.append(ms_return)
        mv_ef = EfficientFrontier(ret_bl, S_bl)
        mv_ef.add_objective(objective_functions.L2_reg)
        mv_ef.min_volatility()
        mv_weights = mv_ef.clean_weights()
        mv_weights = list(mv_weights.values())
        mv_expected_return = np.dot(mv_weights, temp_df.iloc[0].to_numpy().T)
        mv_return = np.dot(mv_weights, trade_day_y.to_numpy().T)
        mv_daily_return.append(mv_return)
        if ms_expected_return > mv_expected_return:
            daily_return.append(ms_return)
        else:
            daily_return.append(mv_return)
    ms_cum_return = np.cumprod(np.array(ms_daily_return)+1)
    mv_cum_return = np.cumprod(np.array(mv_daily_return)+1)
    cum_return = np.cumprod(np.array(daily_return)+1)
    SnP_array = np.array(SnP_Return["SnP Log Return"].loc[trade_days])
    SnP_cum_return = np.cumprod(SnP_array+1)
    print(f"Max Sharpe Cumulative Return: {ms_cum_return[-1]:.2f}")
    print(f"Min Volatility Cumulative Return: {mv_cum_return[-1]:.2f}")
    print(f"Algorithm Cumulative Return: {cum_return[-1]:.2f}")
    print(f"S&P500 Cumulative Return: {SnP_cum_return[-1]:.2f}")
    avg_return_algorithm = np.array(daily_return).mean()
    avg_return_SnP = SnP_array.mean()
    tracking_error = np.std(np.array(daily_return)-SnP_array, ddof=1)
    IR = (avg_return_algorithm-avg_return_SnP)/tracking_error
    annual_IR = IR * np.sqrt(252)
    print(f"Information Ratio: {IR:.2f}")
    print(f"Annualized IR: {annual_IR:.2f}")
    # Plot
    plt.style.use('ggplot')
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.title("Portfolios performance")
    plt.plot(trade_days, ms_cum_return, label="Maximised Sharpe Ratio")
    plt.plot(trade_days, mv_cum_return, label="Minimised volatility")
    plt.plot(trade_days, cum_return, label="Algorithm")
    plt.plot(trade_days, SnP_cum_return, c="Red", label="S&P500 Index")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig("sample_plot.png")
    plt.show()
    # Save results
    result = pd.DataFrame({'ms_daily_return': ms_daily_return, 'mv_daily_return': mv_daily_return, 'algo_daily_return': daily_return, 'SnP_daily_return': SnP_array}, index=trade_days)
    result.to_csv('daily_returns.csv')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock Movement Prediction and Portfolio Optimization")
    parser.add_argument('--start_date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--n_days', type=int, help='Number of trading days to simulate (default: 252)')
    parser.add_argument('--training_len', type=int, help='Training window length (default: 100)')
    args = parser.parse_args()
    main(args) 
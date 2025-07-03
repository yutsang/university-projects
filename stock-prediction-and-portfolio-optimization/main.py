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
from tqdm import tqdm
import warnings

INPUT_DIR = 'input'

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
warnings.filterwarnings('ignore')
logging.getLogger('yfinance').setLevel(logging.CRITICAL)
logging.getLogger('urllib3').setLevel(logging.CRITICAL)
logging.getLogger('pandas').setLevel(logging.CRITICAL)

def check_input_files():
    for fname in ['S&P500.csv', 'market_cap.csv', 'risk_free.csv']:
        fpath = os.path.join(INPUT_DIR, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Required file '{fpath}' not found. Please add it to the input directory.")

def find_std(data, period, start_date):
    df = data.copy()
    end = df.index.get_loc(start_date)
    start = end - period
    if start >= 0:
        log_return = df['Log Return'].iloc[start:end]
        return np.std(log_return, ddof=1)
    return np.nan

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
    return np.nan

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

def main(args):
    check_input_files()
    # Load tickers
    tickers_csv = pd.read_csv(os.path.join(INPUT_DIR, 'S&P500.csv'))
    tickers = list(set(tickers_csv['Symbol']) - set(["BRK.B","BF.B","EMBC","CEG","OGN","CARR","OTIS","CTVA","MRNA","FOX","FOXA","DOW","CDAY","IR"]))
    # Download price data
    end = pd.to_datetime(args.end_date) if args.end_date else pd.to_datetime("2022-04-30")
    start = pd.to_datetime(args.start_date) if args.start_date else end - timedelta(days=5000)
    data = yf.download(tickers, start=start, end=end, group_by="ticker", interval="1d")
    # Drop columns (tickers) with any missing data (strict, as in the notebook)
    data_to_process = data.dropna(axis=1)
    print(f"[DEBUG] Data shape after dropping columns with any NA: {data_to_process.shape}")
    # Only use tickers present in the filtered data
    tickers_in_data = data_to_process.columns.get_level_values(0).unique()
    print(f"[DEBUG] Tickers in data after filtering: {list(tickers_in_data)}")
    if len(tickers_in_data) == 0:
        raise ValueError("No tickers with complete data after filtering. Check your data or relax the filtering.")
    data_processed = data_to_process
    # Download S&P 500 for comparison
    SnP = yf.download('^GSPC', start=data_processed.index[0].date(), end=datetime.today(), interval='1d')
    # Flatten SnP columns if needed
    if isinstance(SnP.columns, pd.MultiIndex):
        SnP.columns = [' '.join(col).strip() for col in SnP.columns.values]
    # Calculate SnP Log Return
    close_candidates = [col for col in SnP.columns if 'Close' in col]
    adj_close_candidates = [col for col in close_candidates if 'Adj Close' in col]
    if adj_close_candidates:
        close_col = adj_close_candidates[0]
    elif close_candidates:
        close_col = close_candidates[0]
    else:
        raise KeyError(f"No column containing 'Close' found in SnP columns: {SnP.columns}")
    SnP['SnP Log Return'] = np.log(SnP[close_col]) - np.log(SnP[close_col]).shift(1)
    # Find the correct column name for SnP Log Return
    snplog_col = [col for col in SnP.columns if 'SnP Log Return' in col][0]
    # Feature engineering
    stocks = {}
    return_lag = ['1D','3D','1W','2W','3W','1M','6W','2M','3M']
    logging.info(f"Starting feature engineering for {len(tickers_in_data)} tickers...")
    with tqdm(total=len(tickers_in_data), desc='Feature Engineering', position=0) as fe_pbar:
        for stock in tickers_in_data:
            fe_pbar.set_postfix_str(f"ticker={stock}")
            stock_df = data_processed[stock][['Close']].copy()
            stock_df['Stock'] = [stock] * stock_df.shape[0]
            stock_df['Log Return'] = np.log(stock_df['Close']) - np.log(stock_df['Close']).shift(1)
            # join with SnP return for generating other features (robust join)
            stock_df = stock_df.join(SnP[[snplog_col]], how='left')
            stock_df.rename(columns={snplog_col: 'SnP Log Return'}, inplace=True)
            stock_df['SnP Log Return_1D'] = stock_df['SnP Log Return'].shift(1)
            stock_df.dropna(inplace=True)
            stock_df['Volatility'] = [find_std(stock_df, 30, date) for date in stock_df.index]
            stock_df['Beta'] = [find_beta(stock_df, 30, date) for date in stock_df.index]
            for var in return_lag:
                name = 'Return_' + var
                stock_df[name] = stock_df['Log Return'].shift(ShiftNum(var))
            stocks[stock] = stock_df
            fe_pbar.update(1)
    if not stocks:
        raise ValueError("No valid tickers with data for feature engineering.")
    cleaned = pd.concat(stocks.values())
    cleaned.dropna(inplace=True)
    # Market cap and risk-free
    market_cap = pd.read_csv(os.path.join(INPUT_DIR, 'market_cap.csv')).astype('int64').iloc[0].to_dict()
    riskfree = pd.read_csv(os.path.join(INPUT_DIR, 'risk_free.csv'))
    # Prepare lists
    cleaned.index = pd.to_datetime(cleaned.index)
    SnP_Return = cleaned[["SnP Log Return"]]
    cleaned = cleaned.drop(columns=["SnP Log Return","Close"], axis=1)
    cleaned = cleaned.sort_values(by=["Stock","Date"])
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
    # Ensure riskfree index is datetime if possible
    if 'Date' in riskfree.columns:
        riskfree['Date'] = pd.to_datetime(riskfree['Date'], dayfirst=True)
        riskfree = riskfree.set_index('Date')
    logging.info(f"Starting backtest for {len(trade_days)} trade days...")
    with tqdm(total=len(trade_days), desc='Backtest', position=1) as pbar:
        for i, trade_day in enumerate(trade_days):
            pbar.set_postfix_str(f"trade_day={trade_day} | step=preparing")
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
            pbar.set_postfix_str(f"trade_day={trade_day} | step=fitting model")
            model = CatBoostRegressor(iterations=100, task_type="CPU", learning_rate=0.1, depth=8, l2_leaf_reg=1e-7, allow_writing_files=False, eval_metric='MAPE', random_seed=0, thread_count=-1, verbose=0)
            eval_set = Pool(eval_x, eval_y, cat_features=[0])
            catboost_train_data = Pool(data=train_x, label=train_y, cat_features=[0])
            model.fit(catboost_train_data, eval_set=eval_set, early_stopping_rounds=10)
            pbar.set_postfix_str(f"trade_day={trade_day} | step=predicting")
            preds_log_return = model.predict(trade_day_x)
            temp_df = pd.DataFrame(preds_log_return).transpose()
            temp_df.columns = stock_list
            log_return_opt_df = log_return_df[first_training_day_index:trade_day_index]
            log_return_opt_df = pd.concat([log_return_opt_df, temp_df])
            log_return_opt_df.index = date_list[first_training_day_index:trade_day_index+1]
            pbar.set_postfix_str(f"trade_day={trade_day} | step=optimizing portfolio")
            # Use map if available, otherwise fallback to applymap
            try:
                portfolio = log_return_opt_df.map(lambda x: exp(x))
            except AttributeError:
                portfolio = log_return_opt_df.applymap(lambda x: exp(x))
            cs_actual = CovarianceShrinkage(portfolio, frequency=len(log_return_opt_df))
            e_cov = cs_actual.ledoit_wolf()
            market_prices = SPY.loc[log_return_opt_df.index[0]:log_return_opt_df.index[-2]]
            # Use the actual date for riskfree lookup
            if isinstance(riskfree.index, pd.DatetimeIndex):
                if trade_day in riskfree.index:
                    riskfree_date = trade_day
                else:
                    # Use the nearest previous date
                    riskfree_date = riskfree.index[riskfree.index.get_indexer([trade_day], method='ffill')[0]]
                annual_risk_free = riskfree.loc[riskfree_date]['Price']/100
            else:
                # fallback: use the first available value
                annual_risk_free = riskfree.iloc[0]['Price']/100
            daily_risk_free = (1+annual_risk_free)**(1/252)-1
            delta = black_litterman.market_implied_risk_aversion(market_prices['Close'], risk_free_rate=daily_risk_free)
            # Filter market_cap to only include tickers in the portfolio and ensure order matches
            filtered_market_cap = {k: v for k, v in market_cap.items() if k in portfolio.columns}
            filtered_market_cap = {k: filtered_market_cap[k] for k in portfolio.columns if k in filtered_market_cap}
            prior = black_litterman.market_implied_prior_returns(filtered_market_cap, delta, e_cov)
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
            pbar.set_postfix_str(f"trade_day={trade_day} | step=done")
            pbar.update(1)
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
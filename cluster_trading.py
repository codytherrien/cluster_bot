import warnings
warnings.filterwarnings('ignore')

import sys
import alpaca_trade_api as tradeapi
from street_cred import *
import datetime
import yfinance as yf
import pandas as pd
from gen_meta_label import gen_meta_label, find_centers
import time
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import BacktestAccount
import TradeAccount

START_TIME = 13 # set to 5 for local machine 13 for cloud
END_TIME = 21 # set hour to 13 for local machine 21 for cloud


def get_stock_history(stock_list, start_date):
    """
    Generates a dataframe of hostorical stock prices
    Input: list of stock tickers
    """
    stock_history = {}
    for ticker in stock_list:
        try:
            stock_history[ticker] = yf.Ticker(ticker).history(start=start_date, interval="1d")
            stock_history[ticker]['ticker'] = ticker
        except:
            pass
    
    for ticker in stock_history:
        stock_history[ticker]['pct_change_open'] = stock_history[ticker]['Open'].pct_change()
        stock_history[ticker]['five_day_mean'] = stock_history[ticker]['pct_change_open'].rolling(5).mean()
        stock_history[ticker]['five_day_var'] = stock_history[ticker]['pct_change_open'].rolling(5).var()
        stock_history[ticker]['twenty_day_mean'] = stock_history[ticker]['pct_change_open'].rolling(20).mean()
        stock_history[ticker]['twenty_day_var'] = stock_history[ticker]['pct_change_open'].rolling(20).var()
    
    multi_stock_df = None
    for ticker in stock_history:
        if multi_stock_df is not None:
            multi_stock_df = pd.concat([multi_stock_df, stock_history[ticker]])
        else:
            multi_stock_df = stock_history[ticker]
    #multi_stock_df.set_index('Date', inplace=True)
    multi_stock_df.sort_index(inplace=True)
    
    return multi_stock_df

def sleeping_stage(api):
    market_close_flag = True
    while datetime.datetime.now().hour < START_TIME - 3:
        time.sleep(7200) # Sleeps for 2 hours
    while datetime.datetime.now().hour < START_TIME: # set to 5 for local machine 13 for cloud
        time.sleep(1200) # sleeps for 20 minutes
    while market_close_flag:
        time.sleep(60)
        try:
            clock = api.get_clock()
            if clock.is_open:
                market_close_flag = False
        except:
            api = tradeapi.REST(keys, keys_to_the_vip, paper_products)

def find_trades(model, stock_df):
    curr_time = "open"
    backtest_account = BacktestAccount.Account(100000, time=curr_time, model=model) 
    hc = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
    stock_df = stock_df[~stock_df[[f'pct_change_{curr_time}', 'five_day_mean', 'five_day_var', 'twenty_day_mean', 'twenty_day_var']\
        ].isin([np.nan, np.inf, -np.inf]).any(1)]
    stock_df['cluster'] = hc.fit_predict(stock_df[[f'pct_change_{curr_time}', 'five_day_mean']])
    df_0 = stock_df[stock_df['cluster'] == 0]
    df_0.name = 0
    df_1 = stock_df[stock_df['cluster'] == 1]
    df_1.name = 1
    df_2 = stock_df[stock_df['cluster'] == 2]
    df_2.name = 2
    df_3 = stock_df[stock_df['cluster'] == 3]
    df_3.name = 3

    tu_cent, td_cent, mru_cent, mrd_cent = find_centers(df_0, df_1, df_2, df_3, time='open')
    dfs = [df_0, df_1, df_2, df_3]

    for df in dfs:
        if df.name == tu_cent['name']:
            #account.set_pos('tu', tu_cent, df)
            backtest_account.set_outlier_pos('tu', td_cent, mru_cent, mrd_cent, df)
        elif df.name == td_cent['name']:
            #account.set_pos('td', td_cent, df)
            backtest_account.set_outlier_pos('td', tu_cent, mru_cent, mrd_cent, df)
        elif df.name == mru_cent['name']:
            #account.set_pos('mru', mru_cent, df)
            backtest_account.set_outlier_pos('mru', tu_cent, td_cent, mrd_cent, df)
        else:
            #account.set_pos('mrd', mrd_cent, df)
            backtest_account.set_outlier_pos('mrd', tu_cent, td_cent, mru_cent, df)
    
    backtest_account.make_trades(stock_df)

    positions = {}
    positions['long'] = backtest_account.long_positions.keys()
    positions['short'] = backtest_account.short_positions.keys()

    return positions

def trim_positions(trade_account, positions):
    portfolio = trade_account.api.list_positions()
    portfolio_dict = {}
    for position in portfolio:
        portfolio_dict[position.symbol] = position.qty
    
    final_positions = {}
    final_positions['long'] = []
    final_positions['short'] = []
    for short_pos in positions['short']:
        add_flag = True
        if short_pos in portfolio_dict.keys():
            if portfolio_dict[short_pos] < 0:
                add_flag = False
        if add_flag:
            final_positions['short'].append(short_pos)
        
    for long_pos in positions['long']:
        add_flag = True
        if long_pos in portfolio_dict.keys():
            if portfolio_dict[long_pos] > 0:
                add_flag = False
        if add_flag:
            final_positions['long'].append(long_pos)
    
    return final_positions

def close_positions(trade_account, positions):
    trade_account.reconnect()
    portfolio = trade_account.api.list_positions()
    for position in portfolio:
        qty = int(position.qty)
        if position.symbol in positions['long'] and qty > 0:
            continue
        elif qty > 0:
            trade_account.api.submit_order(
                symbol = position.symbol,
                qty = qty,
                side = 'sell',
                type = 'market',
                time_in_force='gtc'
            )
        elif position.symbol in positions['short'] and qty < 0:
            continue
        elif qty < 0:
            trade_account.api.submit_order(
                symbol = position.symbol,
                qty = abs(qty),
                side = 'buy',
                type = 'market',
                time_in_force='gtc'
            )

    return trade_account
        
def open_positions(trade_account, positions):
    trade_account.reconnect()
    num_positions = len(positions['long']) + len(positions['short'])
    if num_positions == 0:
        return trade_account
    cash_per_trade = trade_account.curr_cash / num_positions
    for symbol in positions['long']:
        quote = None
        while quote is None:
            try:
                quote = trade_account.api.get_bars(symbol, '1min', limit=1)[0]
            except:
                trade_account.reconnect()
                quote = None
        qty = int(cash_per_trade // quote.c)
        trade_account.api.submit_order(
                symbol = symbol,
                qty = qty,
                side = 'buy',
                type = 'market',
                time_in_force='gtc'
            )

    for symbol in positions['short']:
        quote = trade_account.api.get_bars(symbol, '1min', limit=1)[0]
        qty = int(cash_per_trade // quote.c)
        trade_account.api.submit_order(
                symbol = symbol,
                qty = qty,
                side = 'sell',
                type = 'market',
                time_in_force='gtc'
            )

    return trade_account

def trade_stage(model, stock_list):
    time.sleep(900) # sleeping to wait 15 min delay at open for yfinance data
    last_month = (datetime.date.today() - datetime.timedelta(days=50)).strftime("%Y-%m-%d")
    stock_df = get_stock_history(stock_list, last_month)
    today_df = stock_df.loc[stock_df.index[-1]]

    positions = find_trades(model, today_df)
    trade_account = TradeAccount.Account(list(stock_df['ticker'].unique()))
    trade_account = close_positions(trade_account, positions)
    positions = trim_positions(trade_account, positions)
        
    trade_account = open_positions(trade_account, positions)

    while trade_account.market_open_flag:
        try:
            minute_bars = trade_account.ws.recv()
        except:
            trade_account.stream_connect()
            trade_account.reconnect()
            minute_bars = trade_account.ws.recv()

        if datetime.datetime.now().hour == END_TIME: # set hour to 13 for local machine 21 for cloud
            trade_account.market_open_flag = False
            #if trade submit trade

        if datetime.datetime.today().day == trade_account.last_time.day:
            continue
        trade_account.last_time = datetime.datetime.today()


def main():
    # Data From: https://stockmarketmba.com/stocksinthespmidcap400.php
    mid_cap = pd.read_csv('data/mid_cap_index.csv')
    mid_cap_list = list(mid_cap['Symbol'])

    # Training stage
    last_year = (datetime.date.today() - datetime.timedelta(days=385)).strftime("%Y-%m-%d")
    print("Getting Stock data")
    mid_cap_df = get_stock_history(mid_cap_list, last_year)
    print("Training meta label model")
    meta_label_model = gen_meta_label(mid_cap_df)
    
    #################################################################
    api = tradeapi.REST(keys, keys_to_the_vip, paper_products)
    print("Starting trading stage")
    while datetime.datetime.today().weekday() != 5:
        api = tradeapi.REST(keys, keys_to_the_vip, paper_products)
        print("Starting sleep stage")
        sleeping_stage(api)
        print('Start of trading day: ', datetime.datetime.today().weekday())
        trade_stage(meta_label_model, mid_cap_list)
        print("End of day")

        time.sleep(21600) # Sleep for 6 hours
        # Training stage
        last_year = (datetime.date.today() - datetime.timedelta(days=385)).strftime("%Y-%m-%d")
        mid_cap_df = get_stock_history(mid_cap_list, last_year)
        meta_label_model = gen_meta_label(mid_cap_df)  

if __name__ == "__main__":
    main()
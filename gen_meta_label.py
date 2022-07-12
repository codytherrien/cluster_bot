import pandas as pd
import BacktestAccount
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LogisticRegression

def find_centers(df_0, df_1, df_2, df_3, time='close'):
    """
    df0-3(dataframe): dataframes of centroids
    """
    poss_td = []
    poss_tu = []
    poss_mrd = []
    poss_mru = []
    tu_cent, td_cent, mru_cent, mrd_cent = None, None, None, None
    
    cent_0 = {
        'name': 0,
        f'pct_change_{time}': df_0[f'pct_change_{time}'].mean(),
        'five_day_mean': df_0['five_day_mean'].mean(),
        'five_day_var': df_0['five_day_var'].mean(),
        'twenty_day_mean': df_0['twenty_day_mean'].mean(),
        'twenty_day_var': df_0['twenty_day_var'].mean(),
        'trend_score': df_0[f'pct_change_{time}'].mean() + df_0['five_day_mean'].mean(),
        'mean_revert_score': df_0['five_day_mean'].mean() - df_0[f'pct_change_{time}'].mean()
    }

    cent_1 = {
        'name': 1,
        f'pct_change_{time}': df_1[f'pct_change_{time}'].mean(),
        'five_day_mean': df_1['five_day_mean'].mean(),
        'five_day_var': df_1['five_day_var'].mean(),
        'twenty_day_mean': df_1['twenty_day_mean'].mean(),
        'twenty_day_var': df_1['twenty_day_var'].mean(),
        'trend_score': df_1[f'pct_change_{time}'].mean() + df_1['five_day_mean'].mean(),
        'mean_revert_score': df_1['five_day_mean'].mean() - df_1[f'pct_change_{time}'].mean() 
    }

    cent_2 = {
        'name': 2,
        f'pct_change_{time}': df_2[f'pct_change_{time}'].mean(),
        'five_day_mean': df_2['five_day_mean'].mean(),
        'five_day_var': df_2['five_day_var'].mean(),
        'twenty_day_mean': df_2['twenty_day_mean'].mean(),
        'twenty_day_var': df_2['twenty_day_var'].mean(),
        'trend_score': df_2[f'pct_change_{time}'].mean() + df_2['five_day_mean'].mean(),
        'mean_revert_score': df_2['five_day_mean'].mean() - df_2[f'pct_change_{time}'].mean() 
    }

    cent_3 = {
        'name': 3,
        f'pct_change_{time}': df_3[f'pct_change_{time}'].mean(),
        'five_day_mean': df_3['five_day_mean'].mean(),
        'five_day_var': df_3['five_day_var'].mean(),
        'twenty_day_mean': df_3['twenty_day_mean'].mean(),
        'twenty_day_var': df_3['twenty_day_var'].mean(),
        'trend_score': df_3[f'pct_change_{time}'].mean() + df_3['five_day_mean'].mean(),
        'mean_revert_score': df_3['five_day_mean'].mean() - df_3[f'pct_change_{time}'].mean() 
    }

    cents = [cent_0, cent_1, cent_2, cent_3]

    for cent in cents:
        if cent[f'pct_change_{time}'] < 0 and cent['five_day_mean'] < 0:
            poss_td.append(cent)
        elif cent[f'pct_change_{time}'] > 0 and cent['five_day_mean'] > 0:
            poss_tu.append(cent)
        elif cent[f'pct_change_{time}'] > 0 and cent['five_day_mean'] < 0:
            poss_mrd.append(cent)
        elif cent[f'pct_change_{time}'] < 0 and cent['five_day_mean'] > 0:
            poss_mru.append(cent)
    
    cents = []
    if len(poss_td) == 1:
        td_cent = poss_td[0]
    elif len(poss_td) > 1:
        poss_td = sorted(poss_td, key=lambda x: x['trend_score'])
        td_cent = poss_td[0]
        cents += poss_td[1:]
    if len(poss_tu) == 1:
        tu_cent = poss_tu[0]
    elif len(poss_tu) > 1:
        poss_tu = sorted(poss_tu, key=lambda x: x['trend_score'], reverse=True)
        tu_cent = poss_tu[0]
        cents += poss_tu[1:]
    if len(poss_mrd) == 1:
        mrd_cent = poss_mrd[0]
    elif len(poss_mrd) > 1:
        poss_mrd = sorted(poss_mrd, key=lambda x: x['mean_revert_score'])
        mrd_cent = poss_mrd[0]
        cents += poss_mrd[1:]
    if len(poss_mru) == 1:
        mru_cent = poss_mru[0]
    elif len(poss_mru) > 1:
        poss_mru = sorted(poss_mru, key=lambda x: x['mean_revert_score'], reverse=True)
        mru_cent = poss_mru[0]
        cents += poss_mru[1:]

    if len(cents) == 0:
        return tu_cent, td_cent, mru_cent, mrd_cent
    
    if td_cent is None:
        cents = sorted(cents, key=lambda x: x['trend_score'])
        td_cent = cents[0]
        if len(cents) == 1:
            return tu_cent, td_cent, mru_cent, mrd_cent
        cents = cents[1:]
    
    if tu_cent is None:
        cents = sorted(cents, key=lambda x: x['trend_score'], reverse=True)
        tu_cent = cents[0]
        if len(cents) == 1:
            return tu_cent, td_cent, mru_cent, mrd_cent
        cents = cents[1:]

    if mrd_cent is None:
        cents = sorted(cents, key=lambda x: x['mean_revert_score'])
        mrd_cent = cents[0]
        if len(cents) == 1:
            return tu_cent, td_cent, mru_cent, mrd_cent
        cents = cents[1:]
    
    if mru_cent is None:
        mru_cent = cents[0]
    
    return tu_cent, td_cent, mru_cent, mrd_cent

def backtest(multi_stock_df, starting_cash=100000, time='open'):
    """
    returns a trade history using clustering stragegy without meta-labelling
    Input: 
    multi_stock_df: dataframe of mutiple stock historical trading 
    starting_cash: starting value of trading account (float)
    time: whether trades are executed at open or close (string: "open" or "close")
    """
    print("Multistock df: ")
    print(multi_stock_df)
    dates = multi_stock_df.index.unique()
    dates = dates[22:] # removing dates with NAN values
    print(dates)
    backtest_account = BacktestAccount.Account(starting_cash, time)

    for i in range(len(dates)):
        sample_df = multi_stock_df.loc[dates[i]]
        if i > 0 and time == 'close':
            # making trades on open
            # can only trade after setting initial values
            backtest_account.make_trades(sample_df)
            #backtest_account.update_value(sample_df, sp_500_df.loc[dates[i]])

        # Updating strategy after close
        hc = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
        sample_df = sample_df[~sample_df[[f'pct_change_{time}', 'five_day_mean', 'five_day_var', 'twenty_day_mean', 'twenty_day_var']\
            ].isin([np.nan, np.inf, -np.inf]).any(1)]
        sample_df['cluster'] = hc.fit_predict(sample_df[[f'pct_change_{time}', 'five_day_mean']])
        df_0 = sample_df[sample_df['cluster'] == 0]
        df_0.name = 0
        df_1 = sample_df[sample_df['cluster'] == 1]
        df_1.name = 1
        df_2 = sample_df[sample_df['cluster'] == 2]
        df_2.name = 2
        df_3 = sample_df[sample_df['cluster'] == 3]
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

        if time == 'open':
            # making trades on open
            # can only trade after setting initial values
            backtest_account.make_trades(sample_df)
            #backtest_account.update_value(sample_df, sp_500_df.loc[dates[i]])

    backtest_account.close_positions(multi_stock_df.loc[dates[-1]])

    return backtest_account.trade_history


def gen_meta_label(multi_stock_df, starting_cash=100000, time='open'):
    """
    Generates a meta-label classifier
    Input: 
    multi_stock_df: dataframe of mutiple stock historical trading 
    starting_cash: starting value of trading account (float)
    time: whether trades are executed at open or close (string: "open" or "close")
    """
    trade_history = backtest(multi_stock_df, starting_cash, time)
    trade_history['entry_date'] = pd.to_datetime(trade_history['entry_date'])

    trade_history['side'] = trade_history['side'].apply(lambda x: 1 if x =='long' else 0)
    trade_history['type'] = trade_history['type'].apply(lambda x: 1 if x == 'trend' else 0)
    X = trade_history[[
        'side',
        'type',
        'pct_change_open', 
        'five_day_mean', 
        'five_day_var',
        'twenty_day_mean',
        'twenty_day_var' 
    ]]
    y = trade_history['win'].astype('int')
    weights = {1:1, 0:1}
    model = LogisticRegression(random_state=69, 
        class_weight=weights,
        solver='saga',
        penalty='l1'
    ).fit(X, y)

    return model
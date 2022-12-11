import pandas as pd
import numpy as np

class Account:
    def __init__(self, starting_cash, time='close', transaction_fee=0.00, model=None):
        self.starting_cash = starting_cash
        self.curr_cash = starting_cash
        self.max_cash = starting_cash
        self.model = model
        self.time = time
        self.long_positions = {}
        self.short_positions = {}
        self.transaction_fee = transaction_fee
        self.val_per_pos = 0
        self.start_date = None
        self.end_date = None
        self.max_draw_down = 0.0
        self.max_draw_down_len = 0
        self.curr_draw_down_len = 0
        self.account_values = pd.DataFrame(columns=[
            'date',
            'portfolio_value',
            'S&P_500_value'
        ])
        self.trade_history = pd.DataFrame(columns=[
            'entry_date', 
            'ticker', 
            'entry_price', 
            'side', 
            f'pct_change_{self.time}',
            'five_day_mean',
            'five_day_var',
            'twenty_day_mean',
            'twenty_day_var',
            'exit_price',
            'win'
        ])

    def _centroid_dist(self, row, cent):
        """
        row(series): row from cluster dataframe
        cent(dict): centroid of the position
        """
        dist = 0
        keys = [
            f'pct_change_{self.time}',
            'five_day_mean',
            'five_day_var',
            'twenty_day_mean',
            'twenty_day_var'
        ]
        for key in keys:
            dist += (row[key] - cent[key])**2

        return dist
    
    def _max_centroid_dist(self, row, centroids):
        """
        row(series): row from cluster dataframe
        centroids(list): list of centroids the dataframe is not in
        """
        dist = 0
        keys = [
            f'pct_change_{self.time}',
            'five_day_mean',
            'five_day_var',
            'twenty_day_mean',
            'twenty_day_var'
        ]
        for cent in centroids:
            for key in keys:
                dist += (row[key] - cent[key])**2
        
        return dist

    def set_outlier_pos(self, pos, cent_1, cent_2, cent_3, df, num_pos=3):
        """
        pos(string): type of position
        cent(dict): centroid of the position
        df(dataframe): dataframe of all tickers in the cluster
        num_pos(int): number of positions per centroid
        """
        centroids = [cent_1, cent_2, cent_3]
        df['centroid_dist'] = df.apply(self._max_centroid_dist, centroids=centroids, axis=1)
        positions = df.sort_values('centroid_dist', ascending=False).head(num_pos)['ticker']

        if pos == 'tu':
            self.tu_pos = list(positions)
        elif pos == 'td':
            self.td_pos = list(positions)
        elif pos == 'mru':
            self.mru_pos = list(positions)
        else:
            self.mrd_pos = list(positions)

    def set_pos(self, pos, cent, df, num_pos=3):
        """
        pos(string): type of position
        cent(dict): centroid of the position
        df(dataframe): dataframe of all tickers in the cluster
        num_pos(int): number of positions per centroid
        """
        df['centroid_dist'] = df.apply(self._centroid_dist, cent=cent, axis=1)
        positions = df.sort_values('centroid_dist', ascending=True).head(num_pos)['ticker']

        if pos == 'tu':
            self.tu_pos = list(positions)
        elif pos == 'td':
            self.td_pos = list(positions)
        elif pos == 'mru':
            self.mru_pos = list(positions)
        else:
            self.mrd_pos = list(positions)

    def _close_long_positions(self, new_pos, day_df):
        """
        new_pos(list): list of tickers to keep
        day_df(dataframe): df of all ticker values in s&p 500 for the current day
        """
        remove = []
        for sym in self.long_positions:
            if sym not in new_pos:
                exit_price = day_df[day_df['ticker'] == sym]['open'][0]
                entry_date, num_shares, entrance_price = self.long_positions[sym]
                self.curr_cash += exit_price*num_shares - self.transaction_fee
                self.trade_history['exit_price'].mask((self.trade_history['entry_date'] == entry_date) \
                    & (self.trade_history['ticker'] == sym), exit_price, inplace=True)
                if exit_price > entrance_price:
                    self.trade_history['win'].mask((self.trade_history['entry_date'] == entry_date) \
                        & (self.trade_history['ticker'] == sym), 1, inplace=True)
                else:
                    self.trade_history['win'].mask((self.trade_history['entry_date'] == entry_date) \
                        & (self.trade_history['ticker'] == sym), 0, inplace=True)

                self.trade_history['pct_change'].mask((self.trade_history['entry_date'] == entry_date) \
                    & (self.trade_history['ticker'] == sym), exit_price / entrance_price, inplace=True)
                
                remove.append(sym)
        
        for sym in remove:
            self.long_positions.pop(sym)

    def _close_short_positions(self, new_pos, day_df):
        """
        new_pos(list): list of tickers to keep
        day_df(dataframe): df of all ticker values in s&p 500 for the current day
        """
        remove = []
        for sym in self.short_positions:
            if sym not in new_pos:
                entry_date, num_shares, entrance_price = self.short_positions[sym]
                exit_price = day_df[day_df['ticker'] == sym]['open'][0]
                entrance_val = entrance_price*num_shares
                delta = (entrance_price-exit_price)*num_shares
                self.curr_cash += entrance_val + delta - self.transaction_fee
                self.trade_history['exit_price'].mask((self.trade_history['entry_date'] == entry_date) \
                    & (self.trade_history['ticker'] == sym), exit_price, inplace=True)
                if exit_price < entrance_price:
                    self.trade_history['win'].mask((self.trade_history['entry_date'] == entry_date) \
                        & (self.trade_history['ticker'] == sym), 1, inplace=True)
                else:
                    self.trade_history['win'].mask((self.trade_history['entry_date'] == entry_date) \
                        & (self.trade_history['ticker'] == sym), 0, inplace=True)

                self.trade_history['pct_change'].mask((self.trade_history['entry_date'] == entry_date) \
                    & (self.trade_history['ticker'] == sym), 1 - (exit_price / entrance_price), inplace=True)

                remove.append(sym)
        
        for sym in remove:
            self.short_positions.pop(sym)

    def _open_long_positions(self, long_pos, day_df, trade_type):
        """
        long_pos(list): list of tickers to open new long positions
        day_df(dataframe): df of all ticker values in s&p 500 for the current day
        """
        for sym in long_pos:
            if sym not in self.long_positions.keys():
                row = day_df[day_df['ticker'] == sym]
                share_price = row['open'][0]
                num_shares = self.val_per_pos // share_price
                self.curr_cash -= share_price*num_shares - self.transaction_fee
                self.long_positions[sym] = (row.index[0], num_shares, share_price)
                new_row = {
                    'entry_date': row.index[0], 
                    'ticker': row['ticker'][0], 
                    'entry_price': row['open'][0], 
                    'side': 'long',
                    'type': trade_type, 
                    f'pct_change_{self.time}': row[f'pct_change_{self.time}'][0],
                    'five_day_mean': row['five_day_mean'][0],
                    'five_day_var': row['five_day_var'][0],
                    'twenty_day_mean': row['twenty_day_mean'][0],
                    'twenty_day_var': row['twenty_day_var'][0],
                    'exit_price': 0.0,
                    'win': 0,
                    'pct_change': 0.0
                }
                
                self.trade_history = self.trade_history.append(new_row, ignore_index=True)


    def _open_short_positions(self, short_pos, day_df, trade_type):
        """
        short_pos(list): list of tickers to open new short positions
        day_df(dataframe): df of all ticker values in s&p 500 for the current day
        """
        for sym in short_pos:
            if sym not in self.short_positions.keys():
                row = day_df[day_df['ticker'] == sym]
                share_price = row['open'][0]
                num_shares = self.val_per_pos // share_price
                self.curr_cash -= share_price*num_shares - self.transaction_fee
                self.short_positions[sym] = (row.index[0], num_shares, share_price)
                new_row = {
                    'entry_date': row.index[0], 
                    'ticker': row['ticker'][0], 
                    'entry_price': row['open'][0], 
                    'side': 'short',
                    'type': trade_type, 
                    f'pct_change_{self.time}': row[f'pct_change_{self.time}'][0],
                    'five_day_mean': row['five_day_mean'][0],
                    'five_day_var': row['five_day_var'][0],
                    'twenty_day_mean': row['twenty_day_mean'][0],
                    'twenty_day_var': row['twenty_day_var'][0],
                    'exit_price': 0.0,
                    'win': 0,
                    'pct_change': 0.0
                }

                self.trade_history = self.trade_history.append(new_row, ignore_index=True)

    def _update_drawdown(self):
        if self.account_values['portfolio_value'].iloc[-1] > self.max_cash:
            self.curr_draw_down = 0
            self.curr_draw_down_len = 0
            self.max_cash = self.account_values['portfolio_value'].iloc[-1]
        else:
            curr_draw_down = self.max_cash - self.account_values['portfolio_value'].iloc[-1]
            self.curr_draw_down_len += 1
            if curr_draw_down > self.max_draw_down:
                self.max_draw_down = curr_draw_down
            if self.curr_draw_down_len > self.max_draw_down_len:
                self.max_draw_down_len = self.curr_draw_down_len

    def _meta_label(self, day_df):
        """
        day_df(dataframe): df of all ticker values in s&p 500 for the current day
        """
        remove = []
        for sym in self.tu_pos:
            row = day_df[day_df['ticker'] == sym]
            sample = np.array([[
                        1,
                        1,
                        row[f'pct_change_{self.time}'][0],
                        row['five_day_mean'][0],
                        row['five_day_var'][0],
                        row['twenty_day_mean'][0],
                        row['twenty_day_var'][0]
                    ]])
            pred = self.model.predict(sample)
            if pred == 0:
                remove.append(sym)
        for sym in remove:
            self.tu_pos.remove(sym)

        remove = []
        for sym in self.td_pos:
            row = day_df[day_df['ticker'] == sym]
            sample = np.array([[
                        0,
                        1,
                        row[f'pct_change_{self.time}'][0],
                        row['five_day_mean'][0],
                        row['five_day_var'][0],
                        row['twenty_day_mean'][0],
                        row['twenty_day_var'][0]
                    ]])
            pred = self.model.predict(sample)
            if pred == 0:
                remove.append(sym)
        for sym in remove:
            self.td_pos.remove(sym)

        remove = []
        for sym in self.mru_pos:
            row = day_df[day_df['ticker'] == sym]
            sample = np.array([[
                        1,
                        0,
                        row[f'pct_change_{self.time}'][0],
                        row['five_day_mean'][0],
                        row['five_day_var'][0],
                        row['twenty_day_mean'][0],
                        row['twenty_day_var'][0]
                    ]])
            pred = self.model.predict(sample)
            if pred == 0:
                remove.append(sym)
        for sym in remove:
            self.mru_pos.remove(sym)

        remove = []
        for sym in self.mrd_pos:
            row = day_df[day_df['ticker'] == sym]
            sample = np.array([[
                        0,
                        0,
                        row[f'pct_change_{self.time}'][0],
                        row['five_day_mean'][0],
                        row['five_day_var'][0],
                        row['twenty_day_mean'][0],
                        row['twenty_day_var'][0]
                    ]])
            pred = self.model.predict(sample)
            if pred == 0:
                remove.append(sym)
        for sym in remove:
            self.mrd_pos.remove(sym)

        return 

    def make_trades(self, day_df):
        """
        day_df(dataframe): df of all ticker values in s&p 500 for the current day
        """
        if self.start_date == None:
            self.start_date = day_df.index[0]
        if self.model is None:
            num_positions = len(self.tu_pos) + len(self.td_pos) + len(self.mru_pos) + len(self.mrd_pos)
        else:
            self._meta_label(day_df)
            num_positions = len(self.tu_pos) + len(self.td_pos) + len(self.mru_pos) + len(self.mrd_pos)
        
        long_pos = self.tu_pos + self.mru_pos
        short_pos = self.td_pos + self.mrd_pos

        if len(self.long_positions) > 0:
            self._close_long_positions(long_pos, day_df)
        if len(self.short_positions) > 0:
            self._close_short_positions(short_pos, day_df)

        self.val_per_pos = self.curr_cash / (num_positions - len(self.long_positions) - len(self.short_positions))
        
        self._open_long_positions(self.tu_pos, day_df, 'trend')
        self._open_long_positions(self.mru_pos, day_df, 'mean revert')
        self._open_short_positions(self.td_pos, day_df, 'trend')
        self._open_short_positions(self.mrd_pos, day_df, 'mean revert')

    def _update_portfolio_value(self, day_df):
        """
        day_df(dataframe): df of all ticker values in s&p 500 for the current day
        """
        value = 0
        for pos in self.long_positions:
            _, num_shares, _ = self.long_positions[pos]
            value += num_shares * day_df[day_df['ticker'] == pos]['adjclose'][0]
        
        for pos in self.short_positions:
            _, num_shares, entrance_price = self.short_positions[pos]
            value += num_shares * entrance_price
            value += num_shares * (entrance_price - day_df[day_df['ticker'] == pos]['adjclose'][0])

        self._update_drawdown()

        return value + self.curr_cash

    def update_value(self, day_df, spy):
        """
        day_df(dataframe): df of all ticker values in s&p 500 for the current day
        spy(series): series containing daily values for SPY ticker
        """
        if len(self.account_values) == 0:
            self.spy_start = spy['adjclose']
            spy_value = self.starting_cash
            portfolio_value = spy_value
        else:
            spy_value = spy['adjclose'] / self.spy_start * self.starting_cash
            portfolio_value = self._update_portfolio_value(day_df)
        
        new_row = {
            'date': spy.name,
            'portfolio_value': portfolio_value,
            'S&P_500_value': spy_value 
        }

        self.account_values = self.account_values.append(new_row, ignore_index=True)

    def close_positions(self, day_df):
        """
        day_df(dataframe): df of all ticker values in s&p 500 for the current day
        """
        self._close_long_positions([], day_df)
        self._close_short_positions([], day_df)

        self.end_date = day_df.index[0]

    # Calculate Net Profit
    def get_profit(self):
        print("Profit: ", self.curr_cash - self.starting_cash)

    # Calculate Annualized Returns 
    def get_annualized_returns(self, print_val=True):
        """
        print_val(bool): bool if the annualized return should be printed 
        """
        num_years = (self.end_date - self.start_date)
        num_days = num_years.days / 365
        returns = (self.curr_cash/self.starting_cash)**(1/num_days) - 1
        if print_val:
            print("Annualized Returns: ", returns)
        return returns


    # Calculate Winrate
    def get_winrate(self):
        num_trades = len(self.trade_history)
        winning_trades = len(self.trade_history[self.trade_history['win'] == 1])
        print("Winrate: ", winning_trades / num_trades)
    
    # Calculate % gain on winning trades
    def get_mean_winning_gain(self):
        winning_trades = self.trade_history[self.trade_history['win'] == 1]
        print('Percent Gain on Winning Trades: ', winning_trades['pct_change'].mean())

    # Calculate % loss on losing trades
    def get_mean_losing_loss(self):
        losing_trades = self.trade_history[self.trade_history['win'] == 0]
        print('Percent Loss on Losing Trades: ', losing_trades['pct_change'].mean())

    # Calculate sharpe ratio
    def get_sharpe_ratio(self, risk_free_rate):
        """
        risk_free_rate(float): expected annualized return of risk free investment
        """
        annualized_returns = self.get_annualized_returns(False)
        num_years = (self.end_date - self.start_date) / 365
        portfolio_std = self.trade_history['pct_change'].std()

        print('Sharpe Ratio: ', (annualized_returns - risk_free_rate) / portfolio_std)

    # Calculate max draw down and max draw down length
    def get_drawdown(self):
        print('Max drawdown: ', self.max_draw_down)
        print('Max drawdown length: ', self.max_draw_down_len)

    # Cacluate alpha
    def get_alpha(self, risk_free_rate):
        """
        risk_free_rate(float): expected annualized return of risk free investment
        """
        corr = np.corrcoef(list(self.account_values['portfolio_value']), list(self.account_values['S&P_500_value']))[0,1]
        portfolio_std = self.account_values['portfolio_value'].pct_change().std()
        sp_500_std = self.account_values['S&P_500_value'].pct_change().std()
        beta = corr * (portfolio_std / sp_500_std)

        num_years = (self.end_date - self.start_date)
        num_days = num_years.days / 365
        sp_close_value = self.account_values['S&P_500_value'].iloc[-1]
        sp_returns = (sp_close_value / self.starting_cash)**(1/num_days) - 1

        alpha = self.get_annualized_returns(False) - risk_free_rate - beta*(sp_returns - risk_free_rate)

        print("Alpha: ", alpha)

    # Create plot of returns over time with index
    def gen_account_value_plot(self):
        self.account_values.set_index('date').plot()
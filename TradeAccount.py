import time
import alpaca_trade_api as tradeapi
from street_cred import *
from websocket import create_connection
import json
import ast
import datetime

class Account:
    def __init__(self, stock_list, stream_connect=True):
        self.reconnect_flag = True
        self.market_open_flag = True
        self.stream_flag = True
        self.account_update_flag = True
        self.ws = None
        self.api = None
        self.account = None
        self.curr_cash = None
        self.buying_power = None
        self.stock_list = stock_list
        self.reconnect()
        if stream_connect:
            self.stream_connect()
        self.morning_flag = True
        self.last_time = datetime.datetime.strptime('2021-01-01', '%Y-%m-%d')

    def reconnect(self):
        self.reconnect_flag = True
        while self.reconnect_flag:
            try:
                self.api = tradeapi.REST(keys, keys_to_the_vip, paper_products)
                self.account = self.api.get_account()
                self.curr_cash = float(self.account.cash)
                self.buying_power = float(self.account.buying_power)
                print('Account Opening Value: ' + str(self.curr_cash))
                print('Account Opening Buying Power ', self.buying_power)
                print(f"{datetime.datetime.now().hour}:{datetime.datetime.now().minute}")
                self.reconnect_flag = False
            except:
                time.sleep(10)
    
    def on_open(self):
        print('Websocket Opened')
        auth_data = {
            "action": "auth",
            "key": keys,
            "secret": keys_to_the_vip
        }

        self.ws.send(json.dumps(auth_data))

    def stream_connect(self):
        self.stream_flag = True
        while self.stream_flag:
            #try:
            #self.ws = create_connection("wss://stream.data.alpaca.markets/v1beta1/crypto")
            self.ws = create_connection("wss://stream.data.alpaca.markets/v2/iex")
            #self.ws = create_connection("wss://stream.data.alpaca.markets/v2/sip")
            print(self.ws.recv())
            self.on_open()
            message = self.ws.recv()
            message = ast.literal_eval(message)
            print(message)
            if message[0]["msg"] == "authenticated":
                trades_message = {
                    "action": "subscribe",
                    "bars": self.stock_list
                }
                self.ws.send(json.dumps(trades_message))
                message = self.ws.recv()
                message = ast.literal_eval(message)
                print(message)
                self.stream_flag = False
            #except:
            #    print("Reattempting authentication")

    def close_positions(self):
        self.reconnect_flag = True
        while self.reconnect_flag:
            try:
                orders = self.api.list_orders(status='open')
                positions = self.api.list_positions()

                if orders or positions:
                    if positions:
                        print(positions)
                
                if orders:
                    print("Cancelling open orders:")
                    print([o.id for o in orders])
                    result = [self.api.cancel_order(o.id) for o in orders]
                    print(result)
                
                closed = []
                for p in positions:
                    side = 'sell'
                    if int(p.qty) < 0:
                        p.qty = abs(int(p.qty))
                        side = 'buy'
                    closed.append(
                        self.api.submit_order(
                            symbol = p.symbol,
                            qty = p.qty,
                            side = side,
                            type = 'market',
                            time_in_force = 'day'
                        )
                    )
                
                if closed:
                    print("Submitted Orders: ", closed)
                
                time.sleep(30)
                for o in closed:
                    status = self.api.get_order(o.id)
                    if status.status == 'rejected':
                        print("ORDER FAILED!!")
                
                self.reconnect_flag = False
            
            except:
                time.sleep(30)
                self.api = tradeapi.REST(keys, keys_to_the_vip, paper_products)

        print("Trading Day Over")
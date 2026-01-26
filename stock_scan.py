import pandas as pd
import numpy as np
from datetime import date, timedelta
import yfinance as yf
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import lineStyles

def yfdownload(equity,start_date,today):
    # Download new data
    df_new = yf.download(equity, start=start_date, end=today, group_by='ticker')

    # --- Flatten MultiIndex if exists ---
    if isinstance(df_new.columns, pd.MultiIndex):
        df_new = df_new[equity].copy()

    # Reset index and add Ticker column
    df_new.reset_index(inplace=True)
    df_new["Ticker"] = equity

    # Ensure consistent column order
    df_new = df_new[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker']]

    return df_new

def plot_result(df,equity):
    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax1.plot(df.index, df['Close'], label='Close Price', color='blue')
    ax1.plot(df.index, df['200d'], label='200-day MA', linestyle='--', color='grey')
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Close Price", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Buy/sell markers
    buy_signals = df[df['position'] == 1]
    sell_signals = df[df['position'] == -1]

    ax1.scatter(buy_signals.index, buy_signals['Close'],
                marker='^', color='green', s=100, label='Buy', alpha=0.8, edgecolors='black')
    ax1.scatter(sell_signals.index, sell_signals['Close'],
                marker='v', color='red', s=100, label='Sell', alpha=0.8, edgecolors='black')

    # Secondary axis: moving_avg_rolling_mean
    ax2 = ax1.twinx()
    ax2.plot(df.index, df['moving_avg_rolling_mean'], color='orange', linestyle='--', label='Moving Avg Rolling Mean')
    #ax2.plot(df.index, df['rolling_mean_ln_r'], color='orange', linestyle='--', label='rolling_mean_ln_r')
    ax2.set_ylabel("Double smooth ln (r)", color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    # Add horizontal line at rolling variance = 0.00075
    ax2.axhline(y=0.0008, color='green', linestyle='--', label='Threshold 0.008')

    # Third axis: cumulative profit
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.1))
    ax3.plot(df.index, df['cum_profit'], color='purple', linewidth=2, label='Cumulative Profit')
    ax3.set_ylabel("Cumulative Profit", color='purple')
    ax3.tick_params(axis='y', labelcolor='purple')

    # Title and combined legend
    fig.suptitle(f"Trading Simulation {equity}: Buy/Sell Rebound Strategy", fontsize=16)
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    lines_3, labels_3 = ax3.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2 + lines_3, labels_1 + labels_2 + labels_3, loc='upper left')

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

def stock_scan(equity_list):
    results = []
    for i, equity in enumerate(equity_list):
        investment = 100000
        # --- 1. Load data ---
        try:
            stock_data_df = pd.read_csv("stock_data.csv")

            df = stock_data_df[stock_data_df["Ticker"] == equity].copy()
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.normalize()
            print(df)
            df = stock_data_df[stock_data_df["Ticker"] == equity].copy()

            df['return'] = df['Close'].pct_change()
            df['ln_r'] = np.log(1 + df['return'])

            # --- Rolling statistics ---
            df['200d'] = df['Close'].rolling(window=200).mean()
            df['rolling_mean_ln_r'] = df['ln_r'].rolling(window=30).mean()
            df['moving_avg_rolling_mean'] = df['rolling_mean_ln_r'].rolling(window=60).mean()

            df = df[-250:]  # filter data to last 250 days

            if df.iloc[-1]['Close'] >= df.iloc[-1]['200d']:
                try:

                    # --- Initialize trading columns ---
                    df['position'] = 0      # 1 = buy, -1 = sell, 0 = hold
                    df['cum_return'] = 0.0  # return since last buy
                    df['profit'] = 0.0      # profit of each completed trade
                    df['cum_profit'] = 0.0  # cumulative profit

                    # --- 2. Trading loop: buy first, sell alternately ---
                    in_position = False
                    buy_price = None
                    buy_index = None
                    sell_index = None
                    days_after_sell = 0
                    number_of_stock = 0
                    investment = 100000  # starting capital
                    df['position'] = 0
                    df['profit'] = 0.0

                    for i, index in enumerate(df.index):
                        if i < 3:
                            continue

                        price = df.loc[index, 'Close']

                        # detect rebound for buy
                        ma_min = df['moving_avg_rolling_mean'].min()
                        ma_max = df['moving_avg_rolling_mean'].max()
                        ma_threshold = ma_min + 0.5 * (ma_max - ma_min)

                        ma_prev3 = df.iloc[i - 3]['moving_avg_rolling_mean']
                        ma_prev2 = df.iloc[i - 2]['moving_avg_rolling_mean']
                        ma_prev1 = df.iloc[i - 1]['moving_avg_rolling_mean']
                        ma_curr = df.iloc[i]['moving_avg_rolling_mean']
                        ma_rebound_signal = (ma_curr <= ma_threshold and
                                             ma_curr > ma_prev1 and ma_prev1 > ma_prev2)

                        ma_reverse_signal = ma_curr < ma_prev1 and ma_prev1 < ma_prev2

                        days_after_sell += 1

                        # --- BUY ---
                        if not in_position:
                            if pd.notna(ma_curr) and ma_rebound_signal and (price / df.loc[index, '200d']) >= 1 and days_after_sell > 15:
                                df.loc[index, 'position'] = 1
                                buy_price = price
                                buy_index = index
                                in_position = True
                                number_of_stock = investment / buy_price
                                print('buy price:', buy_price)
                                print('number of stock:', number_of_stock)

                        # --- SELL ---
                        else:  # only if we are in a position
                            cum_return = (price - buy_price) / buy_price

                            # sell conditions
                            sell_condition = (
                                (cum_return >= 0.1 and ma_reverse_signal) or
                                (cum_return <= -0.5 and (price / df.loc[index, '200d']) < 1)
                            )

                            if sell_condition:
                                sell_price = price
                                investment = sell_price * number_of_stock
                                df.loc[index, 'position'] = -1
                                df.loc[index, 'profit'] = (sell_price - buy_price)*number_of_stock
                                sell_index = index
                                number_of_stock = 0
                                days_after_sell = 0
                                # reset state
                                in_position = False

                                print('sell date: ', sell_index )
                                print('sell price: ', sell_price)
                                print('buy price: ', buy_price)
                                print('profit: ', (sell_price - buy_price)*number_of_stock)

                    # --- 3. Compute cumulative profit ---
                    df['cum_profit'] = df['profit'].cumsum()
                    print(df[df['profit'] != 0][['Close', 'profit', 'cum_profit']])
                    print('initial investment = 100,000 THB')
                    print(f'Current Portfolio Value: {investment}, {((investment - 100000) / 100000):.2%},\n'
                          f'price * stocks: {buy_price} x {number_of_stock}')

                    row_result = {
                        'equity': equity,
                        'price': price,
                        'in position': in_position,
                        'number_of_stock': number_of_stock,
                        'buy_price': buy_price if in_position else 0.0,  # store the buy price if in position
                        'cum_profit': df['cum_profit'].iloc[-1] if not df['cum_profit'].empty else 0.0,
                        'current_portfolio_value': investment,
                        'profit_margin': ((investment - 100000) / 100000)
                    }
                    results.append(row_result)

                    # 4. plot result
                    #plot_result(df, equity)
                    #plt.show(block=False)
                except:pass
        except: pass

    result = pd.DataFrame(results)
    print(result)

    # Optional: compute unrealized profit for stocks still in position
    def compute_unrealized(row):
        if row['in position'] and row['number_of_stock'] > 0:
            return row['number_of_stock'] * (row['price'] - row['buy_price'])
        return 0.0

    result['unrealized_profit'] = result.apply(compute_unrealized, axis=1)

    # Total profit per stock: realized + unrealized
    result['total_profit'] = result['cum_profit'] + result['unrealized_profit']

    # Pretty-print summary
    summary = result[['equity', 'in position','buy_price' ,'price', 'number_of_stock', 'cum_profit', 'unrealized_profit', 'total_profit', 'current_portfolio_value', 'profit_margin']].copy()
    summary['profit_margin'] = (summary['profit_margin']*100).round(2).astype(str) + '%'
    summary[['price','cum_profit','buy_price','unrealized_profit','total_profit','current_portfolio_value']] = summary[['price','cum_profit','buy_price','unrealized_profit','total_profit','current_portfolio_value']].round(2)

    print("\n=== Stock Summary ===")
    print(summary.to_string(index=False))

    # Portfolio totals
    total_invested = 100000 * len(result)
    total_realized_value = result['current_portfolio_value'].sum()
    total_realized_profit = total_realized_value - total_invested
    total_margin = total_realized_profit / total_invested

    total_value = result['total_profit'].sum()
    total_value_margin = total_value / total_invested

    print("\n=== Portfolio Totals ===")
    print(f"Total Invested: {total_invested:,.2f} THB")
    print(f"Total Current Value: {total_realized_value:,.2f} THB")
    print(f"Total Realized Profit: {total_realized_profit:,.2f} THB")
    print(f"Realized Profit %: {total_margin:.2%}")
    print(f'Total_Profit Value: {total_value:,.2f}')
    print(f'Total_Profit Margin: {total_value_margin:,.2%}')

    return summary, total_value_margin

#while plt.get_fignums():
#    plt.pause(0.1)

if __name__ == "__main__":
    results = []
    stocks = ['PTTEP.BK']
    summary, margin = stock_scan(stocks)

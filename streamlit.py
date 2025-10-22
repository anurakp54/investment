import streamlit as st
import altair as alt
import pandas as pd
from datetime import date, timedelta
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import os

plt.style.use('fivethirtyeight')

st.set_page_config(layout='wide')

def yfdownload(equity,start_date):
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


def compute_unrealized(row):
    if row['in position'] and row['number_of_stock'] > 0:
        return row['number_of_stock'] * (row['price'] - row['buy_price'])
    return 0.0

def plot_result_altair(df, equity):
    df = df.reset_index()  # ensure 'Date' is a column
    df = df[df['Ticker'] == equity]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['Date', 'Close', '200d', 'moving_avg_rolling_mean', 'cum_profit'])

    base = alt.Chart(df).encode(
        x=alt.X('Date:T', axis=alt.Axis(labelAngle=-90, title='Date'))
    )

    # --- Close Price line ---
    close_line = base.mark_line(color='blue').encode(
        y=alt.Y(
            'Close:Q',
            title='Close Price',
            scale=alt.Scale(domain=[df['Close'].min(), df['Close'].max()])
        ),
        tooltip=['Date', 'Close']
    )

    # --- 200-day MA line ---
    ma_line = base.mark_line(color='grey', strokeDash=[5, 5]).encode(
        y=alt.Y('200d:Q', title='200-day MA'),
        tooltip=['Date', '200d']
    )

    # --- Buy/Sell markers ---
    buy = alt.Chart(df[df['position'] == 1]).mark_point(
        shape='triangle-up', color='green', size=100
    ).encode(
        x='Date:T', y='Close:Q', tooltip=['Date', 'Close']
    )

    sell = alt.Chart(df[df['position'] == -1]).mark_point(
        shape='triangle-down', color='red', size=100
    ).encode(
        x='Date:T', y='Close:Q', tooltip=['Date', 'Close']
    )

    # --- Combine price layers ---
    price_layer = alt.layer(close_line, ma_line, buy, sell)

    # --- Cumulative profit (2nd y-axis) ---
    cum_line = base.mark_line(color='purple', strokeDash=[2, 2]).encode(
        y=alt.Y(
            'cum_profit:Q',
            axis=alt.Axis(
                title='Cumulative Profit',
                titlePadding=40,  # move away from chart
                labelPadding=10
            )
        ),
        tooltip=['Date', 'cum_profit']
    )

    # --- Rolling mean ln(r) (on right axis) ---
    rolling_line = base.mark_line(color='orange').encode(
        y=alt.Y(
            'moving_avg_rolling_mean:Q',
            axis=alt.Axis(
                title=None,
                orient='right',
                labels=False,
                ticks=False
            )
        ),
        tooltip=['Date', 'moving_avg_rolling_mean']
    )

    # --- Combine all and separate y-scales ---
    final_chart = alt.layer(
        price_layer, cum_line, rolling_line
    ).resolve_scale(
        y='independent'
    ).properties(
        width=900,
        height=500,
        title=f"{equity}: Close Price, Cumulative Profit & Rolling Mean ln(r)",
        padding={"right": 100}  # <-- extra margin for secondary axis
    )

    # --- Display in Streamlit ---
    st.altair_chart(final_chart, use_container_width=True)


stocks = [
    'PTT.BK','PTTEP.BK', 'AOT.BK','KTB.BK','BBL.BK', 'SCB.BK', 'KBANK.BK', 'ADVANC.BK', 'DELTA.BK', 'AP.BK',
    'CRC.BK','CPALL.BK','GULF.BK','HMPRO.BK','CK.BK','STECON.BK','BDMS.BK','BH.BK','AAV.BK','AEONTS.BK','AMATA.BK',
    'AURA.BK','AWC.BK','BA.BK','BAM.BK','BANPU.BK','BCH.BK','BCP.BK','BCPG.BK','BDMS.BK','BEM.BK',
    'BGRIM.BK','BJC.BK','BLA.BK','BTG.BK','CBG.BK','CCET.BK','CENTEL.BK','CHG.BK','COM7.BK','CPF.BK','CPN.BK',
    'CRC.BK','DOHOME.BK','EA.BK','EGCO.BK','ERW.BK','GLOBAL.BK','GPSC.BK','GULF.BK','GUNKUL.BK','HANA.BK',
    'HMPRO.BK','ICHI.BK','IRPC.BK','ITC.BK','IVL.BK','JAS.BK','JMART.BK','JMT.BK','JTS.BK','KCE.BK',
    'KKP.BK','LH.BK','M.BK','MBK.BK','MEGA.BK','MINT.BK','MOSHI.BK','MTC.BK','OR.BK','OSP.BK','PLANB.BK','PR9.BK',
    'PRM.BK','PTTGC.BK','QH.BK','RATCH.BK','RCL.BK','SAWAD.BK','SCC.BK','SCGP.BK','SIRI.BK','SISB.BK','SJWD.BK',
    'SPALI.BK','SPRC.BK','STA.BK','STGT.BK','TASCO.BK','TCAP.BK','TFG.BK','TIDLOR.BK','TISCO.BK','TLI.BK','TOA.BK',
    'TOP.BK','TRUE.BK','TTB.BK','TU.BK','VGI.BK','WHA.BK','WHAUP.BK'
]

# INPUT STOCK and CHECK Data if avaliable and uptodate
with st.sidebar:
    stock_manual = st.text_input('[AAPL, GOOG, MSFT, "AMZN"]')
    stock_manual_list = [s.strip() for s in stock_manual.split(',') if s.strip()]
    period = st.slider('Period', 0, 720, value=250, step=10)

    # --- Create checkboxes in sidebar (each with a unique key) ---
    equity_list = []
    for i, stock in enumerate(stocks):
        if st.sidebar.checkbox(stock, key=f"chk_{i}"):
            equity_list.append(stock)

    if stock_manual_list:
        equity_list.extend(stock_manual_list)

# --- Display selected stocks ---
st.write("### Portfolio Management:")
st.markdown(
    """
    <small>
    **Disclaimer:** This application is provided for informational purposes only.  
    It does not constitute financial advice, and no guarantee of profit or protection against loss is implied.  
    Investment decisions are made at the sole discretion and risk of the investor.  
    The developers and providers of this application assume no liability for any financial losses or damages that may result from its use.
    </small>
    """,
    unsafe_allow_html=True
)
if equity_list:
    st.write(equity_list)
else:
    st.info("No equities selected yet.")


today = date.today()
start_date = str(date.today() - timedelta(days=2000))
results = []
print(equity_list)

for i, equity in enumerate(equity_list):
    print(f'equity: {equity}')

    investment = 100000
    # --- 1. Load data ---
    stock_data_df = pd.read_csv("stock_data.csv")
    df = stock_data_df[stock_data_df["Ticker"] == equity].copy()

    if len(df) == 0:
        df_new = yfdownload(equity,start_date)
        # --- Append or replace in main DataFrame ---
        stock_data_df = pd.concat([stock_data_df, df_new], ignore_index=True)
        stock_data_df = stock_data_df.drop_duplicates()
        stock_data_df.to_csv("stock_data.csv", index=False)
        stock_data_df = pd.read_csv("stock_data.csv")
        df = stock_data_df[stock_data_df["Ticker"] == equity].copy()

    df['Date'] = pd.to_datetime(df['Date'])
    last_data_date = df["Date"].iloc[-1]
    today = pd.Timestamp.today().normalize()
    print(df.tail(5))

    if (today - last_data_date).days > 2:
        # Download new data
        df_new = yfdownload(equity,start_date)
        # --- Append or replace in main DataFrame ---
        stock_data_df = pd.concat([stock_data_df, df_new], ignore_index=True)
        stock_data_df = stock_data_df.drop_duplicates()
        stock_data_df.to_csv("stock_data.csv", index=False)
        stock_data_df = pd.read_csv("stock_data.csv")
        df = stock_data_df[stock_data_df["Ticker"] == equity].copy()

    df = df[-1200:]

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
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
            plot_result_altair(df, equity)
            plt.show(block=False)
        except:pass

    else: pass

result = pd.DataFrame(results)
st.write(result)

# Optional: compute unrealized profit for stocks still in position
def compute_unrealized(row):
    if row['in position'] and row['number_of_stock'] > 0:
        return row['number_of_stock'] * (row['price'] - row['buy_price'])
    return 0.0

if result.empty:
    st.warning("No Trading")

else:
    # clean up data

    result = result.replace([np.inf, -np.inf], np.nan).fillna(0)
    result['unrealized_profit'] = result.apply(compute_unrealized, axis=1)

    # Total profit per stock: realized + unrealized
    result['total_profit'] = result['cum_profit'] + result['unrealized_profit']

    # Pretty-print summary

    summary = result[['equity', 'in position','buy_price' ,'price', 'number_of_stock', 'cum_profit', 'unrealized_profit', 'total_profit', 'current_portfolio_value', 'profit_margin']].copy()
    summary = summary.replace([np.inf, -np.inf], np.nan).fillna(0)
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

    st.write("\n=== Portfolio Totals ===")
    st.write(f"Total Invested: {total_invested:,.2f} THB")
    st.write(f"Total Current Value: {total_realized_value:,.2f} THB")
    st.write(f"Total Realized Profit: {total_realized_profit:,.2f} THB")
    st.write(f"Realized Profit %: {total_margin:.2%}")
    st.write(f'Total_Profit Value: {total_value:,.2f}')
    st.write(f'Total_Profit Margin: {total_value_margin:,.2%}')

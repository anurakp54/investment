import pandas as pd
from datetime import date, timedelta
import yfinance as yf
import numpy as np
import os

def get_stock_data_csv(csv_path, ticker, period):
    today = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    default_start = pd.Timestamp.today().normalize() - pd.Timedelta(days=period)
    start_max = pd.Timestamp.today().normalize() - pd.Timedelta(days=2000)
    # --- 1. Read CSV safely ---
    if os.path.exists(csv_path):
        stock_data_df = pd.read_csv(csv_path)
        print(f'Loaded data from CSV at: {csv_path}')
    else:
        stock_data_df = pd.DataFrame()

    # --- FORCE schema ---
    if not stock_data_df.empty:
        if 'Date' not in stock_data_df.columns:
            stock_data_df = stock_data_df.reset_index()

        stock_data_df['Date'] = pd.to_datetime(stock_data_df['Date'], errors='coerce')

    # --- 2. Ensure required columns ---
    if not stock_data_df.empty:
        stock_data_df['Date'] = pd.to_datetime(stock_data_df['Date'], errors='coerce')

    # --- 3. Filter existing data for ticker ---
    existing_df = stock_data_df[stock_data_df['Ticker'] == ticker] \
        if not stock_data_df.empty else pd.DataFrame()

    # --- 4. Determine start date ---
    if existing_df.empty:
        start_date = default_start
        last_date = start_date
        last_date = pd.to_datetime(last_date)
    else:
        last_date = existing_df['Date'].max()
        last_date = pd.to_datetime(last_date)
        start_date = last_date + timedelta(days=1)

    missing_recent = today - last_date > pd.Timedelta(days=2)
    missing_early = (
            existing_df.empty or
            'Date' not in existing_df.columns or
            existing_df['Date'].min() > default_start
    )
    print(f'default start date {default_start}')
    print(f'missing_early: {missing_early}')
    # --- 5. Download new data ---
    if missing_recent or missing_early:
        df_new = yfdownload(ticker, start_max, today)

        if not df_new.empty:
            df_new = df_new.reset_index()  # <-- FIX 1
    else:
        print('data is updated!')
        df_new = pd.DataFrame(columns=stock_data_df.columns)  # <-- FIX 2

    # --- 6. Prepare new data ---
    if not df_new.empty:
        df_new['Date'] = pd.to_datetime(df_new['Date'])
        df_new['Ticker'] = ticker

    # --- 6. Prepare new data ---
    df_new = df_new.copy()
    df_new['Date'] = pd.to_datetime(df_new['Date'])
    df_new['Ticker'] = ticker

    # --- 7. Append & de-duplicate ---
    if stock_data_df.empty:
        updated_df = df_new
    else:
        updated_df = pd.concat([stock_data_df, df_new], ignore_index=True)

    updated_df['Date'] = updated_df['Date'].dt.normalize()
    updated_df = (
        updated_df
        .drop_duplicates(subset=['Ticker', 'Date'])
        .sort_values(['Ticker', 'Date'])
    )

    # --- 8. Write back to CSV ---
    updated_df.to_csv(csv_path, index=False)

    return df_new, updated_df

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

if __name__ == "__main__":

    # initialize with expected columns
    stock_data_df = pd.DataFrame(columns=["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"])
    df = pd.DataFrame(columns=["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"])
    stocks = ['PTT.BK', 'PTTEP.BK']
    try:
        stock_data_df = pd.read_csv("stock_data.csv")
    except:
        stock_data_df = pd.DataFrame(columns=df.columns)

    period = 500

    for i, equity in enumerate(stocks):
        print(f'equity: {equity}')
        today = date.today() - timedelta(days = 1)
        start_date = str(date.today() - timedelta(days=period))
        df,stock_data_df = get_stock_data_csv('stock_data.csv',equity,period)
        print(df)
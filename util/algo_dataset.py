import numpy as np
import pandas as pd
import indicators
import config
import util
import sys
from pathlib import Path

def get_algo_dataset(choose_set_num: int):
    """ Returns df_list, date_range, trend_list, stocks
    """
    # Do not change run_set order. The order is hardcoded into below code
    run_set = ['portfolio1', 'portfolio2']
    choose_set = run_set[choose_set_num]
    df_list = []
    date_range = []
    trend_list = []
    stocks = []

    if choose_set == run_set[0]:
        # Ganti dengan 11 stock codes portfolio 1
        stocks =['AUD', 'CAD', 'CNY', 'EMLC', 'EUR',
                'GBP', 'JPY', 'KRW', 'MYR', 'NZD',
                'PLN', 'SGD', 'USD']
        for stock in stocks:
            df=pd.read_csv(f'data/rl/portfolio1/{stock}.csv', parse_dates=['Date'])
            df = df[df['Close'] > 0].reset_index(drop=True)
            df['returns'] = indicators.day_gain(df, 'Close').dropna()
            df_list.append(df)

        start = '1/1/2021'
        end = '31/12/2023'
        date_range = remove_uncommon_dates(df_list)
        trend_list = util.get_trend_list(stocks, df_list, start=start, end=end)

    elif choose_set == run_set[1]:
        # Ganti dengan 19 stock codes portfolio 2
        stocks =['STOCK1', 'STOCK2', 'STOCK3', 'STOCK4', 'STOCK5',
                'STOCK6', 'STOCK7', 'STOCK8', 'STOCK9', 'STOCK10',
                'STOCK11', 'STOCK12', 'STOCK13', 'STOCK14', 'STOCK15',
                'STOCK16', 'STOCK17', 'STOCK18', 'STOCK19', 'STOCK20', 'STOCK21', ]
        for stock in stocks:
            df=pd.read_csv(f'data/rl/portfolio2/{stock}.csv', parse_dates=['Date'])
            df = df[df['Close'] > 0].reset_index(drop=True)
            df['returns'] = indicators.day_gain(df, 'Close').dropna()
            df_list.append(df)

        start = '1/1/2021'
        end = '31/12/2023'
        date_range = remove_uncommon_dates(df_list)
        trend_list = util.get_trend_list(stocks, df_list, start=start, end=end)
        
    return df_list, date_range, trend_list, stocks

def validate_portfolio_data(df_list, stocks):
    """Validate portfolio data consistency"""
    try:
        # Check for same date range
        start_dates = [df['Date'].min() for df in df_list]
        end_dates = [df['Date'].max() for df in df_list]
        
        if len(set(start_dates)) > 1 or len(set(end_dates)) > 1:
            print("Warning: Different date ranges detected")
            print("Start dates:", start_dates)
            print("End dates:", end_dates)
            
        # Check for missing values
        for i, df in enumerate(df_list):
            missing = df.isnull().sum()
            if missing.sum() > 0:
                print(f"Warning: Missing values in {stocks[i]}:")
                print(missing[missing > 0])
                
        # Check for negative or zero prices
        for i, df in enumerate(df_list):
            invalid_prices = df[df['Close'] <= 0].shape[0]
            if invalid_prices > 0:
                print(f"Warning: {invalid_prices} invalid prices in {stocks[i]}")
                
        # Check for data frequency
        for i, df in enumerate(df_list):
            freq = df['Date'].diff().mode()[0]
            if freq.days != 1:
                print(f"Warning: Non-daily frequency detected in {stocks[i]}")
                
        return True
    except Exception as e:
        print(f"Error validating data: {e}")
        return False

def remove_uncommon_dates(df_list):
    date_range = []
    for date in df_list[0]['Date']:
        empty = 0
        for df in df_list:
            temp_df = df[df['Date'] == date]
            if temp_df.empty:
                empty +=1
        if empty == 0:
            date_range.append(date)
    return date_range

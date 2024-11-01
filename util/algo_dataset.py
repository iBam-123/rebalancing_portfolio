import numpy as np
import pandas as pd
import indicators
import config
import util
import sys
from pathlib import Path

def get_algo_dataset(choose_set_num: int):
    run_set = ['portfolio1', 'portfolio2']
    choose_set = run_set[choose_set_num]
    config = PORTFOLIO_CONFIG[portfolio_name]
    stocks = config['assets']
    df_list = []
    
    # Load data untuk setiap aset
    for stock in stocks:
        df = pd.read_csv(f'data/rl/{portfolio_name}/{stock}.csv', parse_dates=['Date'])
        df = df[df['Close'] > 0].reset_index(drop=True)
        df['returns'] = indicators.day_gain(df, 'Close').dropna()
        df_list.append(df)

    # Get common dates
    date_range = remove_uncommon_dates(df_list)
    
    # Filter date range sesuai config
    start_date = pd.to_datetime(config['start_date'])
    end_date = pd.to_datetime(config['end_date'])
    date_range = [d for d in date_range if start_date <= d <= end_date]
    
    # Get trend list
    trend_list = util.get_trend_list(stocks, df_list, 
                                   start=config['start_date'],
                                   end=config['end_date'])
    
    return df_list, date_range, trend_list, stocks

def setup_portfolio_paths(portfolio_name: str, approach: str, predict: bool):
    """Setup necessary directories for saving results"""
    base_path = f'data/rl/{portfolio_name}'
    
    if approach == 'gradual':
        subfolder = 'non_lagged' if predict else 'lagged'
    else:  # full_swing
        subfolder = 'fs_non_lagged' if predict else 'fs_lagged'
        
    paths = {
        'base': base_path,
        'subfolder': f'{base_path}/{subfolder}',
        'lstm': f'{base_path}/lstm',
        'data': f'{base_path}/data'
    }
    
    # Create directories if they don't exist
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
        
    return paths

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

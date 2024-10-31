import indicators
import util
import config
import numpy as np
import pandas as pd
import math
import sys

def get_portfolio_comp(current_comp: list, df_list: list, base_rates: list, date: pd.Timestamp, 
    cvar_period=[10,10,10], mc_period=[10,10,10], sp_period=[10,10,10], c1=[0,0,0], c2=[0,0,0]):
    """Modified to handle variable number of assets"""
    num_assets = len(current_comp)
    
    # Extend periods and coefficients if needed
    if len(cvar_period) != num_assets:
        cvar_period = [10] * num_assets
    if len(mc_period) != num_assets:
        mc_period = [10] * num_assets
    if len(sp_period) != num_assets:
        sp_period = [10] * num_assets
    if len(c1) != num_assets:
        c1 = [0] * num_assets
    if len(c2) != num_assets:
        c2 = [0] * num_assets
    
    stocks = df_list
    sum_base_rates = sum(base_rates)
    sum_factors = []
    
    for i, stock in enumerate(stocks):
        t = stocks[i].index[stocks[i]['Date'] == date].tolist()
        if t == [] or t[0] == 0 or t[0] == 1:
            return current_comp
        else:
            t = t[0]
            
        sp = f_sp(stock, t, int(sp_period[i]))
        mr = f_mr(stock, t, int(cvar_period[i]), c2=c2[i])
        mc = f_mc(stock, t, int(mc_period[i]), c1[i])
        
        sum_factor = util.modified_tanh(mr * mc) * sp
        sum_factors.append(sum_factor)
        
    norm_sum_factors = []
    adjustable_comp = 1 - sum_base_rates
    
    for i in range(num_assets):
        norm_sum_factors.append(adjustable_comp*util.softmax(sum_factors)[i])
    
    return [base_rates[i] + norm_sum_factors[i] for i in range(num_assets)]

def f_mr(df: pd.DataFrame, t: int, period=10, alpha=0.95, c2=0, price_col='Close'):
	return abs(cvar_percent(df, t, period, alpha, price_col) + c2)

def f_mc(df, t: int, period=10, c1=0):
	mc_df = indicators.macd_line(df, center=False) - indicators.macd_signal(df, center=False)
	if t-period+1 < 0:
		norm_array_df = mc_df.iloc[0:t+1]
	else:
		norm_array_df = mc_df.iloc[t-period+1:t+1]
	norm_mc = util.z_score_normalization(norm_array_df.values[-1], norm_array_df.values.tolist())
	# print("Market Condition = {:.2f}".format(norm_mc))
	return norm_mc + c1

def f_sp(df:pd.DataFrame, t: int, period=10):
    # mean_ema = ema_df.iloc[t-period+1:t+1].mean()
    # print("Swing Potential = {:.2f}".format(mean_ema))
    # t must be bigger than 2 to normalize
    ema_df = indicators.exponential_moving_avg(df, window_size=period, center=False)
    if t-period+2 < 0:
        return util.z_score_normalization(ema_df.iloc[t] ,ema_df.iloc[0:t+1])
    else:
        return util.z_score_normalization(ema_df.iloc[t] ,ema_df.iloc[t-period+1:t+1])
    # return mean_ema

def value_at_risk_percent(df: pd.DataFrame, t: int, period=10, alpha=0.95, price_col='Close'):
    """Calculates the Value at Risk (VaR) of time period
    """
    # t must be bigger than 2 to evaluate percentile
    if t-period+2 < 0:
        var_df = df.iloc[1:t+1]
    else:
        var_df = df.iloc[t-period+1:t+1]
    if price_col=='Close':
        returns_list = var_df['returns'].dropna().values
    else:
        returns_list = indicators.day_gain(var_df, price_col).dropna().values
    if len(returns_list)==0:
        return 0, []
    return np.percentile(returns_list, 100*(1-alpha)), returns_list

def cvar_percent(df: pd.DataFrame, t: int, period=10, alpha=0.95, price_col='Close'):
    """Conditional VaR (CVaR)
    """
    var_percent, returns_list = value_at_risk_percent(df, t, period=period, alpha=alpha, price_col=price_col)
    if len(returns_list)==0:
        return 0
    lower_than_threshold_returns = [returns for returns in returns_list if returns < var_percent]
    return np.nanmean(lower_than_threshold_returns)

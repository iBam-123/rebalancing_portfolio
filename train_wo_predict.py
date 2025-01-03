import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from copy import deepcopy
from tqdm import tqdm
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import datetime
import sys
import math
import util
import config
import indicators
import argparse
import os
from util.algo_dataset import get_algo_dataset
import random
from collections import deque
import time
from datetime import datetime, timedelta

#tf.compat.v1.disable_eager_execution()
#arg_parser = argparse.ArgumentParser()
#arg_parser.add_argument("--choose_set_num", required=True)
#arg_parser.add_argument("--stocks", required=True)
#arg_parser.add_argument("--path", required=True)
#arg_parser.add_argument("--load", action='store_true')
#arg_parser.add_argument("--full_swing", action='store_true')
def parse_arguments():
    parser = argparse.ArgumentParser(description="Train RL model for portfolio optimization")
    parser.add_argument("--portfolio", required=True, choices=['portfolio1', 'portfolio2'], help="Select portfolio to train")
    parser.add_argument("--approach", required=True, choices=['gradual', 'full_swing'], help="Select rebalancing approach")
    parser.add_argument("--predict", action="store_true", help="Use LSTM prediction")
    parser.add_argument("--load", action="store_true", help="Load existing model")
    return parser.parse_args()

# Setelah fungsi parse_arguments()
def get_save_paths(portfolio: str, approach: str, predict: bool = False):
    """Get paths for saving results"""
    base_path = f'data/rl/{portfolio}'
    
    if approach == 'gradual':
        subfolder = 'non_lagged' if predict else 'lagged'
    else:  # full_swing
        subfolder = 'fs_non_lagged' if predict else 'fs_lagged'
    
    folder_path = f'{base_path}/{subfolder}'
    os.makedirs(folder_path, exist_ok=True)
    
    return {
        'model': f'{folder_path}/model.weights.h5',
        'daily_nav': f'{folder_path}/daily_nav.csv',
        'passive_nav': f'{folder_path}/passive_daily_nav.csv'
    }

# Di bagian main atau training loop, setelah inisialisasi model
args = parse_arguments()
save_paths = get_save_paths(args.portfolio, args.approach, args.predict)

# Load dataset
df_list, date_range, trend_list, stocks = util.get_algo_dataset(args.portfolio)

# Sekarang kita bisa menggunakan trend_list
max_epLength = len(trend_list) - 1

def save_training_results(episode, date_range, portfolio_values, 
                         passive_values, stocks, save_paths, save_interval=100):
    """
    Save model and results at specified intervals
    
    Args:
        episode (int): Current episode number
        date_range (list): List of dates
        portfolio_values (list): Portfolio values over time
        passive_values (list): Passive strategy values over time
        stocks (list): List of stock names
        save_paths (dict): Dictionary containing paths for saving files
        save_interval (int): Interval for saving results (default: 100)
    """
    if episode % save_interval == 0:
        # Save daily NAV
        nav_df = pd.DataFrame({
            'Date': date_range,
            'Net': portfolio_values
        })
        nav_df.to_csv(save_paths['daily_nav'], index=False)
        
        # Save passive NAV
        passive_df = pd.DataFrame({
            'Date': date_range,
            **{stock: values for stock, values in zip(stocks, passive_values)}
        })
        passive_df.to_csv(save_paths['passive_nav'], index=False)
        
        print(f"Results saved at episode {episode}")

def get_epsilon(total_steps):
    return max(config.final_exploration,
              config.initial_exploration - 
              (config.exploration_decay * total_steps))
#parameter
risk_level = 1

save_rl_data = True
save_passive = True
save_algo_data = True

update_freq = 10
start_e = 1
end_e = 0.1
annealing_steps = 5000
pre_train_steps = 84400  # 160000
max_epLength = len(trend_list) - 1
h_size = 100
tau = 0.0005

# Set the rate of random action decrease.
e_rate = start_e
step_drop = (start_e - end_e) / annealing_steps

class Qnetwork(tf.keras.Model):
    def __init__(self, hidden_size):
        super(Qnetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(
            hidden_size, 
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.01)
        )
        self.dropout = tf.keras.layers.Dropout(0.2)  # Tambahkan dropout
        self.dense2 = tf.keras.layers.Dense(
            config.num_actions,
            kernel_regularizer=tf.keras.regularizers.l2(0.01)
        )
        
    def call(self, inputs, training=False):  # Tambahkan parameter training
        x = self.dense1(inputs)
        x = self.dropout(x, training=training)  # Gunakan dropout dalam call
        return self.dense2(x)

    def save(self, filepath):
        self.save_weights(filepath)
    
    def load_weights(self, filepath):
        super(Qnetwork, self).load_weights(filepath)

    def get_q_values(self, state):
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        if len(state.shape) == 1:
            state = tf.expand_dims(state, 0)
        return self(state)

    def train_step(self, state, target):
        """Single training step"""
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        target = tf.convert_to_tensor(target, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            q_values, sum_regularization = self(state)
            loss = tf.reduce_sum(tf.square(target - q_values) + sum_regularization)
        
        # Get gradients and apply them
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return loss

def norm_state(state):
    """Normalize state and ensure correct shape"""
    if isinstance(state, tuple):
        state = np.array(state)
    state = np.array(state, dtype=np.float32)
    if len(state.shape) == 1:
        state = np.reshape(state, [1, -1])
    return state


def process_action(action, portfolio_composition):
    MIN_ALLOCATION = 0.05  # 5% minimum per aset
    MAX_ALLOCATION = 0.30
    num_assets = len(portfolio_composition)
    new_portfolio_composition = deepcopy(portfolio_composition)

    
    if args.approach == 'full_swing':
        ####################### Full switch ##############################
        if action == 0:  # High risk focus
            high_risk_portion = num_assets // 3
            for i in range(num_assets):
                if i < high_risk_portion:
                    new_portfolio_composition[i] = 0.6 / high_risk_portion
                elif i < 2 * high_risk_portion:
                    new_portfolio_composition[i] = 0.3 / high_risk_portion
                else:
                    new_portfolio_composition[i] = 0.1 / (num_assets - 2 * high_risk_portion)
                    
        elif action == 1:  # Medium risk focus
            med_risk_portion = num_assets // 3
            for i in range(num_assets):
                if i < med_risk_portion:
                    new_portfolio_composition[i] = 0.3 / med_risk_portion
                elif i < 2 * med_risk_portion:
                    new_portfolio_composition[i] = 0.5 / med_risk_portion
                else:
                    new_portfolio_composition[i] = 0.2 / (num_assets - 2 * med_risk_portion)
                    
        elif action == 2:  # Balanced
            balanced_weight = 1.0 / num_assets
            new_portfolio_composition = [balanced_weight] * num_assets
            
        elif action == 3:  # Conservative
            low_risk_portion = num_assets // 3
            for i in range(num_assets):
                if i < low_risk_portion:
                    new_portfolio_composition[i] = 0.1 / low_risk_portion
                elif i < 2 * low_risk_portion:
                    new_portfolio_composition[i] = 0.2 / low_risk_portion
                else:
                    new_portfolio_composition[i] = 0.7 / (num_assets - 2 * low_risk_portion)
    else:
        ###################### Gradual ##############################
        step = 0.1
        if action == 0:  # Increase high risk
            high_risk_end = num_assets // 3
            for i in range(high_risk_end):
                if new_portfolio_composition[i] + step <= 0.8:
                    new_portfolio_composition[i] += step
                    # Decrease others proportionally
                    decrease = step / (num_assets - 1)
                    for j in range(num_assets):
                        if j != i:
                            new_portfolio_composition[j] = max(0.1, new_portfolio_composition[j] - decrease)
                            
        elif action == 1:  # Increase medium risk
            med_start = num_assets // 3
            med_end = 2 * num_assets // 3
            for i in range(med_start, med_end):
                if new_portfolio_composition[i] + step <= 0.8:
                    new_portfolio_composition[i] += step
                    decrease = step / (num_assets - 1)
                    for j in range(num_assets):
                        if j != i:
                            new_portfolio_composition[j] = max(0.1, new_portfolio_composition[j] - decrease)
                            
        elif action == 2:  # Move towards balanced
            target = 1.0 / num_assets
            for i in range(num_assets):
                if new_portfolio_composition[i] > target:
                    new_portfolio_composition[i] -= step
                else:
                    new_portfolio_composition[i] += step
                    
        elif action == 3:  # Increase low risk
            low_risk_start = 2 * num_assets // 3
            for i in range(low_risk_start, num_assets):
                if new_portfolio_composition[i] + step <= 0.8:
                    new_portfolio_composition[i] += step
                    decrease = step / (num_assets - 1)
                    for j in range(num_assets):
                        if j != i:
                            new_portfolio_composition[j] = max(0.1, new_portfolio_composition[j] - decrease)
    
    # Normalize to ensure sum = 1
    total = sum(new_portfolio_composition)
    new_portfolio_composition = [x/total for x in new_portfolio_composition]
    return new_portfolio_composition

#pertanyakan
def adjust_portfolio_gradual(portfolio, start_idx, end_idx, step):
    """Helper function untuk menyesuaikan alokasi portfolio secara gradual"""
    new_portfolio = portfolio.copy()
    num_assets = len(portfolio)
    
    # Hitung jumlah asset yang akan ditingkatkan
    num_increased_assets = end_idx - start_idx
    
    # Hitung jumlah asset yang akan dikurangi
    num_decreased_assets = num_assets - num_increased_assets
    
    if num_decreased_assets <= 0:
        return new_portfolio
    
    # Hitung pengurangan per asset untuk asset yang dikurangi
    decrease_per_asset = (step * num_increased_assets) / num_decreased_assets
    
    # Sesuaikan alokasi
    for i in range(num_assets):
        if start_idx <= i < end_idx:
            # Tingkatkan alokasi untuk asset dalam range
            new_value = new_portfolio[i] + step
            # Pastikan tidak melebihi batas maksimum (80%)
            new_portfolio[i] = min(0.8, new_value)
        else:
            # Kurangi alokasi untuk asset lainnya
            new_value = new_portfolio[i] - decrease_per_asset
            # Pastikan tidak kurang dari batas minimum (10%)
            new_portfolio[i] = max(0.1, new_value)
    
    # Normalisasi untuk memastikan total = 1
    total = sum(new_portfolio)
    if total != 1.0:
        new_portfolio = [x/total for x in new_portfolio]
    
    # Validasi final
    for i in range(num_assets):
        # Pastikan semua alokasi dalam range yang diizinkan
        if new_portfolio[i] < 0.1:
            new_portfolio[i] = 0.1
        elif new_portfolio[i] > 0.8:
            new_portfolio[i] = 0.8
    
    # Normalisasi final
    total = sum(new_portfolio)
    new_portfolio = [x/total for x in new_portfolio]
    
    return new_portfolio


def get_next_state(current_index, trend_list, date_range, df_list):
    """
    Get next state based on current index and trend list
    
    Args:
        current_index: Current position in trend list
        trend_list: List of trend dates
        date_range: List of all available dates
        df_list: List of dataframes containing price data
    """
    if current_index + 1 >= len(trend_list):
        raise ValueError("Current index is out of range for trend_list")
    
    # Convert trend date to same format as date_range
    date = pd.to_datetime(trend_list[current_index + 1])
    
    # Find matching date in date_range
    date_indices = []
    for i, cur_date in enumerate(date_range):
        if pd.to_datetime(cur_date).date() == date.date():
            date_indices.append(i)
    
    if not date_indices:
        # If exact date not found, find nearest available date
        nearest_date = min(date_range, key=lambda x: abs(pd.to_datetime(x) - date))
        date_indices = [date_range.index(nearest_date)]
        print(f"Warning: Exact date {date} not found, using nearest date {nearest_date}")
    
    date_idx = date_indices[0]
    state_ = ()

    for i in range(2):  # Assuming you're using 2 assets
        price_list = []
        if date_idx - config.price_period >= 0:
            price_dates = date_range[date_idx - config.price_period:date_idx]
        else:
            price_dates = date_range[0:date_idx]
            price_list.extend([0] * (config.price_period - date_idx))

        for date in price_dates:
            try:
                price = df_list[i][df_list[i]['Date'] == date]['Close'].values[0]
                price_list.append(price)
            except IndexError:
                price_list.append(price_list[-1] if price_list else 0)

        df = pd.DataFrame({'Close': price_list})

        df['EMA'] = indicators.exponential_moving_avg(df, window_size=6, center=False)
        df['MACD_Line'] = indicators.macd_line(df, ema1_window_size=3, ema2_window_size=6, center=False)
        df['MACD_Signal'] = indicators.macd_signal(df, window_size=6, ema1_window_size=3, ema2_window_size=6, center=False)

        ema_price = util.z_score_normalization(df.iloc[-1]['EMA'], df['EMA'].tolist())
        macd = df['MACD_Line'].iloc[-1] - df['MACD_Signal'].iloc[-1]
        macd = util.scale(macd, (df['MACD_Line'] - df['MACD_Signal']).tolist())

        if np.isnan(ema_price) or np.isnan(macd):
            print(f'Warning: nan encountered: ema = {ema_price}, macd = {macd}')
            ema_price = 0 if np.isnan(ema_price) else ema_price
            macd = 0 if np.isnan(macd) else macd

        state_ += (ema_price, macd)

    if current_index == -1:
        last_date_delta = 0
    else:
        last_date_delta = (trend_list[current_index + 1] - trend_list[current_index]).days

    state_ += (last_date_delta,)
    return state_

#tanyakan
def get_predicted_indicator_df(df, price_list, scaler, model):
    """
    Generate predicted indicators using LSTM model.
    
    Args:
    df (pd.DataFrame): DataFrame containing historical price data
    price_list (list): List of historical prices
    scaler (sklearn.preprocessing.MinMaxScaler): Scaler used for normalization
    model (tensorflow.keras.Model): Trained LSTM model for prediction
    
    Returns:
    pd.DataFrame: DataFrame with predicted indicators
    """
    # Scale the input data
    scaled_df = scaler.fit_transform(df)
    
    # Prepare data for LSTM prediction
    temp_data1 = []
    # predict 3 days after
    for j in range(3):
        temp_data2 = []
        # lookback period of 7
        for k in range(7):
            temp_data3 = scaled_df[-j - k - 1].tolist()[1:]  # Exclude the first column (Close price)
            temp_data2.append(temp_data3)
        temp_data1.append(temp_data2)
    temp_data1 = np.array(temp_data1)

    # Make predictions using the LSTM model
    prediction = model.predict(temp_data1)
    
    # Combine predictions with other features
    temp_data = np.concatenate((prediction, temp_data1[:, 0, :]), axis=1)
    
    # Inverse transform to get actual price predictions
    pred_close = scaler.inverse_transform(temp_data)[:, 0]
    
    # Append predicted prices to the price list
    for close in pred_close:
        price_list.append(close)
    
    # Create a new DataFrame with the extended price list
    df = pd.DataFrame({'Close': price_list})
    
    # Calculate technical indicators
    df['EMA'] = indicators.exponential_moving_avg(df, window_size=6, center=True)
    df['MACD_Line'] = indicators.macd_line(df, ema1_window_size=3, ema2_window_size=6, center=True)
    df['MACD_Signal'] = indicators.macd_signal(df, window_size=6, ema1_window_size=3, ema2_window_size=6, center=True)
    
    # Return the DataFrame, excluding the first 3 and last 3 rows
    # This is done to remove potential edge effects from the centered indicators
    return df.iloc[3:-3]

def normalize_rewards(rewards):
    rewards = np.array(rewards)
    return np.clip(rewards, -1, 1)
    # Membuat mean = 0 dan std = 1
    return (rewards - rewards.mean()) / (rewards.std() + 1e-8)

def get_reward(asset_list, action, current_index, trend_list, date_range, portfolio_composition, df_list):
    reward_period = 15
    commission_rate = 1.0/800
    
    date = trend_list[current_index]
    
    # Cari tanggal terdekat dalam date_range
    closest_date = min(date_range, key=lambda x: abs(x - date))
    date_idx = date_range.index(closest_date)
    
    if date_idx + reward_period < len(date_range):
        reward_date = date_range[date_idx + reward_period]
    else:
        reward_date = date_range[-1]

    passive_asset_sum, _ = get_reward_asset_sum(asset_list, portfolio_composition, 
                                              closest_date, reward_date, commission_rate, df_list)

    if args.approach == 'full_swing':
        changed_composition_rates = process_action(action, portfolio_composition)
        changed_asset_sum, _ = get_reward_asset_sum(asset_list, changed_composition_rates, 
                                                  closest_date, reward_date, commission_rate, df_list)

        if changed_asset_sum - passive_asset_sum == 0 or passive_asset_sum == 0:
            new_asset_list, nav_reward = calc_actions_nav(asset_list, portfolio_composition, 
                                                        trend_list, current_index, date_range, df_list)
            return 0, changed_composition_rates, new_asset_list

        new_asset_list, nav_reward = calc_actions_nav(asset_list, portfolio_composition, 
                                                    trend_list, current_index, date_range, df_list)

    else:
        changed_asset_sum = 0
        changed_composition_rates = portfolio_composition
        changed_asset_list = deepcopy(asset_list)
        
        for i, cur_date in enumerate(date_range[date_idx:date_idx + reward_period]):
            if cur_date != trend_list[current_index + 1] and i < 3:
                changed_composition_rates = process_action(action, portfolio_composition)
                changed_asset_sum, changed_asset_list = get_reward_asset_sum(
                    changed_asset_list,
                    changed_composition_rates, 
                    cur_date,
                    date_range[date_idx + i + 1],
                    commission_rate,
                    df_list
                )
            else:
                date = cur_date
                break
                
        if changed_asset_sum - passive_asset_sum == 0 or passive_asset_sum == 0:
            new_asset_list, nav_reward = calc_actions_nav(asset_list, portfolio_composition, 
                                                        trend_list, current_index, date_range, df_list)
            return 0, changed_composition_rates, new_asset_list
            
        new_asset_list, nav_reward = changed_asset_list, changed_asset_sum

    trend_list_len = len(trend_list)
    time_scaling_factor = 0.5 * (trend_list_len - current_index) / trend_list_len + 0.5
    raw_reward = (changed_asset_sum - passive_asset_sum) / passive_asset_sum * time_scaling_factor
    nav_reward_scaled = nav_reward / 1000
    
    # Normalize rewards
    rewards_to_normalize = [raw_reward, nav_reward_scaled]
    normalized_rewards = normalize_rewards(rewards_to_normalize)
    
    # Combine normalized rewards
    final_reward = normalized_rewards[0] + normalized_rewards[1]
    
    # Clip reward untuk mencegah nilai ekstrim
    final_reward = np.clip(final_reward, -1, 1)
    
    return final_reward, changed_composition_rates, new_asset_list


def get_reward_asset_sum(asset_list, composition, start_date, end_date, commission_rate, df_list):
    new_asset_list = deepcopy(asset_list)
    
    # Update nilai asset berdasarkan pergerakan harga
    for i, df in enumerate(df_list):
        start_price = df[df['Date'] == start_date]['Close'].values[0]
        end_price = df[df['Date'] == end_date]['Close'].values[0]
        price_change = end_price / start_price
        new_asset_list[i] *= price_change
    
    total_assets = sum(new_asset_list)
    
    transaction_costs = 0

    # Hitung perubahan portfolio berdasarkan komposisi baru
    for i in range(len(new_asset_list)):
        target_amount = composition[i] * total_assets
        amount_change = abs(target_amount - new_asset_list[i])
        transaction_costs += amount_change * commission_rate
    
    # Return yang sudah memperhitungkan biaya transaksi
    return sum(new_asset_list) - transaction_costs, new_asset_list


def calc_actions_nav(asset_list, portfolio_composition, trend_list, index, date_range, df_list, 
                    final_nav=False, commission_rate=1.0/800):
    new_asset_list = deepcopy(asset_list)
    
    if final_nav:
        for j in range(len(df_list)):
            previous_close_price = df_list[j][df_list[j]['Date'] == trend_list[-1]]['Close'].values[0]
            current_close_price = df_list[j][df_list[j]['Date'] == date_range[-1]]['Close'].values[0]
            new_asset_list[j] *= current_close_price / previous_close_price
        return new_asset_list, sum(new_asset_list)

    prev_date = date_range[0] if index == 9 else trend_list[index - 1]
    date = trend_list[index]


    # Update asset values by passive market movement
    for j in range(len(df_list)):
        previous_close_price = df_list[j][df_list[j]['Date'] == prev_date]['Close'].values[0]
        current_close_price = df_list[j][df_list[j]['Date'] == date]['Close'].values[0]
        new_asset_list[j] = new_asset_list[j] * current_close_price / previous_close_price
    
    total_assets = sum(new_asset_list)

    # Update asset values by portfolio adjustment
    for j in range(len(df_list)):
        amount_change = portfolio_composition[j] * total_assets - new_asset_list[j]
        if amount_change <= 0:
            new_asset_list[j] = new_asset_list[j] + amount_change
        else:
            new_asset_list[j] = new_asset_list[j] + amount_change * (1 - commission_rate) ** 2

    return new_asset_list, sum(new_asset_list)

def get_initial_state(df_list, trend_list, date_range):
    """
    Mengambil state awal berdasarkan data yang ada.
    
    Args:
        df_list: Daftar DataFrame yang berisi data harga untuk setiap aset.
        trend_list: Daftar tanggal yang menunjukkan tren beli/jual.
        date_range: Rentang tanggal yang digunakan untuk pelatihan.
    
    Returns:
        state: State awal untuk model.
    """
    # Misalnya, kita ambil state dari tanggal pertama di trend_list
    initial_date = trend_list[0]
    initial_index = date_range.index(initial_date)
    
    # Ambil data harga dari DataFrame untuk tanggal ini
    state = []
    
    for df in df_list:
        price = df[df['Date'] == initial_date]['Close'].values[0]
        state.append(price)  # Atau indikator lain yang Anda gunakan
    
    # Jika Anda juga ingin menambahkan informasi lain (seperti indikator teknis), lakukan di sini.
    
    return np.array(state)

def train_model(df_list, date_range, trend_list, stocks, args):
    # Initialize networks
    #max_next_q = tf.zeros([config.batch_size, 1])
    #config.batch_size = 32
    mainQN = Qnetwork(config.hidden_layer_size)
    targetQN = Qnetwork(config.hidden_layer_size)
    mainQN.optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
    
    num_assets = len(stocks)
    portfolio_composition = [1.0/num_assets] * num_assets  # Equal weight initialization
    asset_list = [10000/num_assets] * num_assets  # Initial investment split equally
    
    # Initialize replay buffer
    epsilon = config.initial_exploration
    replay_buffer = deque(maxlen=config.buffer_size)
    early_stopping_patience = 50
    
    # Training metrics
    episode_rewards = []
    losses = []
    patience = 50
    best_reward = float('-inf')
    episodes_without_improvement = 0
    current_index = 0
    with tqdm(total=config.num_episodes, desc="Training Progress") as pbar:
      for episode in range(config.num_episodes):
          state = get_next_state(current_index-1, trend_list, date_range, df_list)
          episode_reward = 0
          current_index = 9
          episode_reward = 0

          while current_index < len(trend_list)-1:  # Replace with your episode termination condition
              # Get action
              if np.random.rand() < epsilon:
                  action = np.random.randint(0, config.num_actions)
              else:
                  q_values = mainQN.get_q_values(norm_state(state))
                  action = np.argmax(q_values[0])

              # Take action and get next state and reward
              next_state, reward, done, new_portfolio_composition, new_asset_list = step(
                  action,
                  asset_list,
                  current_index,
                  trend_list,
                  date_range,
                  portfolio_composition,
                  df_list
              )

              replay_buffer.append((
                  state,
                  action,
                  reward,
                  next_state,
                  done
              ))
              
              asset_list = new_asset_list
              episode_reward += reward

              # Training
              if len(replay_buffer) >= config.batch_size:
                  # Sample minibatch
                  minibatch = random.sample(replay_buffer, config.batch_size)
                  
                  # Prepare batch data
                  states = np.array([norm_state(trans[0]) for trans in minibatch])
                  actions = np.array([trans[1] for trans in minibatch])
                  rewards = np.array([trans[2] for trans in minibatch])
                  next_states = np.array([norm_state(trans[3]) for trans in minibatch])
                  dones = np.array([trans[4] for trans in minibatch])

                  states = tf.convert_to_tensor(states, dtype=tf.float32)
                  next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
                  rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
                  actions = tf.convert_to_tensor(actions, dtype=tf.int32)
                  dones = tf.convert_to_tensor(dones, dtype=tf.float32)
                  
                  rewards = tf.reshape(rewards, (config.batch_size, 1))
                  dones = tf.reshape(dones, (config.batch_size, 1))

                  rewards = normalize_rewards(rewards)
                  
                  # Calculate target Q-values
                  next_q_values = targetQN.get_q_values(next_states)
                  max_next_q = tf.reduce_max(next_q_values, axis=1)
                  max_next_q = tf.reshape(max_next_q, [-1, 1])  # Reshape to column vector
                  
                  # Calculate targets
                  targets = rewards[:, None] + config.gamma * (1 - dones[:, None]) * max_next_q

                  with tf.GradientTape() as tape:
                      # Get current Q-values
                      current_q_values = mainQN(states)
                    
                      # Get Q-values for actions taken
                      action_masks = tf.one_hot(actions, config.num_actions)
                      predicted_q_values = tf.reduce_sum(current_q_values * action_masks, axis=1, keepdims=True)
                      
                      # Calculate loss
                      loss = tf.reduce_mean(tf.square(targets - predicted_q_values))

                  # Calculate and apply gradients
                  gradients = tape.gradient(loss, mainQN.trainable_variables)
                  mainQN.optimizer.apply_gradients(zip(gradients, mainQN.trainable_variables))
                  
                  losses.append(loss.numpy())

                  
              state = next_state
              portfolio_composition = new_portfolio_composition
              asset_list = new_asset_list
              episode_reward += reward
              current_index += 1
              
              # Decay epsilon
              if epsilon > config.final_exploration:
                  epsilon -= config.exploration_decay
              
              if done:
                  break
                  # Update target network periodically
          if episode % config.target_update_freq == 0:
              targetQN.set_weights(mainQN.get_weights())
          
          episode_rewards.append(episode_reward)
          
          pbar.update(1)
          pbar.set_postfix({
                'Episode Reward': f'{episode_reward:.2f}',
                'Epsilon': f'{epsilon:.2f}'
            })
          
          # Logging
          if episode % 100 == 0:
              avg_reward = np.mean(episode_rewards[-100:])
              avg_loss = np.mean(losses[-100:]) if losses else 0
              print(f"Episode: {episode}")
              print(f"Average Reward: {avg_reward:.2f}")
              print(f"Average Reward (Normalized): {np.mean(normalize_rewards(episode_rewards[-100:])):.2f}")
              print(f"Average Loss: {avg_loss:.4f}")
              print(f"Max Reward: {max(episode_rewards):.2f}")
              print(f"Min Reward: {min(episode_rewards):.2f}")
              print(f"Portfolio Value: {sum(asset_list):.2f}")
              print("------------------------")

          avg_reward = np.mean(episode_rewards[-100:])

          if avg_reward > best_reward:
              best_reward = avg_reward
              episodes_without_improvement = 0
          else:
              episodes_without_improvement += 1
            
          if episodes_without_improvement >= patience:
              print(f"Early stopping at episode {episode}")
              break
    
    # Save results
    save_paths = get_save_paths(args.portfolio, args.approach, args.predict)
    
    # Save model
    mainQN.save(save_paths['model'])
    
    # Save daily NAV
    df_nav = pd.DataFrame({'Net': asset_list}, index=date_range[:len(asset_list)])
    df_nav = df_nav.reindex(date_range, method='ffill')
    df_nav.reset_index(inplace=True)
    df_nav.columns = ['Date', 'Net']
    
    df_nav.to_csv(save_paths['daily_nav'], index=False)

    #save passive NAV
    df_passive = calculate_passive_nav(date_range, df_list)
    df_passive.to_csv(save_paths['passive_nav'], index=False)
    
    return asset_list

def step(action, asset_list, current_index, trend_list, date_range, portfolio_composition, df_list):
    """
    Mengambil tindakan dan mengembalikan state baru, reward, dan status selesai.
    
    Args:
        action: Tindakan yang diambil.
        current_index: Indeks saat ini dalam trend_list.
        trend_list: Daftar tanggal yang menunjukkan tren beli/jual.
        date_range: Rentang tanggal yang digunakan untuk pelatihan.
        portfolio_composition: Komposisi portofolio saat ini.
        df_list: Daftar DataFrame yang berisi data harga untuk setiap aset.
    
    Returns:
        next_state: State baru setelah mengambil tindakan.
        reward: Reward yang diterima dari tindakan.
        done: Boolean yang menunjukkan apakah episode telah selesai.
    """
    new_portfolio_composition = process_action(action, portfolio_composition)
    
    reward, new_portfolio_composition, new_asset_list = get_reward(
        asset_list, 
        action, 
        current_index, 
        trend_list,
        date_range, 
        portfolio_composition, 
        df_list
    )
    
    next_state = get_next_state(current_index, trend_list, date_range, df_list)
    
    done = current_index >= len(trend_list) - 2
    
    return next_state, reward, done, new_portfolio_composition, new_asset_list


def save_results(date_range, asset_list, portfolio, approach, predict):
    """Save training results"""
    # Determine save path
    base_path = f'data/rl/{portfolio}'
    if approach == 'gradual':
        subfolder = 'non_lagged' if predict else 'lagged'
    else:  # full_swing
        subfolder = 'fs_non_lagged' if predict else 'fs_lagged'
    
    save_path = f'{base_path}/{subfolder}'
    os.makedirs(save_path, exist_ok=True)

    # Save NAV data
    df_nav = pd.DataFrame({
        'Date': date_range,
        'Net': [sum(assets) for assets in asset_list]
    })

    for i, stock in enumerate(stocks):
      active_data[stock] = [assets[i] for assets in asset_list]

    df_nav = pd.DataFrame(active_data)
    df_nav[stock] = [assets[i] for assets in asset_list]
    df_nav.to_csv(f'{save_path}/daily_nav.csv', index=False)

    passive_asset_list = []
    initial_investment = 10000  # Initial investment per asset
        
        # Initialize passive portfolio
    passive_assets = [initial_investment] * len(stocks)
    passive_asset_list.append(passive_assets.copy())

    for i in range(1, len(date_range)):
            current_passive = []
            for j, stock in enumerate(stocks):
                # Get price changes
                prev_close = df_list[j][df_list[j]['Date'] == date_range[i-1]]['Close'].values[0]
                curr_close = df_list[j][df_list[j]['Date'] == date_range[i]]['Close'].values[0]
                
                # Update asset value based on price change
                passive_value = passive_asset_list[-1][j] * (curr_close / prev_close)
                current_passive.append(passive_value)
            
            passive_asset_list.append(current_passive)

    passive_data = {
        'Date': date_range,
        'Net': [sum(passive_assets) for passive_assets in passive_asset_list]
    }
    
    # Add column for each asset in passive portfolio
    for i, stock in enumerate(stocks):
        passive_data[stock] = [passive_assets[i] for passive_assets in passive_asset_list]

    #save passive NAV
    df_passive = pd.DataFrame(passive_data)
    df_passive.to_csv(save_paths['passive_nav'], index=False)
    print(f"Passive portfolio data saved to: {save_paths['passive_nav']}")


def calculate_daily_nav(portfolio_list, trend_list, date_range, df_list):
    """Calculate daily NAV for the portfolio"""
    nav_dict = {'Date': [], 'Net': []}
    current_portfolio = portfolio_list[0]
    current_assets = [1000000/len(df_list)] * len(df_list)
    
    trend_idx = 0
    for date in date_range:
        if trend_idx < len(trend_list) and date == trend_list[trend_idx]:
            current_portfolio = portfolio_list[trend_idx]
            trend_idx += 1
            
        # Update asset values
        for i, df in enumerate(df_list):
            if i == 0:  # Only add date once
                nav_dict['Date'].append(date)
            
            current_price = df[df['Date'] == date]['Close'].values[0]
            current_assets[i] *= current_price / df[df['Date'] == date]['Close'].values[0]
            
        nav_dict['Net'].append(sum(current_assets))
        
    return pd.DataFrame(nav_dict)

def calculate_passive_nav(date_range, df_list, initial_investment=1000000):
    """Calculate passive buy-and-hold NAV"""
    passive_values = []
    num_assets = len(df_list)
    asset_values = [initial_investment/num_assets] * num_assets  # Equal initial investment
    
    for date in date_range:
        current_total = 0
        for i, df in enumerate(df_list):
            row = df[df['Date'] == date]
            if not row.empty:
                current_price = row['Close'].values[0]
                if len(passive_values) > 0:
                    prev_date = passive_values[-1]['Date']
                    prev_price = df[df['Date'] == prev_date]['Close'].values[0]
                    asset_values[i] *= current_price / prev_price
                current_total += asset_values[i]
        
        passive_values.append({
            'Date': date,
            'Net': current_total
        })
    
    return pd.DataFrame(passive_values)

def get_action(state, mainQN, epsilon=0.0):
    if np.random.random() < epsilon:
        return np.random.randint(0, config.num_actions)
    
    state_tensor = norm_state(state)
    print("State tensor shape in get_action:", state_tensor.shape)
    
    q_values = mainQN(state_tensor)
    print("Q values shape in get_action:", q_values.shape)
    
    if len(q_values.shape) == 3:
        q_values = tf.reshape(q_values, [-1, config.num_actions])
    
    return np.argmax(q_values[0])

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Setup paths and load data
    save_paths = get_save_paths(args.portfolio, args.approach, args.predict)
    df_list, date_range, trend_list, stocks = util.get_algo_dataset(args.portfolio)

    # Initialize networks
    h_size = config.hidden_layer_size
    mainQN = Qnetwork(h_size)
    targetQN = Qnetwork(h_size)

    if args.load:
        try:
            # Coba load existing model
            mainQN.load_weights(save_paths['model'])
            print(f"Model loaded from: {save_paths['model']}")
            # Copy weights ke target network
            targetQN.set_weights(mainQN.get_weights())
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Starting with a new model.")
            
    mainQN.optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
    
    # Initialize replay buffer
    replay_buffer = deque(maxlen=config.buffer_size)
    
    # Initialize portfolio
    num_assets = len(stocks)
    portfolio_composition = [1.0/num_assets] * num_assets  # Equal weight initially
    asset_list = [10000/num_assets] * num_assets  # Initial investment split equally

    # Training loop
    for episode in range(config.num_episodes):
        current_index = 9  # Start from 10th data point
        state = get_next_state(current_index-1, trend_list, date_range, df_list)
        episode_reward = 0
        
        while current_index < len(trend_list)-1:
            # Get action using epsilon-greedy
            if np.random.rand() < epsilon:
                action = np.random.randint(0, config.num_actions)
            else:
                q_values = mainQN.get_q_values(norm_state(state))
                action = get_action(q_values[0].numpy())
            
            # Take action and get reward
            reward, new_portfolio_composition, new_asset_list = get_reward(
                asset_list, action, current_index, trend_list, 
                date_range, portfolio_composition, df_list
            )
            
            # Get next state
            next_state = get_next_state(current_index, trend_list, date_range, df_list)
            
            # Store transition in replay buffer
            replay_buffer.append((
                state, action, reward, next_state, 
                current_index >= len(trend_list)-2
            ))
            
            # Train on mini-batch
            if len(replay_buffer) >= config.batch_size:
                minibatch = random.sample(replay_buffer, config.batch_size)
                
                # Prepare batch data
                states = np.array([norm_state(trans[0]) for trans in minibatch])
                actions = np.array([trans[1] for trans in minibatch])
                rewards = np.array([trans[2] for trans in minibatch])
                next_states = np.array([norm_state(trans[3]) for trans in minibatch])
                dones = np.array([trans[4] for trans in minibatch])
                
                # Calculate target Q-values
                next_q_values = targetQN.get_q_values(next_states)
                max_next_q = tf.reduce_max(next_q_values, axis=1)
                targets = rewards + config.gamma * (1 - dones) * max_next_q
                
                # Train main network
                loss = mainQN.train_step(states, targets)
            
            # Update state and portfolio
            state = next_state
            portfolio_composition = new_portfolio_composition
            asset_list = new_asset_list
            current_index += 1

            # Decay epsilon
            if epsilon > config.final_exploration:
                epsilon -= config.exploration_decay
        
        # Update target network periodically
        if episode % config.target_update_freq == 0:
            targetQN.set_weights(mainQN.get_weights())
            
        # Log progress
        if episode % config.log_freq == 0:
            print(f"Episode {episode}, Epsilon: {epsilon:.4f}")
    
    # Simpan model utama (mainQN)
    mainQN.save(save_paths['model'])
    print(f"Model disimpan di: {save_paths['model']}")
    
    # Simpan data NAV harian
    df_nav = pd.DataFrame({
        'Date': date_range,
        'Net': asset_list
    })
    df_nav.to_csv(save_paths['daily_nav'], index=False)
    print(f"Data NAV harian disimpan di: {save_paths['daily_nav']}")

    df_passive = calculate_passive_nav(date_range, df_list)
    df_passive.to_csv(save_paths['passive_nav'], index=False)
    print(f"Data NAV pasif disimpan di: {save_paths['passive_nav']}")
    
    print("Proses selesai!")

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Get save paths
    save_paths = get_save_paths(args.portfolio, args.approach, args.predict)
    df_list, date_range, trend_list, stocks = util.get_algo_dataset(args.portfolio)

    # Tambahkan kode berikut di sini
    trend_list = sorted(list(set(trend_list)))  # Menghapus duplikat dan mengurutkan
    date_range = sorted(list(set(date_range)))  # Menghapus duplikat dan mengurutkan

    trend_list = [pd.Timestamp(d) for d in trend_list]
    date_range = [pd.Timestamp(d) for d in date_range]

    # Hanya gunakan tanggal dalam trend_list yang ada di date_range
    trend_list = [d for d in trend_list if d in date_range]
    
    # Create Q-networks
    mainQN = Qnetwork(config.hidden_layer_size)
    targetQN = Qnetwork(config.hidden_layer_size)

    try:
        print("Starting training...")
        print(f"Portfolio: {args.portfolio}")
        print(f"Approach: {args.approach}")
        print(f"Number of assets: {len(df_list)}")
        print(f"Date range: {date_range[0]} to {date_range[-1]}")
        print(f"Number of trading days: {len(date_range)}")
        
        final_asset_list = train_model(df_list, date_range, trend_list, stocks, args)
        
        # Print final results
        print("\nTraining completed successfully!")
        print("Final portfolio values:")
        for i, asset_value in enumerate(final_asset_list):
            print(f"Asset {i+1}: ${asset_value:,.2f}")
        print(f"Total portfolio value: ${sum(final_asset_list):,.2f}")
        
        print(f"\nResults saved to: {os.path.dirname(save_paths['model'])}")
        
        print("Training selesai. Menjalankan analisis performa...")
        import subprocess
        subprocess.run(["python", "performance_analysis.py", 
                    "--portfolio", args.portfolio, 
                    "--approach", args.approach, 
                    "--predict" if args.predict else ""])

    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise
    
    finally:
        # Clean up TensorFlow session
        tf.compat.v1.reset_default_graph()

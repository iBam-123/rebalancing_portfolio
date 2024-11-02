# Default parameters
PORTFOLIO_CONFIG = {
    'portfolio1': {
        'num_assets': 14,
        'assets': ['AUD', 'CAD', 'CNY', 'EMLC', 'EUR', 'GBP', 'INR', 'JPY', 'KRW', 'MYR', 'NZD', 'PLN', 'SGD', 'USD'],
        'start_date': '2021-01-01',
        'end_date': '2023-12-31'
    },
    'portfolio2': {
        'num_assets': 20,
        'assets': ['STOCK1', 'STOCK2', 'STOCK3', 'STOCK4', 'STOCK5',
                  'STOCK6', 'STOCK7', 'STOCK8', 'STOCK9', 'STOCK10',
                  'STOCK11', 'STOCK12', 'STOCK13', 'STOCK14', 'STOCK15',
                  'STOCK16', 'STOCK17', 'STOCK18', 'STOCK19', 'STOCK20'],
        'start_date': '2021-01-01',
        'end_date': '2023-12-31'
    }
}
label_name = 'Close'
weight_decay_beta = 0.00001
state_dimension = 5
num_actions = 4
price_period = 30

# Training parameters
num_episodes = 350
gamma = 0.99
batch_size = 128
buffer_size = 50000
initial_exploration = 1.0
final_exploration = 0.1
final_exploration_frame = 10000
exploration_frame = 500000
target_update_freq = 1000
exploration_decay = (initial_exploration - final_exploration) / exploration_frame

save_interval = 100
start_e = 1.0
end_e = 0.1
annealing_steps = 10000
target_update_freq = 1000
log_freq = 100
# Network parameters
hidden_layer_size = 32
learning_rate = 0.001
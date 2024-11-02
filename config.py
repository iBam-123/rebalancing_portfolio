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
        'assets': ['USD Govt', 'USD SSA', 'EMUSD Int.', 'EMUSD Ext.', 'US MBS Int.', 'US MBS Ext.', 'US Corp Int.', 'US Corp Ext.', 'EUR Govt', 'EUR SSA', 'GBP Govt', 'GBP SSA', 'AUD Govt', 'AUD SSA', 'NZD Govt', 'CAD Govt', 'CAD SSA', 'CNY Govt', 'JPY Govt', 'EMLC'],
        'start_date': '2021-01-01',
        'end_date': '2023-12-31'
    }
}
label_name = 'Close'
weight_decay_beta = 0.00001
state_dimension = 5
num_actions = 4
price_period = 10

# Training parameters
num_episodes = 1000
gamma = 0.99
batch_size = 32
buffer_size = 100000
initial_exploration = 1.0
final_exploration = 0.1
final_exploration_frame = 10000
exploration_frame = 500000
target_update_freq = 1000
exploration_decay = (initial_exploration - final_exploration) / exploration_frame

# Network parameters
hidden_layer_size = 64
learning_rate = 0.001
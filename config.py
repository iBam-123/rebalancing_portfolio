# Default parameters
label_name = 'Close'
weight_decay_beta = 0.00001
state_dimension = 5
num_actions = 4
price_period = 10

# Training parameters
gamma = 0.99
batch_size = 32
buffer_size = 1000000
initial_exploration = 1.0
final_exploration = 0.1
final_exploration_frame = 10000
exploration_frame = 500000
exploration_decay = (initial_exploration - final_exploration) / exploration_frame

# Network parameters
hidden_layer_size = 50 
learning_rate = 0.001
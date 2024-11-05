import torch
import numpy as np

# --------------------- Computer Parameters ---------------------
max_workers = 8

# --------------------- Wandb ---------------------
project_name = 'EdgeGAT-CVRPSTW'

# --------------------- Graph Parameters ---------------------
# change the number of nearest neighbors when using different environment
k_distance_nearest_neighbors_percent = 0.1
k_time_nearest_neighbors_percent = 0.1

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# number of instances
num_samll_instances = 1280000
num_medium_instances = 640000
num_large_instances = 640000


# --------------------- Simulation Parameters ---------------------
small_params = {
    'batch_size': 512,
    'grid_size': 10.,
    'num_customers': 20,
    'num_vehicles_choices': [2,3],
    'vehicle_capacity_choices': [60.,60.],
    'customer_demand_range_choices': [(0.,10.),(0.,15.)],
    'time_window_range': (0.,30.),
    'time_window_length': -1,
    'early_penalty_alpha_range': (0.,0.2),
    'late_penalty_beta_range': (0.,1.),
    'wait_times':  np.linspace(0, 3, 10),
    'demand_unmet_penalty': -100,
    'index': 0
}

medium_params = {
    'batch_size': 256,
    'grid_size': 10.,
    'num_customers': 50,
    'num_vehicles_choices': [2,3,4,5],
    'vehicle_capacity_choices': [150.,150.,150.,150.],
    'customer_demand_range_choices': [(0.,10.),(0.,15.),(0.,20.),(0.,25.)],
    'time_window_range': (0.,40.),
    'time_window_length': -1,
    'early_penalty_alpha_range': (0.,0.2),
    'late_penalty_beta_range': (0.,1.),
    'wait_times':  np.linspace(0, 4, 20),
    'index': 0
}

large_params = {
    'batch_size': 256,
    'grid_size': 10.,
    'num_customers': 100,
    'num_vehicles_choices': [2,3,4,5],
    'vehicle_capacity_choices': [300.,300.,300.,300.],
    'customer_demand_range_choices': [(0.,10.),(0.,15.),(0.,20.),(0.,25.)],
    'time_window_range': (0.,60.),
    'time_window_length': -1,
    'early_penalty_alpha_range': (0.,0.2),
    'late_penalty_beta_range': (0.,1.),
    'wait_times':  np.linspace(0, 4, 30),
    'index': 0
}

extra_large_params = {
    'batch_size': 256,
    'grid_size': 10.,
    'num_customers': 150,
    'num_vehicles_choices': [5],
    'vehicle_capacity_choices': [180.],
    'customer_demand_range_choices': [(0.,10.)],
    'time_window_range': (0.,60.),
    'time_window_length': 20.,
    'early_penalty_alpha_range': (0.1,0.1),
    'late_penalty_beta_range': (0.5,0.5),
    'wait_times':  np.linspace(0, 4, 20),
    'index': 0
}

# --------------- Logger ----------------
record_gradient = True
reward_window_size = 100

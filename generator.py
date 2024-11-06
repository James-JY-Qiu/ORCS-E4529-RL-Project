import numpy as np
import pandas as pd


def generate_instance(
    grid_size=10,
    num_customers=20,
    num_vehicles_choices=[2,3],
    vehicle_capacity_choices=[60.,60.],
    customer_demand_range_choices=[(0.,10.),(0.,15.)],
    time_window_range=(0.,30.),
    time_window_length=-1,
    early_penalty_alpha_range=(0.,0.2),
    late_penalty_beta_range=(0.,1.),
    index=0
):
    depot = np.random.uniform(0, grid_size, size=(1, 2))
    customers = np.random.uniform(0, grid_size, size=(num_customers, 2))

    if time_window_length == -1:
        time_windows = np.random.uniform(time_window_range[0], time_window_range[1], size=(num_customers, 2))
        time_windows = np.sort(time_windows, axis=1)
    else:
        time_windows = np.random.uniform(time_window_range[0], time_window_range[1], size=(num_customers, 1))
        time_windows = np.concatenate([time_windows, time_windows + time_window_length], axis=1)

    alpha = np.random.uniform(early_penalty_alpha_range[0], early_penalty_alpha_range[1], size=(num_customers,))
    beta = np.random.uniform(late_penalty_beta_range[0], late_penalty_beta_range[1], size=(num_customers,))

    num_vehicle_idx = index
    num_vehicles = num_vehicles_choices[num_vehicle_idx]
    vehicle_capacity = vehicle_capacity_choices[num_vehicle_idx]
    customer_demand_range = customer_demand_range_choices[num_vehicle_idx]
    demand = np.random.uniform(customer_demand_range[0], customer_demand_range[1], size=(num_customers,))
    while sum(demand) > vehicle_capacity * num_vehicles:
        demand = np.random.uniform(customer_demand_range[0], customer_demand_range[1], size=(num_customers,))

    # 合并所有信息
    # merge all the information
    customer_data = {
        "Customer_ID": np.arange(1, num_customers + 1),
        "X": customers[:, 0],
        "Y": customers[:, 1],
        "Demand": demand,
        "Start_Time_Window": time_windows[:, 0],
        "End_Time_Window": time_windows[:, 1],
        "Alpha": alpha,
        "Beta": beta,
        "Service_Time": np.zeros((num_customers,), dtype=float),
    }
    company_data = {
        'depot': depot,
        'Num_Customers': num_customers,
        'Num_Vehicles': num_vehicles,
        'Vehicle_Capacity': vehicle_capacity,
        'Max_Time': time_window_range[1] * 2
    }

    customer_df = pd.DataFrame(customer_data)
    customer_df['Is_customer'] = 1

    # 将仓库数据和客户数据合并
    # merge the depot data and customer data
    customer_df = pd.concat([pd.DataFrame(depot, columns=['X', 'Y']), customer_df], ignore_index=True)
    customer_df.fillna(0, inplace=True)
    customer_df.iloc[0, 5] = company_data['Max_Time']

    return customer_df, company_data


if __name__ == '__main__':
    customer_data, company_data = generate_instance()

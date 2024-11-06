import numpy as np
from generator import generate_instance

from concurrent.futures import ProcessPoolExecutor


def generate_vrp_instance(args):
    (grid_size, num_customers, num_vehicles_choices, vehicle_capacity_choices, customer_demand_range_choices,
     time_window_range, time_window_length, early_penalty_alpha_range, late_penalty_beta_range, index) = args

    customer_data, company_data = generate_instance(
        grid_size=grid_size,
        num_customers=num_customers,
        num_vehicles_choices=num_vehicles_choices,
        vehicle_capacity_choices=vehicle_capacity_choices,
        customer_demand_range_choices=customer_demand_range_choices,
        time_window_range=time_window_range,
        time_window_length=time_window_length,
        early_penalty_alpha_range=early_penalty_alpha_range,
        late_penalty_beta_range=late_penalty_beta_range,
        index=index
    )
    vrp_env = VRPEnv(customer_data, company_data)
    return vrp_env


class VRPEnv:
    def __init__(
            self,
            customer_data,
            company_data
    ):
        """
        初始化状态对象，参考 generate_instance()
        Initialize the state object, refer to generate_instance()
        customer_data: 生成的客户数据 generated customer data
        company_data: 生成的公司数据 generated company data
        """
        self.customer_data = customer_data
        self.company_data = company_data

        self.num_customers = company_data['Num_Customers']
        self.customer_node_index_list = list(range(1, self.num_customers + 1))
        self.vehicle_count = company_data['Num_Vehicles']
        self.vehicle_capacity = company_data['Vehicle_Capacity']
        self.customer_demands = customer_data['Demand'].values
        self.customer_time_windows = customer_data[['Start_Time_Window', 'End_Time_Window']].values
        self.customer_service_times = customer_data['Service_Time'].values
        self.customer_penalties = customer_data[['Alpha', 'Beta']].values

        # 所有存储变量
        # Store all variables
        self.vehicle_positions = None
        self.vehicle_leave = None
        self.remaining_capacities = None
        self.time_elapsed = None
        self.current_customer_demands = None
        self.finished_customers = None
        self.finished_vehicle = None
        self.travel_time_matrix = None
        self.num_nodes = None
        self.wait_times = None
        self.num_wait_time_dummy_node = None
        self.wait_time_dummy_node_index_list = None
        self.dummy_wait_node_index_thred = None
        self.demand_unmet_penalty = None

        self.reset()  # 初始化状态 reset status

    def reset(self):
        # 初始化每辆车的状态
        # Initialize the status of each vehicle
        # 每辆车当前的位置index（初始在仓库）
        # The current position index of each vehicle (initially in the depot)
        self.vehicle_positions = np.zeros(self.vehicle_count, dtype=np.int64)
        # 判断车辆是否离开depot
        # Determine whether the vehicle has left the depot
        self.vehicle_leave = np.zeros(self.vehicle_count, dtype=bool)
        # 每辆车的剩余容量
        # Remaining capacity of each vehicle
        self.remaining_capacities = np.full((self.vehicle_count,), self.vehicle_capacity, dtype=np.float64)
        # 判断车辆是否结束
        # Determine whether the vehicle has finished
        self.finished_vehicle = np.zeros(self.vehicle_count, dtype=bool)
        # 每辆车已经经过的时间
        # The time each vehicle has passed
        self.time_elapsed = np.zeros(self.vehicle_count, dtype=np.float64)
        # 客户当前的需求
        # The current demand of the customer
        self.current_customer_demands = self.customer_demands.copy()
        # 客户是否已完成需求
        # Whether the customer has completed the demand
        self.finished_customers = np.zeros(self.num_customers + 1, dtype=bool)

    def update_parameters(
            self, distance_matrix, num_nodes, wait_times, demand_unmet_penalty
    ):
        """
        更新环境参数
        Update the environment parameters
        Args:
            distance_matrix: 每个实例的距离矩阵 distance matrix for each instance
            num_nodes: 每个实例的节点数量 number of nodes for each instance
            wait_times: 等待时间列表 list of wait times
            demand_unmet_penalty: 未满足需求的惩罚 penalty for unmet demand
        """
        self.travel_time_matrix = distance_matrix
        self.num_nodes = num_nodes
        self.wait_times = wait_times
        self.num_wait_time_dummy_node = len(wait_times)
        self.wait_time_dummy_node_index_list = list(range(
            self.num_customers + 1, self.num_customers + 1 + self.num_wait_time_dummy_node
        ))
        self.dummy_wait_node_index_thred = self.wait_time_dummy_node_index_list[0]

        self.demand_unmet_penalty = demand_unmet_penalty

    def update_vehicle_position(self, vehicle_id, new_position):
        """
        更新车辆的位置及状态
        Update the position and status of the vehicle
        Args:
            vehicle_id: 车辆的ID vehicle ID
            new_position: 车辆的新位置（目标客户ID） new position of the vehicle (target customer ID)
        Returns:
            last_position: 上一个位置（当前客户ID） last position (current customer ID)
        """
        last_position = self.vehicle_positions[vehicle_id]
        # 判断车辆是否结束
        # Determine whether the vehicle has finished
        if new_position == 0:
            self.finished_vehicle[vehicle_id] = True
        # 更新车辆位置到新的位置
        # Update the vehicle position to the new position
        self.vehicle_positions[vehicle_id] = new_position
        if last_position == 0 and new_position in self.wait_time_dummy_node_index_list:
            self.vehicle_leave[vehicle_id] = True
        else:
            # 更新车辆时间
            # Update the vehicle time
            if last_position in self.wait_time_dummy_node_index_list:
                travel_time = self.travel_time_matrix[0, new_position]
                wait_time_index = last_position - self.dummy_wait_node_index_thred
                service_time_last_customer = self.wait_times[wait_time_index]
            else:
                travel_time = self.travel_time_matrix[last_position, new_position]
                service_time_last_customer = self.customer_service_times[last_position]
            self.time_elapsed[vehicle_id] += service_time_last_customer + travel_time
            # 更新车辆的剩余容量
            # Update the remaining capacity of the vehicle
            if new_position != 0:
                current_capacities = self.remaining_capacities[vehicle_id]
                current_customer_demand = self.current_customer_demands[new_position]
                self.current_customer_demands[new_position] = max(
                    0, current_customer_demand - current_capacities
                )
                self.remaining_capacities[vehicle_id] = max(
                    0, current_capacities - current_customer_demand
                )
                # 标记该客户已完成需求
                # Mark the customer as having completed the demand
                if self.current_customer_demands[new_position] == 0:
                    self.finished_customers[new_position] = True

        return last_position

    def get_current_customer_max_time(self):
        """
        获取当前客户的最大时间
        Get the maximum time of the current customer
        :return: 当前剩余客户的最大时间 Maximum time of the remaining customers
        """
        max_time = 0
        for customer_id in range(1, self.num_customers + 1):
            if not self.finished_customers[customer_id]:
                max_time = max(max_time, self.customer_time_windows[customer_id][1])

        return max_time

    def get_valid_actions(self, vehicle_id):
        """
        获取给定车辆的有效行动（即所有未完成的客户）
        Get the valid actions for the given vehicle (i.e., all unfinished customers)
        Args:
            vehicle_id: 车辆ID vehicle ID
        Returns:
            valid_customers: 该车辆可访问的合法客户列表 list of valid customers that the vehicle can access
        """
        current_location = self.vehicle_positions[vehicle_id]
        # 如果车辆没离开depot，则选择等待 action
        # If the vehicle has not left the depot, select the wait action
        if not self.vehicle_leave[vehicle_id]:
            valid_customers = self.wait_time_dummy_node_index_list
        # 如果车辆 remaining capacity 为 0，所有用户已经访问完，或者已经在depot，则返回 depot
        # If the vehicle's remaining capacity is 0, all users have been visited, or it is already at the depot, return the depot
        elif (
                self.remaining_capacities[vehicle_id] == 0 or
                self.finished_customers[1:].all() or
                current_location == 0
        ):
            valid_customers = [0]
        else:
            valid_customers = []
            # 如果其他车辆容量足够服务剩下的客户，并且自身容量不足10%，则可以选择返回depot
            # If the remaining capacity of other vehicles is sufficient to serve the remaining customers, and the capacity of the vehicle itself is less than 10%, it can choose to return to the depot
            if self.remaining_capacities[vehicle_id] < self.vehicle_capacity * 0.1:
                total_remaining_demands = self.current_customer_demands[1:].sum()
                total_other_active_vehicles_capacities = sum(
                    [self.remaining_capacities[v] for v in range(self.vehicle_count)
                     if not self.finished_vehicle[v] and v != vehicle_id]
                )
                if total_remaining_demands <= total_other_active_vehicles_capacities:
                    valid_customers.append(0)
            for customer_id in range(1, self.num_customers + 1):
                if self.finished_customers[customer_id]:
                    continue
                # 如果剩余容量足够服务该客户，则加入到valid_customers
                # If the remaining capacity is sufficient to serve the customer, add it to valid_customers
                if self.current_customer_demands[customer_id] <= self.remaining_capacities[vehicle_id]:
                    valid_customers.append(customer_id)

            # 如果没有合法客户，则返回 depot
            # If there are no valid customers, return the depot
            if not valid_customers:
                valid_customers.append(0)

        return valid_customers

    def return_neg_inf_mask(self, vehicle_id):
        """
        为车辆的决策空间应用掩码，防止选择非法客户
        Apply a mask to the decision space of the vehicle to prevent the selection of illegal customers
        Args:
            vehicle_id: 车辆ID vehicle ID
        Returns:
            neg_inf_mask: -inf 掩码 -inf mask
        """
        neg_inf_mask = np.full(self.num_nodes, float('-inf'))
        valid_customers = self.get_valid_actions(vehicle_id)
        neg_inf_mask[valid_customers] = 0.0
        return neg_inf_mask

    def calculate_reward(self, vehicle_id, action, last_position):
        """
        计算执行一个action的reward
        Calculate the reward of executing
        Args:
            vehicle_id: 车辆ID vehicle ID
            action: 动作 action
            last_position: 上一个位置 last position
        Returns:
            reward: 奖励 reward
        """
        if action in self.wait_time_dummy_node_index_list:
            return 0.0
        distance = self.travel_time_matrix[last_position, action]
        arrive_time = self.time_elapsed[vehicle_id]
        customer_time_window = self.customer_time_windows[action]
        customer_penalty = self.customer_penalties[action]

        reward = -distance
        # 没有回到 depot，则要计算 penalty
        # If not returned to the depot, the penalty must be calculated
        if action != 0:
            if arrive_time < customer_time_window[0]:
                reward -= customer_penalty[0] * (customer_time_window[0] - arrive_time)
            elif arrive_time > customer_time_window[1]:
                reward -= customer_penalty[1] * (arrive_time - customer_time_window[1])
        # 如果所有车辆已完成，则减去unmet demand惩罚
        # If all vehicles have been completed, subtract the unmet demand penalty
        if self.finished_vehicle.all() and not self.finished_customers[1:].all():
            reward += self.demand_unmet_penalty * sum(self.finished_customers[1:] == False)

        return reward

    def judge_finish(self):
        """
        判断是否结束
        Judge whether it is over
        Returns:
            done: 是否结束 whether it is over
        """
        finished = self.finished_vehicle.all()
        return bool(finished)

    def step(self, vehicle_id, action):
        """
        执行一个action，返回 reward, 以及是否结束
        Execute an action, return reward, and whether it is over
        Args:
            vehicle_id: 车辆ID vehicle ID
            action: 动作 action
        Returns:
            reward: 奖励 reward
        """
        if self.finished_vehicle[vehicle_id]:
            return 0.0

        # 更新车辆位置
        # Update the vehicle position
        last_position = self.update_vehicle_position(vehicle_id, action)
        # 计算奖励
        # Calculate the reward
        reward = self.calculate_reward(vehicle_id, action, last_position)

        return reward


class BatchVRPEnvs:
    def __init__(
            self,
            batch_size,
            wait_times,
            max_workers,
            demand_unmet_penalty,
            grid_size,
            num_customers,
            num_vehicles_choices,
            vehicle_capacity_choices,
            customer_demand_range_choices,
            time_window_range,
            time_window_length,
            early_penalty_alpha_range,
            late_penalty_beta_range,
            index,
    ):
        """
        初始化状态对象，参考 generate_instance()
        Initialize the state object, refer to generate_instance()
        batch_size: batch 的大小 batch size
        wait_times: 等待时间 list of wait times
        max_workers: 最大工作进程数 maximum number of working processes
        """
        self.batch_size = batch_size
        self.wait_times = wait_times
        self.dummy_wait_node_index_thred = num_customers + 1
        self.num_wait_time_dummy_node = len(wait_times)
        self.max_workers = max_workers

        self.demand_unmet_penalty = demand_unmet_penalty

        self.grid_size = grid_size
        self.num_customers = num_customers
        self.num_vehicles_choices = num_vehicles_choices
        self.vehicle_capacity_choices = vehicle_capacity_choices
        self.vehicle_capacity = vehicle_capacity_choices[index]
        self.customer_demand_range_choices = customer_demand_range_choices
        self.time_window_range = time_window_range
        self.time_window_length = time_window_length
        self.early_penalty_alpha_range = early_penalty_alpha_range
        self.late_penalty_beta_range = late_penalty_beta_range
        self.index = index

        # 存储所有env
        # Store all env
        self.envs = []

        # 存储batch数据
        # Store batch data
        self.batch_customer_data = None
        self.batch_company_data = None

    def reset(self, generate=True):
        if generate:
            try:
                # 使用多进程并行生成 VRP 环境
                # Use multi-process parallel generation of VRP environment
                with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                    args = [(
                        self.grid_size, self.num_customers, self.num_vehicles_choices, self.vehicle_capacity_choices,
                        self.customer_demand_range_choices, self.time_window_range, self.time_window_length,
                        self.early_penalty_alpha_range, self.late_penalty_beta_range, self.index
                    ) for _ in range(self.batch_size)]
                    # 并行生成
                    # Parallel generation
                    self.envs = list(executor.map(generate_vrp_instance, args))
                    # 记录 batch 数据
                    # Record batch data
                    self.batch_customer_data = [env.customer_data for env in self.envs]
                    self.batch_company_data = [env.company_data for env in self.envs]
            except KeyboardInterrupt:
                print("KeyboardInterrupt, shutting down the process pool")
            finally:
                # 确保在手动中断后关闭进程池
                # Ensure that the process pool is closed after manual interruption
                executor.shutdown(wait=True)
        else:
            for env in self.envs:
                env.reset()

    def update_parameters(self, batch_distance_matrices, batch_num_nodes):
        """
        更新环境参数
        Update the environment parameters
        Args:
            batch_distance_matrices: 每个实例的距离矩阵 distance matrix for each instance
            batch_num_nodes: 每个实例的节点数量 number of nodes for each instance
        """
        for i, vrp_env in enumerate(self.envs):
            vrp_env.update_parameters(
                batch_distance_matrices[i],
                batch_num_nodes[i],
                self.wait_times,
                self.demand_unmet_penalty,
            )

    def get_current_batch_status(self, return_neg_inf_mask=False):
        """
        返回当前 batch 的状态
        return the status of the current batch
        Returns:
            batch_status: 当前 batch 的状态 current status of the batch
        """
        batch_vehicle_positions = []
        batch_remaining_capacities = []
        batch_time_elapsed = []
        batch_customer_max_time = []
        batch_customer_remaining_demands = []
        batch_neg_inf_mask = []
        for env in self.envs:
            batch_vehicle_positions.append(env.vehicle_positions)
            batch_remaining_capacities.append(env.remaining_capacities)
            batch_time_elapsed.append(env.time_elapsed)
            batch_customer_max_time.append(env.get_current_customer_max_time())
            batch_customer_remaining_demands.append(env.current_customer_demands)
            if return_neg_inf_mask:
                instance_neg_inf_mask = []
                vehicle_count = env.vehicle_count
                for vehicle_id in range(vehicle_count):
                    neg_inf_mask = env.return_neg_inf_mask(vehicle_id)
                    instance_neg_inf_mask.append(neg_inf_mask)
                instance_neg_inf_mask = np.array(instance_neg_inf_mask)
                batch_neg_inf_mask.append(instance_neg_inf_mask)

        status = {
            'batch_vehicle_positions': np.array(batch_vehicle_positions),
            'batch_remaining_capacities': np.array(batch_remaining_capacities),
            'batch_time_elapsed': np.array(batch_time_elapsed),
            'batch_customer_max_time': np.array(batch_customer_max_time),
            'batch_customer_remaining_demands': np.array(batch_customer_remaining_demands)
        }
        if return_neg_inf_mask:
            status['batch_neg_inf_mask'] = np.array(batch_neg_inf_mask)

        return status
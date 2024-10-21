import torch
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
        customer_data: 生成的客户数据
        company_data: 生成的公司数据
        """
        self.customer_data = customer_data
        self.company_data = company_data

        self.num_customers = company_data['Num_Customers']
        self.dummy_wait_node_index_thred = self.num_customers + 1
        self.vehicle_count = company_data['Num_Vehicles']
        self.vehicle_capacity = company_data['Vehicle_Capacity']
        self.customer_demands = customer_data['Demand'].values
        self.customer_time_windows = customer_data[['Start_Time_Window', 'End_Time_Window']].values
        self.customer_service_times = customer_data['Service_Time'].values
        self.customer_penalties = customer_data[['Alpha', 'Beta']].values

        # 所有存储变量
        self.vehicle_positions = None
        self.vehicle_leave = None
        self.remaining_capacities = None
        self.time_elapsed = None
        self.current_customer_demands = None
        self.finished_customers = None
        self.travel_time_matrix = None
        self.num_nodes = None
        self.wait_times = None
        self.num_wait_time_dummy_node = None

        self.reset()  # 初始化状态

    def reset(self):
        # 初始化每辆车的状态
        # 每辆车当前的位置index（初始在仓库）
        self.vehicle_positions = np.zeros(self.vehicle_count, dtype=np.int64)
        # 判断车辆是否离开depot
        self.vehicle_leave = np.zeros(self.vehicle_count, dtype=bool)
        # 每辆车的剩余容量
        self.remaining_capacities = np.full((self.vehicle_count,), self.vehicle_capacity, dtype=np.float64)
        # 每辆车已经经过的时间
        self.time_elapsed = np.zeros(self.vehicle_count, dtype=np.float64)
        # 客户当前的需求
        self.current_customer_demands = self.customer_demands.copy()
        # 客户是否已完成需求
        self.finished_customers = np.zeros(self.num_customers + 1, dtype=bool)

    def update_parameters(self, distance_matrix, num_nodes, wait_times):
        """
        更新环境参数
        Args:
            distance_matrix: 每个实例的距离矩阵
            num_nodes: 每个实例的节点数量
            wait_times: 等待时间列表
        """
        self.travel_time_matrix = distance_matrix
        self.num_nodes = num_nodes
        self.wait_times = wait_times
        self.num_wait_time_dummy_node = len(wait_times)

    def update_vehicle_position(self, vehicle_id, new_position):
        """
        更新车辆的位置及状态
        Args:
            vehicle_id: 车辆的ID
            new_position: 车辆的新位置（目标客户ID）
        Returns:
            last_position: 上一个位置（当前客户ID）
        """
        last_position = self.vehicle_positions[vehicle_id]

        # 如果车辆没离开 depot，则执行等待 action 或者从dummy waiting nodes 返回 depot
        if not self.vehicle_leave[vehicle_id]:
            # 如果车辆在 dummy waiting nodes，记录等待时间，更新 self.batch_vehicle_leave
            if last_position >= self.dummy_wait_node_index_thred:
                # 更新车辆时间
                wait_time_idx = last_position - self.dummy_wait_node_index_thred
                wait_time = self.wait_times[wait_time_idx]
                self.time_elapsed[vehicle_id] += wait_time
                self.vehicle_leave[vehicle_id] = True
            # 更新车辆位置到新的位置
            self.vehicle_positions[vehicle_id] = new_position

        # 如果车辆没有待在 depot 不动 （运送完毕）
        elif not (last_position == new_position == 0):
            # 更新车辆时间
            travel_time = self.travel_time_matrix[last_position, new_position]
            service_time_last_customer = self.customer_service_times[last_position]
            self.time_elapsed[vehicle_id] += service_time_last_customer + travel_time
            # 更新车辆位置为新的客户
            self.vehicle_positions[vehicle_id] = new_position
            # 更新车辆的剩余容量
            current_capacities = self.remaining_capacities[vehicle_id]
            current_customer_demand = self.current_customer_demands[new_position]
            self.current_customer_demands[new_position] = max(
                0, current_customer_demand - current_capacities
            )
            self.remaining_capacities[vehicle_id] = max(
                0, current_capacities - current_customer_demand
            )
            # 标记该客户已完成需求
            if self.current_customer_demands[new_position] == 0:
                self.finished_customers[new_position] = True

        return last_position

    def get_valid_actions(self, vehicle_id):
        """
        获取给定车辆的有效行动（即所有未完成的客户）
        Args:
            vehicle_id: 车辆ID
        Returns:
            valid_customers: 该车辆可访问的合法客户列表
        """
        # 如果车辆没离开depot，则选择等待 action 或者从dummy waiting nodes 返回 depot
        if not self.vehicle_leave[vehicle_id]:
            current_position = self.vehicle_positions[vehicle_id]
            # 如果车辆在 dummy waiting nodes, 只能返回 depot
            if current_position >= self.dummy_wait_node_index_thred:
                valid_customers = [0]
            # 如果车辆在 depot，则前往 dummy waiting nodes
            elif current_position == 0:
                valid_customers = list(range(
                    self.dummy_wait_node_index_thred,
                    self.dummy_wait_node_index_thred + self.num_wait_time_dummy_node
                ))
            else:
                raise Exception('Logic Error! Check the update of self.batch_vehicle_leave')

        # 如果车辆 remaining capacity 为 0，或者所有用户已经访问完，则返回 depot
        elif (self.remaining_capacities[vehicle_id] == 0 or
              self.finished_customers[1:].all()):
            valid_customers = [0]
        else:
            valid_customers = []
            for customer_id in range(1, self.num_customers + 1):
                if self.finished_customers[customer_id]:
                    continue
                valid_customers.append(customer_id)

        return valid_customers

    def return_neg_inf_mask(self, vehicle_id):
        """
        为车辆的决策空间应用掩码，防止选择非法客户
        Args:
            vehicle_id: 车辆ID
        Returns:
            neg_inf_mask: -inf 掩码
        """
        neg_inf_mask = torch.full((self.num_nodes,), float('-inf'), requires_grad=False)
        valid_customers = self.get_valid_actions(vehicle_id)
        neg_inf_mask[valid_customers] = 0.0
        return neg_inf_mask

    def calculate_reward(self, vehicle_id, action, last_position):
        """
        计算执行一个action的reward
        Args:
            vehicle_id: 车辆ID
            action: 动作
            last_position: 上一个位置
        Returns:
            reward: 奖励
        """
        if (last_position == action == 0) or (
                action >= self.dummy_wait_node_index_thred) or (
                last_position >= self.dummy_wait_node_index_thred
        ):
            return 0.0
        distance = self.travel_time_matrix[last_position, action]
        arrive_time = self.time_elapsed[vehicle_id]
        customer_time_window = self.customer_time_windows[action]
        customer_penalty = self.customer_penalties[action]

        reward = -distance
        # 没有回到 depot，则要计算 penalty
        if action != 0:
            if arrive_time < customer_time_window[0]:
                reward -= customer_penalty[0] * (customer_time_window[0] - arrive_time)
            elif arrive_time > customer_time_window[1]:
                reward -= customer_penalty[1] * (arrive_time - customer_time_window[1])

        return reward

    def judge_finish(self):
        """
        判断是否结束
        Returns:
            done: 是否结束
        """
        finished = self.finished_customers[1:].all() and (self.vehicle_positions == 0).all()
        return bool(finished)

    def step(self, vehicle_id, action):
        """
        执行一个action，返回 reward, 以及是否结束
        Args:
            vehicle_id: 车辆ID
            action: 动作
        Returns:
            reward: 奖励
        """
        # 更新车辆位置
        last_position = self.update_vehicle_position(vehicle_id, action)
        # 计算奖励
        reward = self.calculate_reward(vehicle_id, action, last_position)

        return reward


class BatchVRPEnvs:
    def __init__(
            self,
            batch_size,
            wait_times,
            max_workers,
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
        batch_size: batch 的大小
        wait_times: 等待时间 list
        max_workers: 最大工作进程数
        """
        self.batch_size = batch_size
        self.wait_times = wait_times
        self.dummy_wait_node_index_thred = num_customers + 1
        self.num_wait_time_dummy_node = len(wait_times)
        self.max_workers = max_workers

        self.grid_size = grid_size
        self.num_customers = num_customers
        self.num_vehicles_choices = num_vehicles_choices
        self.vehicle_capacity_choices = vehicle_capacity_choices
        self.customer_demand_range_choices = customer_demand_range_choices
        self.time_window_range = time_window_range
        self.time_window_length = time_window_length
        self.early_penalty_alpha_range = early_penalty_alpha_range
        self.late_penalty_beta_range = late_penalty_beta_range
        self.index = index

        # 存储所有env
        self.envs = []

        # 存储batch数据
        self.batch_customer_data = None
        self.batch_company_data = None

    def reset(self, generate=True):
        if generate:
            try:
                # 使用多进程并行生成 VRP 环境
                with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                    args = [(
                        self.grid_size, self.num_customers, self.num_vehicles_choices, self.vehicle_capacity_choices,
                         self.customer_demand_range_choices, self.time_window_range, self.time_window_length,
                         self.early_penalty_alpha_range, self.late_penalty_beta_range, self.index
                    ) for _ in range(self.batch_size)]
                    # 并行生成
                    self.envs = list(executor.map(generate_vrp_instance, args))
                    # 记录 batch 数据
                    self.batch_customer_data = [env.customer_data for env in self.envs]
                    self.batch_company_data = [env.company_data for env in self.envs]
            except KeyboardInterrupt:
                print("手动中断，正在关闭进程池...")
            finally:
                # 确保在手动中断后关闭进程池
                executor.shutdown(wait=True)
        else:
            for env in self.envs:
                env.reset()

    def update_parameters(self, batch_distance_matrices, batch_num_nodes):
        """
        更新环境参数
        Args:
            batch_distance_matrices: 每个实例的距离矩阵
            batch_num_nodes: 每个实例的节点数量
        """
        for i, vrp_env in enumerate(self.envs):
            vrp_env.update_parameters(batch_distance_matrices[i], batch_num_nodes[i], self.wait_times)

    def get_current_batch_status(self):
        """
        返回当前 batch 的状态
        Returns:
            batch_status: 当前 batch 的状态
        """
        batch_remaining_capacities = [env.remaining_capacities for env in self.envs]
        batch_time_elapsed = [env.time_elapsed for env in self.envs]
        batch_vehicle_positions = [env.vehicle_positions for env in self.envs]
        batch_current_customer_demands = np.array([env.current_customer_demands for env in self.envs])
        batch_customer_time_windows = np.array([env.customer_time_windows for env in self.envs])
        return {
            'batch_remaining_capacities': batch_remaining_capacities,
            'batch_time_elapsed': batch_time_elapsed,
            'batch_vehicle_positions': batch_vehicle_positions,
            'batch_current_customer_demands': batch_current_customer_demands,
            'batch_customer_time_windows': batch_customer_time_windows,
        }





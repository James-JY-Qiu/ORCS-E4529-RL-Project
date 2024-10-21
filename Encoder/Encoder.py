import dgl
import torch
from torch import nn

from Encoder import build_graph


class Encoder(nn.Module):
    def __init__(
            self,
            encoder_model,
            encoder_params,
            device
    ):
        """
        :param encoder_model: 编码器模型
        :param encoder_params: 编码器参数
        :param device: 设备
        """
        super(Encoder, self).__init__()
        self.encoder = encoder_model(**encoder_params)
        self.device = device

        # 所有存储变量
        self.num_customers = None
        self.wait_times = None
        self.dummy_wait_node_index_thred = None
        self.num_wait_time_dummy_node = None
        self.batch_graphs = None
        self.batch_node_features = None
        self.batch_edge_features = None
        self.batch_distance_matrices = None
        self.batch_encode_node_features = None
        self.batch_global_embedding = None
        self.batch_num_nodes = None

    def add_dummy_wait_time_nodes(self, g):
        """
        添加虚拟的等待时间节点连接到depot
        Args:
            g: 生成好的图
        """
        g.add_nodes(self.num_wait_time_dummy_node)
        # 获取已有的节点特征
        existing_node_features = g.ndata['features']
        # 创建新的节点特征
        new_node_features = torch.zeros(size=(self.num_wait_time_dummy_node, existing_node_features.size(1)))
        # 坐标和depot一致
        new_node_features[:, 0] = g.ndata['features'][0, 0]
        new_node_features[:, 1] = g.ndata['features'][0, 1]
        # Service Time等于等待时间
        new_node_features[:, 9] = self.wait_times
        # 更新图的节点特征
        existing_node_features[-self.num_wait_time_dummy_node:, :] = new_node_features
        g.ndata['features'] = existing_node_features

        # 添加边
        src_nodes = [0] * self.num_wait_time_dummy_node + list(
            range(self.num_customers + 1, self.num_customers + self.num_wait_time_dummy_node + 1))
        dst_nodes = list(range(self.num_customers + 1, self.num_customers + self.num_wait_time_dummy_node + 1)) + [
            0] * self.num_wait_time_dummy_node
        g.add_edges(src_nodes, dst_nodes)
        # 获取已有的边的特征
        existing_edge_features = g.edata['edge_features']
        # 创建新的边特征
        new_edge_features = torch.zeros(size=(len(src_nodes), existing_edge_features.size(1)))
        # 最早服务时间
        new_edge_features[self.num_wait_time_dummy_node:, 1] = self.wait_times
        # 最晚服务时间
        new_edge_features[self.num_wait_time_dummy_node:, 2] = self.wait_times
        # 最早服务开始时间差
        new_edge_features[self.num_wait_time_dummy_node:, 3] = self.wait_times
        # 最早服务结束时间差
        new_edge_features[:self.num_wait_time_dummy_node, 4] = self.wait_times
        new_edge_features[self.num_wait_time_dummy_node:, 4] = -self.wait_times
        # 最晚服务开始时间差
        new_edge_features[self.num_wait_time_dummy_node:, 5] = self.wait_times
        # 最晚服务结束时间差
        new_edge_features[:self.num_wait_time_dummy_node, 6] = self.wait_times
        new_edge_features[self.num_wait_time_dummy_node:, 6] = -self.wait_times
        # 更新图的边特征
        existing_edge_features[-len(src_nodes):, :] = new_edge_features
        g.edata['edge_features'] = existing_edge_features

    def encode(self, batch_customer_data, batch_company_data, num_customers, wait_times):
        batch_size = len(batch_customer_data)
        self.num_customers = num_customers
        self.wait_times = torch.tensor(wait_times, dtype=torch.float32)
        self.dummy_wait_node_index_thred = num_customers + 1
        self.num_wait_time_dummy_node = len(wait_times)

        self.batch_graphs = []
        self.batch_node_features = []
        self.batch_edge_features = []
        self.batch_distance_matrices = []
        for i in range(batch_size):
            customer_data = batch_customer_data[i]
            company_data = batch_company_data[i]
            g, node_features, edge_features, distance_matrix = build_graph(
                customer_data, company_data['depot']
            )
            self.batch_node_features.append(node_features)
            self.batch_edge_features.append(edge_features)
            self.batch_distance_matrices.append(distance_matrix)
            self.add_dummy_wait_time_nodes(g)
            self.batch_graphs.append(g)

        batch_graphs = dgl.batch(self.batch_graphs).to(self.device)
        batch_encode_node_features = self.encoder(batch_graphs)  # (total_num_nodes, embedding_dims)
        # 获取每个图的节点数量
        self.batch_num_nodes = [g.number_of_nodes() for g in self.batch_graphs]
        # 将节点特征重新拆分成每个图的特征
        self.batch_encode_node_features = torch.split(batch_encode_node_features, self.batch_num_nodes)
        # 将分割后的特征拼接为形状为 (batch_size, num_nodes, embedding_dims)
        self.batch_encode_node_features = torch.stack(self.batch_encode_node_features)
        # 计算全局embedding (batch_size, embedding_dims)
        self.batch_global_embedding = torch.mean(self.batch_encode_node_features, dim=1)

    def get_current_batch_state(
            self,
            batch_remaining_capacities,
            batch_time_elapsed,
            batch_vehicle_positions,
            batch_current_customer_demands,
            batch_customer_time_windows
    ):
        """
        获取batch中所有instances的所有车辆和客户的当前状态
        Args:
            batch_remaining_capacities: 每辆车的剩余容量
            batch_time_elapsed: 每辆车的经过时间
            batch_vehicle_positions: 车辆的位置
            batch_current_customer_demands: 当前客户的需求
            batch_customer_time_windows: 客户的时间窗
        Returns:
            current_state: 当前的状态，包括每辆车和未完成客户的信息
        """
        batch_size = len(batch_remaining_capacities)
        tensor_dtype = self.batch_global_embedding.dtype

        batch_remaining_capacities_tensor_list = [
            torch.tensor(arr, dtype=tensor_dtype, device=self.device)
            for arr in batch_remaining_capacities
        ]
        batch_remaining_capacities_tensor = torch.nn.utils.rnn.pad_sequence(
            batch_remaining_capacities_tensor_list, batch_first=True, padding_value=0.0
        )

        batch_time_elapsed_tensor_list = [
            torch.tensor(arr, dtype=tensor_dtype, device=self.device)
            for arr in batch_time_elapsed
        ]
        batch_time_elapsed_tensor = torch.nn.utils.rnn.pad_sequence(
            batch_time_elapsed_tensor_list, batch_first=True, padding_value=0.0
        )

        batch_vehicle_positions_tensor_list = [
            torch.tensor(arr, dtype=torch.int64, device=self.device)
            for arr in batch_vehicle_positions
        ]
        batch_vehicle_positions_tensor = torch.nn.utils.rnn.pad_sequence(
            batch_vehicle_positions_tensor_list, batch_first=True, padding_value=0
        )

        # (1) (batch_size, 1 + M, embedding_dims + 2)
        # 1.1 Global embedding part: shape (batch_size, 1, embedding_dims + 2)
        global_remaining_capacity = batch_remaining_capacities_tensor.sum(dim=1, keepdim=True)  # shape (batch_size, 1)
        global_time_elapsed = batch_time_elapsed_tensor.mean(dim=1, keepdim=True)  # shape (batch_size, 1)
        global_context = torch.cat((global_remaining_capacity, global_time_elapsed), dim=1)  # shape (batch_size, 2)
        global_embedding = torch.cat((
            self.batch_global_embedding.unsqueeze(1), global_context.unsqueeze(1)
        ), dim=2)  # shape (batch_size, 1, embedding_dims + 2)

        # 1.2 Vehicles embedding part: shape (batch_size, num_vehicles, embedding_dims + 2)
        vehicle_node_embeddings = torch.gather(
            self.batch_encode_node_features,
            1,
            batch_vehicle_positions_tensor.unsqueeze(-1).expand(-1, -1, self.batch_encode_node_features.size(-1))
        )  # shape (batch_size, num_vehicles, embedding_dims)
        vehicle_context = torch.cat((
            batch_remaining_capacities_tensor.unsqueeze(-1), batch_time_elapsed_tensor.unsqueeze(-1)
        ), dim=2)  # shape (batch_size, num_vehicles, 2)
        vehicle_embeddings = torch.cat((
            vehicle_node_embeddings, vehicle_context
        ), dim=2)  # shape (batch_size, num_vehicles, embedding_dims + 2)

        # Combine global and vehicles embeddings: shape (batch_size, 1 + num_vehicles, embedding_dims + 2)
        current_vehicle_embeddings = torch.cat((
            global_embedding, vehicle_embeddings
        ), dim=1)

        # (2) (batch_size, N, embedding_dims + 2)
        # Customer positions embeddings
        customer_demands = torch.tensor(
            batch_current_customer_demands, dtype=tensor_dtype, device=self.device
        ).unsqueeze(-1)  # shape (batch_size, num_positions, 1)
        customer_time_windows_start = torch.tensor(
            batch_customer_time_windows[:, :, 0], dtype=tensor_dtype, device=self.device
        ).unsqueeze(-1)  # shape (batch_size, num_positions, 1)
        customer_context = torch.cat((
            customer_demands, customer_time_windows_start
        ), dim=2)  # shape (batch_size, num_customers, 2)
        wait_time_node_context = torch.zeros(
            batch_size, self.num_wait_time_dummy_node, 2,
            dtype=tensor_dtype, device=self.device
        )  # shape (batch_size, num_wait_dummy_nodes, 2)
        customer_new_context = torch.cat((
            customer_context, wait_time_node_context
        ), dim=1)  # shape (batch_size, num_positions, 2)
        current_customer_embeddings = torch.cat((
            self.batch_encode_node_features, customer_new_context
        ), dim=2)  # shape (batch_size, num_positions, embedding_dims + 2)

        return current_vehicle_embeddings, current_customer_embeddings
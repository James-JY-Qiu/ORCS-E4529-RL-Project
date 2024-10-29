import dgl
import torch
from torch import nn

from Encoder import build_graph


class Encoder(nn.Module):
    def __init__(
            self,
            encoder_model,
            encoder_params,
            k_distance_nearest_neighbors_percent,
            k_time_nearest_neighbors_percent,
            device
    ):
        """
        :param encoder_model: 编码器模型
        :param encoder_params: 编码器参数
        :k_num_nearest_neighbors: 节点的最近邻节点数量
        :param device: 设备
        """
        super(Encoder, self).__init__()
        self.encoder = encoder_model(**encoder_params)
        self.k_distance_nearest_neighbors_percent = k_distance_nearest_neighbors_percent
        self.k_time_nearest_neighbors_percent = k_time_nearest_neighbors_percent
        self.device = device

        # 所有存储变量
        self.num_wait_time_dummy_node = None
        self.batch_graphs = None
        self.batch_node_features = None
        self.batch_edge_features = None
        self.batch_distance_matrices = None
        self.batch_encode_node_features = None
        self.batch_global_embedding = None
        self.batch_num_nodes = None

    def encode(self, batch_customer_data, batch_company_data, wait_times):
        batch_size = len(batch_customer_data)

        self.num_wait_time_dummy_node = len(wait_times)
        self.batch_graphs = []
        self.batch_node_features = []
        self.batch_edge_features = []
        self.batch_distance_matrices = []
        for i in range(batch_size):
            customer_data = batch_customer_data[i]
            company_data = batch_company_data[i]
            g, node_features, edge_features, distance_matrix = build_graph(
                customer_data, company_data, wait_times,
                self.k_distance_nearest_neighbors_percent, self.k_time_nearest_neighbors_percent,
            )
            self.batch_node_features.append(node_features)
            self.batch_edge_features.append(edge_features)
            self.batch_distance_matrices.append(distance_matrix)
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
            batch_vehicle_positions,
            batch_remaining_capacities,
            batch_time_elapsed,
            batch_customer_max_time,
            batch_customer_remaining_demands,
            include_global=True
    ):
        """
        获取batch中所有instances的所有车辆和客户的当前状态
        Args:
            batch_vehicle_positions: 车辆的位置
            batch_remaining_capacities: 车辆的剩余容量
            batch_time_elapsed: 车辆的已用时间
            batch_customer_max_time: 客户的最大时间
            batch_customer_remaining_demands: 客户的剩余需求
            include_global: 是否包含全局静态信息
        Returns:
            current_state: 当前的状态，包括每辆车和未完成客户的信息
        """
        batch_size = len(batch_vehicle_positions)
        tensor_dtype = self.batch_global_embedding.dtype

        batch_vehicle_positions_tensor = torch.tensor(batch_vehicle_positions, dtype=torch.int64, device=self.device)
        batch_remaining_capacities_tensor = torch.tensor(batch_remaining_capacities, dtype=tensor_dtype, device=self.device)
        batch_time_elapsed_tensor = torch.tensor(batch_time_elapsed, dtype=tensor_dtype, device=self.device)
        batch_customer_max_time_tensor = torch.tensor(batch_customer_max_time, dtype=tensor_dtype, device=self.device)

        # 1.1 Vehicles embedding part: shape (batch_size, num_vehicles, embedding_dims + 2)
        vehicle_node_embeddings = torch.gather(
            self.batch_encode_node_features,
            1,
            batch_vehicle_positions_tensor.unsqueeze(-1).expand(-1, -1, self.batch_encode_node_features.size(-1))
        )  # shape (batch_size, num_vehicles, embedding_dims)
        vehicle_context = torch.cat((
            batch_remaining_capacities_tensor.unsqueeze(-1), batch_time_elapsed_tensor.unsqueeze(-1)
        ), dim=-1)  # shape (batch_size, num_vehicles, 2)
        vehicle_embeddings = torch.cat((
            vehicle_node_embeddings, vehicle_context
        ), dim=-1)  # shape (batch_size, num_vehicles, embedding_dims + 2)

        # 1.2 Combine global and vehicles embeddings: shape (batch_size, 1 + num_vehicles, embedding_dims + 2)
        if include_global:
            # 1.1 Global embedding part: shape (batch_size, 1, embedding_dims + 2)
            global_remaining_capacity = batch_remaining_capacities_tensor.sum(dim=1, keepdim=True)  # shape (batch_size, 1)
            global_context = torch.cat((
                global_remaining_capacity, batch_customer_max_time_tensor.unsqueeze(-1)), dim=-1
            )  # shape (batch_size, 2)
            global_embedding = torch.cat((
                self.batch_global_embedding.unsqueeze(1), global_context.unsqueeze(1)
            ), dim=2)  # shape (batch_size, 1, embedding_dims + 2)
            current_vehicle_embeddings = torch.cat((
                global_embedding, vehicle_embeddings
            ), dim=1)
        else:
            current_vehicle_embeddings = vehicle_embeddings  # shape (batch_size, num_vehicles, embedding_dims + 2)

        # (2) (batch_size, N, embedding_dims + 1)
        # Customer positions embeddings
        customer_demands = torch.tensor(
            batch_customer_remaining_demands, dtype=tensor_dtype, device=self.device
        ).unsqueeze(-1)  # shape (batch_size, num_positions, 1)
        wait_time_node_context = torch.zeros(
            batch_size, self.num_wait_time_dummy_node, 1,
            dtype=tensor_dtype, device=self.device
        )  # shape (batch_size, num_wait_dummy_nodes, 1)
        customer_new_context = torch.cat((
            customer_demands, wait_time_node_context
        ), dim=1)  # shape (batch_size, num_positions, 1)
        current_customer_embeddings = torch.cat((
            self.batch_encode_node_features, customer_new_context
        ), dim=2)  # shape (batch_size, num_positions, embedding_dims + 1)

        return current_vehicle_embeddings, current_customer_embeddings

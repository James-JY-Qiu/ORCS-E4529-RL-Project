import dgl
import numpy as np
import torch
from torch import nn

from Encoder import build_graph


class Encoder(nn.Module):
    def __init__(
            self,
            encoder_model,
            encoder_params,
            k_distance_nearest_neighbors,
            k_time_nearest_neighbors,
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
        self.k_distance_nearest_neighbors = k_distance_nearest_neighbors
        self.k_time_nearest_neighbors = k_time_nearest_neighbors
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
                customer_data, company_data, wait_times,
                self.k_distance_nearest_neighbors, self.k_time_nearest_neighbors,
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
    ):
        """
        获取batch中所有instances的所有车辆和客户的当前状态
        Args:
            batch_vehicle_positions: 车辆的位置
        Returns:
            current_state: 当前的状态，包括每辆车和未完成客户的信息
        """
        batch_vehicle_positions_tensor = torch.tensor(
            batch_vehicle_positions, dtype=torch.int64, device=self.device
        )
        # State Embedding (batch_size, 1 + M, embedding_dims)
        # 1.1 Vehicles embedding part (batch_size, M, embedding_dims)
        vehicle_node_embeddings = torch.gather(
            self.batch_encode_node_features,
            1,
            batch_vehicle_positions_tensor.unsqueeze(-1).expand(-1, -1, self.batch_encode_node_features.size(-1))
        )
        # 1.2 Combine global and vehicles embeddings: shape (batch_size, 1 + M, embedding_dims)
        current_vehicle_embeddings = torch.cat((
            self.batch_global_embedding.unsqueeze(1), vehicle_node_embeddings
        ), dim=1)

        return current_vehicle_embeddings, self.batch_encode_node_features
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
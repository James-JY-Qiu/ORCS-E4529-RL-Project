import torch
import dgl
import numpy as np
from scipy.spatial.distance import cdist


def build_graph(df_customers, depot):
    # 计算时间窗口的长度
    df_customers['Window_length'] = df_customers['End_Time_Window'] - df_customers['Start_Time_Window']

    # 计算极角 polar angle
    depot_x, depot_y = depot[0][0], depot[0][1]
    df_customers['Polar_angle'] = np.arctan2(df_customers['Y'] - depot_y, df_customers['X'] - depot_x)

    # 构建节点特征矩阵
    node_features = df_customers[
        ['X', 'Y', 'Demand', 'Start_Time_Window', 'End_Time_Window', 'Window_length', 'Is_customer', 'Alpha', 'Beta',
         'Service_Time', 'Polar_angle']].values
    node_features_tensor = torch.tensor(node_features, dtype=torch.float32)

    # 构建图
    g = dgl.knn_graph(node_features_tensor[:, :2], k=node_features_tensor.shape[0] - 1)

    # 添加节点特征
    g.ndata['features'] = node_features_tensor

    # 构建欧几里得距离矩阵
    positions = df_customers[['X', 'Y']].values
    distance_matrix = cdist(positions, positions, metric='euclidean')

    # 计算最早和最晚服务时间矩阵
    start_time = df_customers['Start_Time_Window'].values
    end_time = df_customers['End_Time_Window'].values
    service_time = df_customers['Service_Time'].values
    earliest_service_time_matrix = start_time[:, None] + service_time[:, None] + distance_matrix
    latest_service_time_matrix = end_time[:, None] + service_time[:, None] + distance_matrix

    # 服务时间差（矩阵化）
    earliest_start_diff = earliest_service_time_matrix - start_time[None, :]
    earliest_end_diff = end_time[None, :] - earliest_service_time_matrix
    latest_start_diff = latest_service_time_matrix - start_time[None, :]
    latest_end_diff = end_time[None, :] - latest_service_time_matrix

    # 惩罚项（alpha和beta矩阵化）
    alpha = df_customers['Alpha'].values
    beta = df_customers['Beta'].values
    num_nodes = df_customers.shape[0]
    alpha_matrix = np.tile(alpha, (num_nodes, 1))
    beta_matrix = np.tile(beta, (num_nodes, 1))

    # 计算每条边终点相对于起点的极角
    src, dst = g.edges()
    src_positions = positions[src]  # 起点位置
    dst_positions = positions[dst]  # 终点位置

    # 计算相对坐标
    dx = dst_positions[:, 0] - src_positions[:, 0]
    dy = dst_positions[:, 1] - src_positions[:, 1]

    # 计算极角
    relative_polar_angle = np.arctan2(dy, dx)

    # 使用src和dst从各个矩阵中提取对应的边特征
    edge_distance = distance_matrix[src, dst]  # 提取对应的边距离特征
    edge_earliest_service_time = earliest_service_time_matrix[src, dst]  # 最早服务时间
    edge_latest_service_time = latest_service_time_matrix[src, dst]  # 最晚服务时间
    edge_earliest_start_diff = earliest_start_diff[src, dst]  # 最早开始时间差
    edge_earliest_end_diff = earliest_end_diff[src, dst]  # 最早结束时间差
    edge_latest_start_diff = latest_start_diff[src, dst]  # 最晚开始时间差
    edge_latest_end_diff = latest_end_diff[src, dst]  # 最晚结束时间差
    edge_alpha = alpha_matrix[src, dst]  # Alpha 惩罚
    edge_beta = beta_matrix[src, dst]  # Beta 惩罚

    # 拼接所有边特征到三维矩阵中，确保它们的形状一致
    edge_features = np.stack([
        edge_distance,  # 欧几里得距离
        edge_earliest_service_time,  # 最早服务时间
        edge_latest_service_time,  # 最晚服务时间
        edge_earliest_start_diff,  # 最早服务时间差
        edge_earliest_end_diff,  # 最早结束时间差
        edge_latest_start_diff,  # 最晚开始服务差异
        edge_latest_end_diff,  # 最晚结束服务差异
        edge_alpha,  # Alpha惩罚
        edge_beta,  # Beta惩罚
        relative_polar_angle  # 边的极角
    ], axis=-1)

    # 转换为PyTorch张量
    edge_features_tensor = torch.tensor(edge_features, dtype=torch.float32)

    # 将提取到的边特征添加到图中
    g.edata['edge_features'] = edge_features_tensor

    return g, node_features, edge_features, distance_matrix

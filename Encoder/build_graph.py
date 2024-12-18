import pandas as pd
import torch
import dgl
import numpy as np
from scipy.spatial.distance import cdist
from math import ceil


def find_k_dist_nodes(similarity_matrix, num_customers, num_wait_time_dummy_node, k):
    similarity_matrix = similarity_matrix.copy()
    # 将自己到自己的距离设为无穷大，避免选择自己
    # Set the distance from itself to itself to infinity to avoid selecting itself
    np.fill_diagonal(similarity_matrix, np.inf)

    # 对于第 [1:1+num_customers] 的节点，找到 [1:1+num_customers] 中距离它最近的 k 个节点（不包括它自身）
    # Find the k nearest nodes in [1:1+num_customers] to the node [1:1+num_customers] (excluding itself)
    sub_matrix_customers = similarity_matrix[1:1 + num_customers, 1:1 + num_customers]
    # 对每一行进行排序并取最近的 k 个
    # Sort each row and take the nearest k
    nearest_customers_indices = np.argsort(sub_matrix_customers, axis=1)[:, :k]

    # 对于第 [1+num_customers:1+num_customers+num_wait_time_dummy_node] 的节点，找到 [1:1+num_customers] 中距离它最近的 k 个节点
    # Find the k nearest nodes in [1:1+num_customers] to the node [1+num_customers:1+num_customers+num_wait_time_dummy_node]
    sub_matrix_depots_to_customers = similarity_matrix[1 + num_customers:1 + num_customers + num_wait_time_dummy_node, 1:1 + num_customers]
    # 对每一行进行排序并取最近的 k 个
    # Sort each row and take the nearest k
    nearest_depots_indices = np.argsort(sub_matrix_depots_to_customers, axis=1)[:, :k]

    return nearest_customers_indices+1, nearest_depots_indices+1


def build_graph(df_customers, company_data, wait_times, k_distance_percent, k_time_percent, node_features_only=False):
    depot = company_data['depot']
    num_customers = company_data['Num_Customers']
    k_distance = ceil(num_customers * k_distance_percent)
    k_time = ceil(num_customers * k_time_percent)
    # 添加等待时间节点
    # Add waiting time node
    num_wait_time_dummy_node = len(wait_times)
    df_customers_wait_time = pd.DataFrame({
        'X': np.full(num_wait_time_dummy_node, depot[0][0]),
        'Y': np.full(num_wait_time_dummy_node, depot[0][1]),
        'Demand': np.zeros(num_wait_time_dummy_node),
        'Start_Time_Window': np.zeros(num_wait_time_dummy_node),
        'End_Time_Window': np.full(num_wait_time_dummy_node, company_data['Max_Time']),
        'Alpha': np.zeros(num_wait_time_dummy_node),
        'Beta': np.zeros(num_wait_time_dummy_node),
        'Service_Time': wait_times,
        'Is_customer': 0
    })
    df_customers = pd.concat([df_customers, df_customers_wait_time], ignore_index=True)

    # 计算时间窗口的长度
    # Calculate the length of the time window
    df_customers['Window_length'] = df_customers['End_Time_Window'] - df_customers['Start_Time_Window']

    # 计算极角 polar angle
    # Calculate the polar angle
    depot_x, depot_y = depot[0][0], depot[0][1]
    df_customers['Polar_angle'] = np.arctan2(df_customers['Y'] - depot_y, df_customers['X'] - depot_x)

    # 构建节点特征矩阵
    # Build node feature matrix
    node_features = df_customers[
        ['X', 'Y', 'Demand', 'Start_Time_Window', 'End_Time_Window', 'Window_length', 'Is_customer', 'Alpha', 'Beta', 'Service_Time', 'Polar_angle']
    ].values
    node_features_tensor = torch.tensor(node_features, dtype=torch.float32)

    # 构建欧几里得距离矩阵
    # Build Euclidean distance matrix
    positions = df_customers[['X', 'Y']].values
    distance_matrix = cdist(positions, positions, metric='euclidean')

    if node_features_only:
        return node_features_tensor, distance_matrix

    # 构建图
    # Build graph
    g = dgl.graph(([], []))  # 创建一个空图 Create an empty graph

    # 添加节点以及节点特征
    # Add nodes and node features
    g.add_nodes(node_features.shape[0])
    g.ndata['features'] = node_features_tensor

    # 根据距离矩阵计算最近邻节点
    # Calculate the nearest neighbor nodes based on the distance matrix
    nearest_dist_customer_to_customer, nearest_dist_depot_to_customer = find_k_dist_nodes(
        distance_matrix, num_customers, num_wait_time_dummy_node, k_distance
    )

    # 计算最早和最晚服务时间矩阵
    # Calculate the earliest and latest service time matrix
    start_time = df_customers['Start_Time_Window'].values
    end_time = df_customers['End_Time_Window'].values
    service_time = df_customers['Service_Time'].values
    alpha = df_customers['Alpha'].values
    beta = df_customers['Beta'].values
    earliest_service_time_matrix = start_time[:, None] + service_time[:, None] + distance_matrix
    latest_service_time_matrix = end_time[:, None] + service_time[:, None] + distance_matrix

    # 服务时间差
    # Service time difference
    earliest_start_diff = earliest_service_time_matrix - start_time[None, :]
    earliest_end_diff = end_time[None, :] - earliest_service_time_matrix
    latest_start_diff = latest_service_time_matrix - start_time[None, :]
    latest_end_diff = end_time[None, :] - latest_service_time_matrix

    # 构建时间相似度矩阵
    # Build time similarity matrix
    overlap_matrix = np.minimum(end_time[None, :], latest_service_time_matrix) - np.maximum(start_time[None, :], earliest_service_time_matrix)
    overlap_matrix[overlap_matrix < 0] = 0
    early_no_overlap = (start_time[None, :] - latest_service_time_matrix) * alpha[None, :]
    early_no_overlap[early_no_overlap < 0] = 0
    late_no_overlap = (earliest_service_time_matrix - end_time[None, :]) * beta[None, :]
    late_no_overlap[late_no_overlap < 0] = 0
    time_similarity_matrix = -(overlap_matrix - early_no_overlap - late_no_overlap)

    nearest_time_customer_to_customer, nearest_time_depot_to_customer = find_k_dist_nodes(
        time_similarity_matrix, num_customers, num_wait_time_dummy_node, k_time
    )

    # 添加边
    # Add edges
    src = [0] * (node_features.shape[0] - 1) \
          + [dummy_depot for dummy_depot in range(num_customers + 1, num_customers + 1 + num_wait_time_dummy_node) for _ in range(k_distance)] \
          + [dummy_depot for dummy_depot in range(num_customers + 1, num_customers + 1 + num_wait_time_dummy_node) for _ in range(k_time)] \
          + [customer for customer in range(1, num_customers + 1) for _ in range(k_distance)] \
          + [customer for customer in range(1, num_customers + 1) for _ in range(k_time)] \
          + [customer for customer in range(1, num_customers + 1)]
    dst = list(range(1, node_features.shape[0])) \
          + nearest_dist_depot_to_customer.flatten().tolist() \
          + nearest_time_depot_to_customer.flatten().tolist() \
          + nearest_dist_customer_to_customer.flatten().tolist() \
          + nearest_time_customer_to_customer.flatten().tolist() \
          + [0] * num_customers
    # 使用集合去重，集合中的每一条边用 (src, dst) 的形式表示
    # Use set to deduplicate, each edge in the set is represented in the form of (src, dst)
    edges = set(zip(src, dst))
    # 解压去重后的边
    # Unzip deduplicated edges
    src_unique, dst_unique = zip(*edges)
    g.add_edges(src_unique, dst_unique)

    # 惩罚项（alpha和beta矩阵化）
    # Penalty term (alpha and beta matrix)
    alpha = df_customers['Alpha'].values
    beta = df_customers['Beta'].values
    num_nodes = df_customers.shape[0]
    alpha_matrix = np.tile(alpha, (num_nodes, 1))
    beta_matrix = np.tile(beta, (num_nodes, 1))

    # 计算每条边终点相对于起点的极角
    # Calculate the polar angle of the end point relative to the starting point of each edge
    src, dst = g.edges()
    src_positions = positions[src]  # 起点位置 Starting point position
    dst_positions = positions[dst]  # 终点位置 End point position

    # 计算相对坐标
    # Calculate relative coordinates
    dx = dst_positions[:, 0] - src_positions[:, 0]
    dy = dst_positions[:, 1] - src_positions[:, 1]

    # 计算极角
    # Calculate the polar angle
    relative_polar_angle = np.arctan2(dy, dx)

    # 使用src和dst从各个矩阵中提取对应的边特征
    # Extract the corresponding edge features from each matrix using src and dst
    edge_distance = distance_matrix[src, dst]  # 提取对应的边距离特征 Extract the corresponding edge distance feature
    edge_earliest_service_time = earliest_service_time_matrix[src, dst]  # 最早服务时间 Earliest service time
    edge_latest_service_time = latest_service_time_matrix[src, dst]  # 最晚服务时间 Latest service time
    edge_earliest_start_diff = earliest_start_diff[src, dst]  # 最早开始时间差 Earliest start time difference
    edge_earliest_end_diff = earliest_end_diff[src, dst]  # 最早结束时间差 Earliest end time difference
    edge_latest_start_diff = latest_start_diff[src, dst]  # 最晚开始时间差 Latest start time difference
    edge_latest_end_diff = latest_end_diff[src, dst]  # 最晚结束时间差 Latest end time difference
    edge_alpha = alpha_matrix[src, dst]  # Alpha 惩罚 Alpha penalty
    edge_beta = beta_matrix[src, dst]  # Beta 惩罚 Beta penalty
    edge_similarity = time_similarity_matrix[src, dst]  # 时间相似度 Time similarity

    # 拼接所有边特征到三维矩阵中，确保它们的形状一致
    # Concatenate all edge features into a three-dimensional matrix to ensure that their shapes are consistent
    edge_features = np.stack([
        edge_distance,  # 欧几里得距离 Euclidean distance
        edge_earliest_service_time,  # 最早服务时间 Earliest service time
        edge_latest_service_time,  # 最晚服务时间 Latest service time
        edge_earliest_start_diff,  # 最早服务时间差 Earliest service time difference
        edge_earliest_end_diff,  # 最早结束时间差 Earliest end time difference
        edge_latest_start_diff,  # 最晚开始服务差异 Latest start service difference
        edge_latest_end_diff,  # 最晚结束服务差异 Latest end service difference
        edge_alpha,  # Alpha惩罚 Alpha penalty
        edge_beta,  # Beta惩罚 Beta penalty
        relative_polar_angle,  # 边的极角 Polar angle of the edge
        edge_similarity  # 时间相似度 Time similarity
    ], axis=-1)

    # 转换为PyTorch张量
    # Convert to PyTorch tensor
    edge_features_tensor = torch.tensor(edge_features, dtype=torch.float32)

    # 将提取到的边特征添加到图中
    # Add the extracted edge features to the graph
    g.edata['edge_features'] = edge_features_tensor

    return g, node_features, edge_features, distance_matrix

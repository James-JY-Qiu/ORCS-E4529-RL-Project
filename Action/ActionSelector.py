import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class ActionSelector(nn.Module):
    def __init__(self, embedding_dim, heads, dynamic_vehicle_dim, dynamic_customer_dim):
        """
        初始化ActionSelector，用于计算动作选择
        Args:
            embedding_dim: 节点嵌入的维度
            heads: 多头注意力机制的头数
            dynamic_vehicle_dim: 动态车辆嵌入的维度
            dynamic_customer_dim: 动态顾客嵌入的维度
        """
        super(ActionSelector, self).__init__()
        self.embedding_dim = embedding_dim
        self.heads = heads
        self.dynamic_vehicle_dim = dynamic_vehicle_dim

        # 车辆全连接层，用于车辆维度改变
        self.fc_vehicle = nn.Linear(embedding_dim + dynamic_vehicle_dim, embedding_dim)
        # 顾客全连接层，用于顾客维度改变
        self.fc_customer = nn.Linear(embedding_dim + dynamic_customer_dim, embedding_dim)

        # 多头注意力层，用于计算车辆自注意力
        self.mha_vehicles = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=heads, batch_first=True)
        # 多头注意力层，用于计算车辆-环境交叉注意力
        self.mha_vehicles_env = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=heads, batch_first=True)

        # 查询权重矩阵 W_Q 和 键权重矩阵 W_K
        self.W_Q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W_K = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.sqrt_d_k = math.sqrt(embedding_dim)

    def compute_batch_compatibility(self, vehicle_embeddings, action_embeddings):
        """
        计算每个客户与车辆的兼容性分数
        Args:
            vehicle_embeddings: 车辆嵌入矩阵 (batch_size, M, embedding_dim)
            action_embeddings: 动作嵌入矩阵 (batch_size, N, embedding_dim)
        Returns:
            compatibility_scores: 兼容性分数 (batch_size, M, N)
        """
        # 计算查询向量 q_c
        q_c = self.W_Q(vehicle_embeddings)  # (batch_size, M, embedding_dim)
        # 计算键向量 k_i
        k = self.W_K(action_embeddings)  # (batch_size, N, embedding_dim)

        # 使用 q_c 和 k 计算兼容性 u_{i,m,t}
        # 使用 einsum 来计算兼容性分数，等效于 torch.matmul(q_c, k.transpose(-1, -2))
        compatibility_scores = torch.einsum('bme,bne->bmn', q_c, k) / self.sqrt_d_k

        return compatibility_scores  # 返回 (batch_size, M, N) 的相似度矩阵

    def apply_batch_mask(self, compatibility_scores, neg_inf_mask):
        """
        对已经访问过的客户和容量不足的客户应用掩码
        Args:
            compatibility_scores: 兼容性分数矩阵 (batch_size, M, N)
            neg_inf_mask: -inf mask (batch_size, M, N)
        Returns:
            masked_scores: 应用掩码后的兼容性分数
        """
        masked_scores = compatibility_scores + neg_inf_mask
        return masked_scores

    def compute_batch_action_probabilities(self, masked_scores):
        """
        通过softmax计算动作概率
        Args:
            masked_scores: 应用掩码后的兼容性分数 (batch_size, M, N)
        Returns:
            action_probs: 动作概率矩阵 (batch_size, M, N)
        """
        # 对每辆车的分数进行 softmax
        action_probs = F.softmax(masked_scores, dim=2)
        return action_probs

    def select_batch_action(self, action_probs, mode):
        """
        根据计算出的概率选择动作
        Args:
            action_probs: 动作概率矩阵 (batch_size, M, N)
            mode: 动作选择的模式，'greedy' 或 'sampling'
            depot_index: depot的索引
        Returns:
            selected_actions: 选择的客户索引 (batch_size, M)
            log_probs: 对数概率 (在sampling模式下)
        """
        log_probs = None
        if mode == 'greedy':
            # 选择概率最大的客户索引
            selected_actions = torch.argmax(action_probs, dim=2)
        elif mode == 'sampling':
            # 通过概率进行抽样选择
            # 首先将 action_probs reshape 成 (batch_size * M, N)
            batch_size, M, N = action_probs.shape
            action_probs_flat = action_probs.view(batch_size * M, N)  # (batch_size * M, N)
            # 从 (batch_size * M, N) 的概率分布中采样 num_samples=1
            sampled_actions = torch.multinomial(action_probs_flat, num_samples=1)  # (batch_size * M, 1)
            # 还原回 (batch_size, M) 的形状
            selected_actions = sampled_actions.view(batch_size, M)  # (batch_size, M)
            # 计算采样动作的对数概率
            prob_mask = torch.zeros_like(action_probs_flat)
            prob_mask.scatter_(1, sampled_actions, 1)
            selected_probs = (action_probs_flat * prob_mask).sum(dim=1, keepdim=True).view(batch_size, M)
            log_probs = torch.log(selected_probs)  # (batch_size, M)
        else:
            raise ValueError("Mode must be 'greedy' or 'sampling'")

        return selected_actions, log_probs

    def forward(self, batch_vehicle_state_embeddings, batch_customer_state_embeddings, neg_inf_mask, mode='greedy'):
        """
        处理动作选择的完整流程
        Args:
            batch_vehicle_state_embeddings: 车辆当前状态嵌入矩阵 (batch_size, M, embedding_dim)
            batch_customer_state_embeddings: 顾客当前状态嵌入矩阵 (batch_size, N, embedding_dim)
            neg_inf_mask: -inf mask (batch_size, M, N)
            mode: 选择动作的模式，'greedy' 或 'sampling'
        Returns:
            selected_actions: 选择的客户索引 (batch_size, M)
            log_probs: 对数概率 (在sampling模式下) (batch_size, M)
        """
        # 0. 维度改变
        batch_vehicle_state_embeddings = self.fc_vehicle(batch_vehicle_state_embeddings)
        batch_customer_state_embeddings = self.fc_customer(batch_customer_state_embeddings)

        # 1. 使用多头注意力机制计算 new_batch_vehicle_status_embeddings
        vehicle_self_attn, _ = self.mha_vehicles(
            batch_vehicle_state_embeddings,
            batch_vehicle_state_embeddings,
            batch_vehicle_state_embeddings
        )  # (batch_size, M, embedding_dim)

        # 2. 使用多头注意力机制计算车辆-环境交叉注意力
        vehicle_env_attn, _ = self.mha_vehicles_env(
            vehicle_self_attn,
            batch_customer_state_embeddings,
            batch_customer_state_embeddings
        )  # (batch_size, M, embedding_dim)

        # 3. 计算车辆嵌入
        vehicle_embeddings = vehicle_self_attn + vehicle_env_attn

        # 4. 计算兼容性分数
        compatibility_scores = self.compute_batch_compatibility(
            vehicle_embeddings, batch_customer_state_embeddings
        )

        # 5. 对没有需求的客户进行掩码处理
        masked_scores = self.apply_batch_mask(compatibility_scores, neg_inf_mask)

        # 6. 通过 softmax 计算动作概率
        action_probs = self.compute_batch_action_probabilities(masked_scores)

        # 7. 根据模式选择动作
        selected_actions, log_probs = self.select_batch_action(action_probs, mode)

        return selected_actions, log_probs

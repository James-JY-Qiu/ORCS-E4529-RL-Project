import math

import torch
import torch.nn as nn


class CriticEstimator(nn.Module):
    def __init__(self, num_agents, state_dim, action_dim, embedding_dim, num_heads, activation, device):
        super(CriticEstimator, self).__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim
        assert embedding_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.activation = activation

        input_dim = state_dim + action_dim

        self.g_weight = nn.Parameter(torch.randn(num_agents, input_dim, embedding_dim))
        self.g_bias = nn.Parameter(torch.randn(num_agents, embedding_dim))

        self.W_Q_share = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W_K_share = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.V_share = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.sqrt_d_k_share = math.sqrt(self.head_dim)

        self.gx_weight_1 = nn.Parameter(torch.randn(num_agents, embedding_dim * 2, embedding_dim))
        self.gx_bias_1 = nn.Parameter(torch.randn(num_agents, embedding_dim))
        self.gx_weight_2 = nn.Parameter(torch.randn(num_agents, embedding_dim, 1))
        self.gx_bias_2 = nn.Parameter(torch.randn(num_agents, 1))

        self.g_o_weight = nn.Parameter(torch.randn(num_agents, state_dim, embedding_dim))
        self.g_o_bias = nn.Parameter(torch.randn(num_agents, embedding_dim))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.g_weight)
        torch.nn.init.zeros_(self.g_bias)
        torch.nn.init.kaiming_uniform_(self.W_Q_share.weight)
        torch.nn.init.kaiming_uniform_(self.W_K_share.weight)
        torch.nn.init.kaiming_uniform_(self.V_share.weight)
        torch.nn.init.kaiming_uniform_(self.gx_weight_1)
        torch.nn.init.zeros_(self.gx_bias_1)
        torch.nn.init.kaiming_uniform_(self.gx_weight_2)
        torch.nn.init.zeros_(self.gx_bias_2)
        torch.nn.init.kaiming_uniform_(self.g_o_weight)
        torch.nn.init.zeros_(self.g_o_bias)

    def split_heads(self, x):
        # x: [batch_size, num_agents, embedding_dim] -> [batch_size, num_agents, num_heads, head_dim]
        batch_size = x.size(0)
        x = x.view(batch_size, self.num_agents, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)  # [batch_size, num_heads, num_agents, head_dim]

    def calculate_e(self, state, action, baseline_agent):
        if baseline_agent is not None:
            other_agents_state = torch.cat([state[:, :baseline_agent], state[:, baseline_agent + 1:]], dim=1)
            other_agents_action = torch.cat([action[:, :baseline_agent], action[:, baseline_agent + 1:]], dim=1)
            other_agents_x = torch.cat([other_agents_state, other_agents_action], dim=-1)
            other_agents_g_weight = torch.cat([self.g_weight[:baseline_agent], self.g_weight[baseline_agent + 1:]], dim=0)
            other_agents_g_bias = torch.cat([self.g_bias[:baseline_agent], self.g_bias[baseline_agent + 1:]], dim=0)
            other_agents_e = torch.einsum('bni,nio->bno', other_agents_x, other_agents_g_weight) + other_agents_g_bias  # [batch_size, num_agents-1, embedding_dim]
            agent_e = torch.einsum('bni,nio->bno', state[:, baseline_agent].unsqueeze(1), self.g_o_weight[baseline_agent].unsqueeze(0)) + self.g_o_bias[baseline_agent]  # [batch_size, 1, embedding_dim]
            e = torch.cat([other_agents_e[:, :baseline_agent], agent_e, other_agents_e[:, baseline_agent:]], dim=1)
        else:
            x = torch.cat([state, action], dim=-1)  # (batch_size, num_agents, state_dim + action_dim)
            e = torch.einsum('bni,nio->bno', x, self.g_weight) + self.g_bias  # [batch_size, num_agents, embedding_dim]
        return e

    def calculate_score(self, e):
        Q = self.split_heads(self.W_Q_share(e))  # (batch_size, num_heads, num_agents, head_dim)
        K = self.split_heads(self.W_K_share(e))  # (batch_size, num_heads, num_agents, head_dim)

        attention = torch.einsum('bhqd,bhkd->bhqk', Q, K) / self.sqrt_d_k_share
        score = torch.softmax(attention, dim=-1)  # (batch_size, num_heads, num_agents, num_agents)
        return score

    def calculate_v(self, e):
        v = self.split_heads(self.V_share(e))  # (batch_size, num_heads, num_agents, head_dim)
        v = self.activation(v)
        return v

    def calculate_x(self, score, v):
        x = torch.einsum('bhnm,bhnd->bhnd', score, v)  # (batch_size, num_heads, num_agents, head_dim)
        # Concatenate heads back together
        x = x.permute(0, 2, 1, 3).contiguous()  # [batch_size, num_agents, num_heads, head_dim]
        batch_size = x.size(0)
        x = x.view(batch_size, self.num_agents, self.embedding_dim)  # Combine all heads

        return x  # (batch_size, num_agents, embedding_dim)

    def forward(self, state, action, baseline_agent=None):
        e = self.calculate_e(state, action, baseline_agent)
        score = self.calculate_score(e)
        v = self.calculate_v(e)
        x = self.calculate_x(score, v)

        ex = torch.cat([e, x], dim=-1)

        fc_1 = torch.einsum('bni,nio->bno', ex, self.gx_weight_1) + self.gx_bias_1
        fc_1 = self.activation(fc_1)
        Q = torch.einsum('bni,nio->bno', fc_1, self.gx_weight_2) + self.gx_bias_2  # (batch_size, num_agents, 1)
        return Q

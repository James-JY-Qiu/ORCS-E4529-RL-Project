import copy
import traceback
from datetime import datetime
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

import wandb
from torch.optim import Adam

from Action import ActionSelector, CriticEstimator
from Encoder import Encoder, MultiLayerEdgeGAT
from Env import BatchVRPEnvs

from train_utils import return_batch_neg_inf_masks, batch_steps, env_encode, record_gradients, soft_update_target_model

from params import k_distance_nearest_neighbors, k_time_nearest_neighbors
from params import small_params, num_samll_instances
from params import project_name, device, max_workers, record_gradient, reward_window_size



# --------------------- Hyperparameters ---------------------
# --------- encoder -----------

# MultiLayerEdge
out_feats = 128
MultiLayerEdgeGATParams = {
    'in_feats': 11,
    'edge_feats': 10,
    'units': 128,
    'num_heads': 8,
    'num_layers': 2,
    'feat_drop': 0.0,
    'attn_drop': 0.0,
    'edge_drop': 0.0,
    'activation': F.leaky_relu
}
embedding_dim = out_feats
# --------- decoder -----------

# action
action_heads = 8
dynamic_vehicle_dim = 2
dynamic_customer_dim = 1

# critic
num_agents = 2
critic_heads = 8
alpha = 0.1
gamma = 0.0
tau = 1e-3
critic_activation = F.leaky_relu

# train
epochs = 100

# optimizer
actor_rl = 1e-3
critic_lr = 1e-3

# replay buffer
replay_buffer_batch_size = 128
replay_buffer_capacity = 50000

# logger
log_interval = 10


Experience = namedtuple('Experience', ['epoch_id', 'batch_id', 'instance_id', 'state', 'action', 'reward', 'next_state'])


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)  # 使用 deque 作为循环队列，达到容量后自动移除旧样本
        self.capacity = capacity

    def add(self, experience):
        self.buffer.append(experience)  # 添加新经验

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)  # 随机抽样一批经验

    def size(self):
        return len(self.buffer)  # 当前缓冲区中的经验样本数

    def get_first(self):
        if len(self.buffer) > 0:
            return self.buffer[0]  # 获取最早添加的样本
        else:
            return None  # 如果缓冲区为空，返回 None


def update_buffer(replay_buffer, last_batch_status, actions, current_batch_status, agent_rewards, batch_size, epoch_id, batch_id):
    agent_rewards = agent_rewards.detach().cpu().numpy()
    for i in range(batch_size):
        state = (
            last_batch_status['batch_vehicle_positions'][i],
            last_batch_status['batch_remaining_capacities'][i],
            last_batch_status['batch_time_elapsed'][i],
            last_batch_status['batch_customer_max_time'][i],
            last_batch_status['batch_customer_remaining_demands'][i],
            last_batch_status['batch_neg_inf_mask'][i],
        )
        action = actions[i]
        reward = agent_rewards[i].sum()
        next_state = (
            current_batch_status['batch_vehicle_positions'][i],
            current_batch_status['batch_remaining_capacities'][i],
            current_batch_status['batch_time_elapsed'][i],
            current_batch_status['batch_customer_max_time'][i],
            current_batch_status['batch_customer_remaining_demands'][i],
            current_batch_status['batch_neg_inf_mask'][i],
        )
        experience = Experience(epoch_id, batch_id, i, state, action, reward, next_state)
        replay_buffer.add(experience)


def update_critics(
        env, encoder, num_agents, critic_estimator, target_critic_estimator, critic_optimizer,
        alpha, gamma, tau, action_selector, replay_buffer, buffer_batch_size,
        history_data, device, record_gradient
):
    # epoch_id, batch_id, instance_id, state, action, reward, next_state
    sampled_experiences = replay_buffer.sample(buffer_batch_size)
    batch_customer_data = [
        history_data[(experience.epoch_id, experience.batch_id, experience.instance_id)][0]
        for experience in sampled_experiences
    ]
    batch_company_data = [
        history_data[(experience.epoch_id, experience.batch_id, experience.instance_id)][1]
        for experience in sampled_experiences
    ]

    with torch.no_grad():
        env_encode(encoder, batch_customer_data, batch_company_data, env.num_customers, env.wait_times)

    batch_current_vehicle_positions = []
    batch_current_remaining_capacities = []
    batch_current_time_elapsed = []
    batch_current_customer_max_time = []
    batch_current_customer_remaining_demands = []
    batch_current_neg_inf_mask = []

    batch_current_actions = []

    batch_current_rewards = []

    batch_next_vehicle_positions = []
    batch_next_remaining_capacities = []
    batch_next_time_elapsed = []
    batch_next_customer_max_time = []
    batch_next_customer_remaining_demands = []
    batch_next_neg_inf_mask = []

    for i in range(buffer_batch_size):
        state = sampled_experiences[i].state
        batch_current_vehicle_positions.append(state[0])
        batch_current_remaining_capacities.append(state[1])
        batch_current_time_elapsed.append(state[2])
        batch_current_customer_max_time.append(state[3])
        batch_current_customer_remaining_demands.append(state[4])
        instance_current_neg_inf_mask = torch.tensor(state[5], dtype=torch.float, device=device)
        batch_current_neg_inf_mask.append(instance_current_neg_inf_mask)

        batch_current_actions.append(sampled_experiences[i].action)

        batch_current_rewards.append(sampled_experiences[i].reward)

        next_state = sampled_experiences[i].next_state
        batch_next_vehicle_positions.append(next_state[0])
        batch_next_remaining_capacities.append(next_state[1])
        batch_next_time_elapsed.append(next_state[2])
        batch_next_customer_max_time.append(next_state[3])
        batch_next_customer_remaining_demands.append(next_state[4])
        instance_next_neg_inf_mask = torch.tensor(next_state[5], dtype=torch.float, device=device)
        batch_next_neg_inf_mask.append(instance_next_neg_inf_mask)

    # 获取当前状态的embedding
    current_vehicle_embeddings, current_customer_embeddings = encoder.get_current_batch_state(
        batch_vehicle_positions=np.array(batch_current_vehicle_positions),
        batch_remaining_capacities=np.array(batch_current_remaining_capacities),
        batch_time_elapsed=np.array(batch_current_time_elapsed),
        batch_customer_max_time=np.array(batch_current_customer_max_time),
        batch_customer_remaining_demands=np.array(batch_current_customer_remaining_demands),
        include_global=False
    )  # (batch_size, num_agents, state_dim), (batch_size, num_customers, action_dim)

    batch_current_actions = np.array(batch_current_actions)
    batch_current_actions_tensor = torch.tensor(batch_current_actions, dtype=torch.int64, device=device)  # (batch_size, num_agents)
    current_action_embeddings = torch.gather(
        current_customer_embeddings,
        1,
        batch_current_actions_tensor.unsqueeze(-1).expand(-1, -1, current_customer_embeddings.size(-1))
    )  # (batch_size, num_agents, action_dim)

    # 计算 Q(o,a)
    Q = critic_estimator(current_vehicle_embeddings, current_action_embeddings)  # (batch_size, num_agents, 1)

    # 计算 b(o, a\i)
    agent_baseline = []
    for agent_id in range(num_agents):
        b = critic_estimator(current_vehicle_embeddings, current_action_embeddings, baseline_agent=agent_id)  # (batch_size, num_agents, 1)
        agent_baseline.append(b.mean(dim=1, keepdim=True))  # (batch_size, 1, 1)
    agent_baseline = torch.cat(agent_baseline, dim=1)  # (batch_size, num_agents, 1)

    # 获取 reward
    batch_current_rewards_tensor = torch.tensor(batch_current_rewards, dtype=torch.float, device=device)  # (batch_size)

    # 获取下一个状态的embedding
    next_vehicle_embeddings, next_customer_embeddings = encoder.get_current_batch_state(
        batch_vehicle_positions=np.array(batch_next_vehicle_positions),
        batch_remaining_capacities=np.array(batch_next_remaining_capacities),
        batch_time_elapsed=np.array(batch_next_time_elapsed),
        batch_customer_max_time=np.array(batch_next_customer_max_time),
        batch_customer_remaining_demands=np.array(batch_next_customer_remaining_demands),
        include_global=False
    )  # (batch_size, num_agents, state_dim), (batch_size, num_customers, action_dim)

    # 计算下一个状态的动作
    batch_next_neg_inf_mask = torch.stack(batch_next_neg_inf_mask, dim=0).to(device)
    with torch.no_grad():
        next_actions, next_log_probs = action_selector(
            next_vehicle_embeddings,
            next_customer_embeddings,
            batch_next_neg_inf_mask,
            mode='sampling'
        )
    next_actions_embeddings = torch.gather(
        next_customer_embeddings,
        1,
        next_actions.unsqueeze(-1).expand(-1, -1, current_customer_embeddings.size(-1))
    )  # (batch_size, num_agents, action_dim)

    # 计算 Q_target(o',a')
    Q_target = target_critic_estimator(next_vehicle_embeddings, next_actions_embeddings)  # (batch_size, num_agents, 1)

    # 计算 b_target(o', a'\i)
    agent_baseline_target = []
    for agent_id in range(num_agents):
        b = target_critic_estimator(next_vehicle_embeddings, next_actions_embeddings, baseline_agent=agent_id)  # (batch_size, num_agents, 1)
        agent_baseline_target.append(b.mean(dim=1, keepdim=True))  # (batch_size, 1, 1)
    agent_baseline_target = torch.cat(agent_baseline_target, dim=1)  # (batch_size, num_agents, 1)

    # 计算 y_Q
    y_Q = batch_current_rewards_tensor + gamma * (Q_target - alpha * next_log_probs.unsqueeze(-1))  # (batch_size, num_agents, 1)

    # 计算 critic_loss
    critic_loss = torch.mean((Q - y_Q) ** 2, dim=0).sum()

    # 计算 y_b
    y_b = batch_current_rewards_tensor + gamma * (agent_baseline_target - alpha * next_log_probs.unsqueeze(-1))  # (batch_size, num_agents, 1)

    # 计算 baseline_loss
    baseline_loss = torch.mean((agent_baseline - y_b) ** 2, dim=0).sum()

    # 更新梯度
    critic_optimizer.zero_grad()

    # 反向传播
    critic_loss.backward(retain_graph=True)
    baseline_loss.backward()

    if record_gradient:
        critic_gradients = record_gradients(critic_estimator)
    else:
        critic_gradients = None

    # 使用优化器更新参数
    critic_optimizer.step()

    # soft update target critic estimator
    soft_update_target_model(target_critic_estimator, critic_estimator, tau)

    # experience data for actor
    sampled_experience_dataset = (
        batch_customer_data,
        batch_company_data,
        batch_current_vehicle_positions,
        batch_current_remaining_capacities,
        batch_current_time_elapsed,
        batch_current_customer_max_time,
        batch_current_customer_remaining_demands,
        batch_current_neg_inf_mask
    )

    return Q, sampled_experience_dataset, critic_loss, baseline_loss, critic_gradients


def update_actor(
        env, Q, sampled_experience_dataset, encoder, action_selector, critic_estimator, actor_optimizer,
        num_agents, alpha, record_gradient, buffer_batch_size
):
    batch_customer_data = sampled_experience_dataset[0]
    batch_company_data = sampled_experience_dataset[1]

    env_encode(encoder, batch_customer_data, batch_company_data, env.num_customers, env.wait_times)

    batch_current_vehicle_positions = sampled_experience_dataset[2]
    batch_current_remaining_capacities = sampled_experience_dataset[3]
    batch_current_time_elapsed = sampled_experience_dataset[4]
    batch_current_customer_max_time = sampled_experience_dataset[5]
    batch_current_customer_remaining_demands = sampled_experience_dataset[6]
    batch_current_neg_inf_mask = sampled_experience_dataset[7]

    # 获取当前状态的embedding
    current_vehicle_embeddings, current_customer_embeddings = encoder.get_current_batch_state(
        batch_vehicle_positions=np.array(batch_current_vehicle_positions),
        batch_remaining_capacities=np.array(batch_current_remaining_capacities),
        batch_time_elapsed=np.array(batch_current_time_elapsed),
        batch_customer_max_time=np.array(batch_current_customer_max_time),
        batch_customer_remaining_demands=np.array(batch_current_customer_remaining_demands),
        include_global=True
    )  # (batch_size, num_agents, state_dim), (batch_size, num_customers, action_dim)

    # 计算当前策略下的动作和对应的概率
    batch_current_neg_inf_mask = torch.stack(batch_current_neg_inf_mask, dim=0).to(device)
    actions, log_probs = action_selector(
        current_vehicle_embeddings,
        current_customer_embeddings,
        batch_current_neg_inf_mask,
        mode='sampling',
    )
    current_action_embeddings = torch.gather(
        current_customer_embeddings,
        1,
        actions.unsqueeze(-1).expand(-1, -1, current_customer_embeddings.size(-1))
    )  # (batch_size, num_agents, action_dim)

    # 计算每个agent的基线
    agent_baseline = []
    with torch.no_grad():
        for agent_id in range(num_agents):
            b = critic_estimator(current_vehicle_embeddings[:, 1:], current_action_embeddings, baseline_agent=agent_id)  # (batch_size, num_agents, 1)
            agent_baseline.append(b.mean(dim=1, keepdim=True))  # (batch_size, 1, 1)
    agent_baseline = torch.cat(agent_baseline, dim=1)  # (batch_size, num_agents, 1)

    advantage = Q - agent_baseline  # (batch_size, num_agents, 1)
    actor_loss = torch.mean(- (advantage - alpha * log_probs.unsqueeze(-1)) * log_probs.unsqueeze(-1), dim=0).sum()

    # 更新梯度
    actor_optimizer.zero_grad()

    # 反向传播
    actor_loss.backward()
    if record_gradient:
        action_selector_gradients = record_gradients(action_selector)
        encoder_gradients = record_gradients(encoder)
    else:
        action_selector_gradients = None
        encoder_gradients = None

    # 使用优化器更新参数
    actor_optimizer.step()

    return actor_loss, action_selector_gradients, encoder_gradients


def add_history_data(history_data, epoch_id, batch_id, env, batch_size):
    for instance_id in range(batch_size):
        history_data[(epoch_id, batch_id, instance_id)] = (
            env.batch_customer_data[instance_id],
            env.batch_company_data[instance_id]
        )


def clear_history_data(buffer, history_data):
    earliest_experience = buffer.get_first()
    earliest_epoch_id, earliest_batch_id, earliest_instance_id = earliest_experience[:3]
    delete_indices = []
    for epoch_id, batch_id, instance_id in history_data.keys():
        if (epoch_id < earliest_epoch_id) or (epoch_id == earliest_epoch_id and batch_id < earliest_batch_id):
            delete_indices.append((epoch_id, batch_id, instance_id))
        else:
            break

    for index in delete_indices:
        del history_data[index]


def run_batch(
        T, env, encoder, action_selector, critic_estimator, target_critic_estimator, actor_optimizer, critic_optimizer,
        num_agents, replay_buffer, buffer_batch_size, alpha, gamma, tau,
        mode, device, epoch_id, batch_id, history_data, record_gradient,
        window_size, Q_list, critic_loss_list, critic_baseline_loss_list, actor_loss_list, log_buffer, log_interval
):
    """
    执行一次完整的 batch (即跑完一遍所有的 instances)
    """
    env.reset(generate=True)
    batch_size = env.batch_size
    # 添加历史数据
    add_history_data(history_data, epoch_id, batch_id, env, batch_size)

    # 编码环境数据
    with torch.no_grad():
        encoder.encode(
            batch_customer_data=env.batch_customer_data,
            batch_company_data=env.batch_company_data,
            num_customers=env.num_customers,
            wait_times=env.wait_times
        )
    env.update_parameters(encoder.batch_distance_matrices, encoder.batch_num_nodes)

    # 保存当前的数据以便模型更新之后重新编码
    batch_customer_data = env.batch_customer_data
    batch_company_data = env.batch_company_data

    # 记录 instance 是否结束
    instance_status = np.zeros(batch_size, dtype=bool)
    # 记录 reward
    reward_info = torch.zeros(batch_size, dtype=torch.float, device=device)
    while not instance_status.all():
        with torch.no_grad():
            encoder.encode(
                batch_customer_data=batch_customer_data,
                batch_company_data=batch_company_data,
                num_customers=env.num_customers,
                wait_times=env.wait_times
            )
        env.update_parameters(encoder.batch_distance_matrices, encoder.batch_num_nodes)

        current_batch_status = env.get_current_batch_status()
        current_vehicle_embeddings, current_customer_embeddings = encoder.get_current_batch_state(**current_batch_status)
        batch_neg_inf_mask = return_batch_neg_inf_masks(env).to(device)
        with torch.no_grad():
            actions, _ = action_selector(
                current_vehicle_embeddings,
                current_customer_embeddings,
                batch_neg_inf_mask,
                mode=mode
            )
        # 执行选择的动作，并更新环境
        actions = actions.detach().cpu().numpy()
        step_rewards, last_batch_status, current_batch_status, agent_rewards = batch_steps(env, actions, instance_status, device, record_buffer=True, num_agents=num_agents)
        reward_info = reward_info + step_rewards
        # 更新buffer
        update_buffer(replay_buffer, last_batch_status, actions, current_batch_status, agent_rewards, batch_size, epoch_id, batch_id)
        # 进行经验回放
        Q, sampled_experience_dataset, critic_loss, critic_baseline_loss, critic_gradients = update_critics(
            env, encoder, num_agents, critic_estimator, target_critic_estimator, critic_optimizer,
            alpha, gamma, tau, action_selector, replay_buffer, buffer_batch_size,
            history_data, device, record_gradient
        )
        # 更新Actor
        actor_loss, action_selector_gradients, encoder_gradients = update_actor(
            env, Q.detach(), sampled_experience_dataset, encoder, action_selector, critic_estimator, actor_optimizer,
            num_agents, alpha, record_gradient, buffer_batch_size
        )
        # 更新buffer和历史数据
        clear_history_data(replay_buffer, history_data)

        Q_val = Q.mean(dim=0).sum().item()
        Q_list.append(Q_val)
        critic_loss_list.append(critic_loss.item())
        critic_baseline_loss_list.append(critic_baseline_loss.item())
        actor_loss_list.append(actor_loss.item())

        log_dict = {
            "epoch": epoch_id,
            "batch_id": batch_id,
            "T": T,
            "Q": Q_val,
            "critic_loss": critic_loss.item(),
            "critic_baseline_loss": critic_baseline_loss.item(),
            "critic_gradients": critic_gradients,
            "actor_loss": actor_loss.item(),
            "action_selector_gradients": action_selector_gradients,
            "encoder_gradients": encoder_gradients,
        }

        if window_size > 0 and T > window_size:
            log_dict['Q_window_avg'] = np.mean(Q_list)
            log_dict['critic_loss_window_avg'] = np.mean(critic_loss_list)
            log_dict['critic_baseline_loss_window_avg'] = np.mean(critic_baseline_loss_list)
            log_dict['actor_loss_window_avg'] = np.mean(actor_loss_list)

        log_buffer.append(log_dict)
        if len(log_buffer) >= log_interval:
            for log in log_buffer:
                wandb.log(log)
            log_buffer.clear()

        print(
            f"epoch: {epoch_id}, batch_id: {batch_id}, T: {T}, Q: {Q_val}, "
            f"critic_loss: {critic_loss.item()}, "
            f"critic_baseline_loss: {critic_baseline_loss.item()}, "
            f"actor_loss: {actor_loss.item()}"
        )

        T += 1

    return reward_info


def train_model(
        env_params,
        env_generator,
        encoder,
        action_selector,
        actor_optimizer,
        num_agents,
        critic_estimator,
        target_critic_estimator,
        critic_optimizer,
        epochs,
        replay_buffer_batch_size,
        replay_buffer_capacity,
        alpha,
        gamma,
        tau,
        device,
        max_workers,
        record_gradient,
        window_size,
        log_interval
):
    """
    Training loop for the MAAM model using GAT encoder.

    Args:
        env_params: Env名称，CVRPSTW 环境参数, 数量
        env_generator: 环境生成器
        encoder: 编码节点特征
        action_selector: 选择器
        actor_optimizer: 表演者优化器
        num_agents: agent数量
        critic_estimator: 评价器
        target_critic_estimator: 目标评价器
        critic_optimizer: 评论者优化器
        epochs: epochs 数量
        replay_buffer_batch_size: 缓冲区批次大小
        replay_buffer_capacity: 缓冲区容量
        alpha: 熵系数
        gamma: 折扣因子
        tau: 软更新参数
        device: CPU or GPU,
        max_workers: 最大工作线程数
        record_gradient: 是否记录梯度
        window_size: 窗口大小
        log_interval: 日志间隔
    """
    env_name, env_params, total_num_instances = env_params
    batch_size = env_params['batch_size']
    batch_times = total_num_instances // batch_size
    assert total_num_instances % batch_size == 0, f"The total number of every env must be divided by its corresponding batch size"

    # wandb
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    wandb.init(
        project=f"{project_name}_MAAC",
        name=f"{env_name}_{env_params['index']}_{current_time}",
        config={
            "env_name": env_name,
            "env_params": env_params,
            "total_num_instances": total_num_instances,
            "epochs": epochs,
            "encoder": str(encoder),
            "action_selector": str(action_selector),
            "actor_optimizer": str(actor_optimizer),
            "critic_estimator": str(critic_estimator),
            "target_critic_estimator": str(target_critic_estimator),
            "critic_optimizer": str(critic_optimizer),
            "device": device,
            "max_workers": max_workers
        }
    )

    # 记录时间步
    T = 0

    # buffer and history data
    replay_buffer = ReplayBuffer(capacity=replay_buffer_capacity)
    history_data = {}

    # 记录window
    Q_list = deque(maxlen=window_size)
    critic_loss_list = deque(maxlen=window_size)
    critic_baseline_loss_list = deque(maxlen=window_size)
    actor_loss_list = deque(maxlen=window_size)
    reward_list = deque(maxlen=window_size)

    # 记录log
    log_buffer = []

    for epoch in tqdm(range(epochs)):
        env = env_generator(max_workers=max_workers, **env_params)
        for batch_id in tqdm(range(batch_times)):
            # ============================= Multi-agent Attention Soft Actor-Critic =============================
            # 1. run batch
            reward_info = run_batch(
                T, env, encoder, action_selector, critic_estimator, target_critic_estimator,
                actor_optimizer, critic_optimizer,
                num_agents, replay_buffer, replay_buffer_batch_size, alpha, gamma, tau,
                'sampling', device, epoch, batch_id, history_data, record_gradient,
                window_size, Q_list, critic_loss_list, critic_baseline_loss_list, actor_loss_list,
                log_buffer, log_interval
            )

            # 2. 记录 reward
            reward_info_mean = reward_info.mean().item()
            reward_list.append(reward_info_mean)

            log_dict = {
                'epoch': epoch,
                'batch_id': batch_id,
                'reward': reward_info_mean,
            }
            if window_size > 0 and batch_id > window_size:
                log_dict['reward_window_mean'] = np.mean(reward_list)

            wandb.log(log_dict)

            print(f"epoch: {epoch}, batch_id: {batch_id}, reward: {reward_info_mean}")

            # ============================= End Multi-agent Attention Soft Actor-Critic =============================


if __name__ == '__main__':
    # encoder
    graph_encoder = Encoder(
        encoder_model=MultiLayerEdgeGAT,
        encoder_params=MultiLayerEdgeGATParams,
        k_distance_nearest_neighbors=k_distance_nearest_neighbors,
        k_time_nearest_neighbors=k_time_nearest_neighbors,
        device=device
    )
    graph_encoder.to(device)

    # action selector
    action_selector = ActionSelector(
        embedding_dim=embedding_dim,
        heads=action_heads,
        dynamic_vehicle_dim=dynamic_vehicle_dim,
        dynamic_customer_dim=dynamic_customer_dim
    )
    action_selector.to(device)

    # critic estimator
    critic_estimator = CriticEstimator(
        num_agents=num_agents,
        state_dim=embedding_dim+dynamic_vehicle_dim,
        action_dim=embedding_dim+dynamic_customer_dim,
        embedding_dim=embedding_dim,
        num_heads=critic_heads,
        activation=critic_activation,
        device=device
    )
    critic_estimator.to(device)

    # target critic estimator
    target_critic_estimator = copy.deepcopy(critic_estimator)
    for param in target_critic_estimator.parameters():
        param.requires_grad = False
    target_critic_estimator.to(device)

    # optimizer
    actor_optimizer = Adam(
        list(graph_encoder.parameters()) + list(action_selector.parameters()),
        lr=actor_rl
    )
    critic_optimizer = Adam(
        critic_estimator.parameters(),
        lr=critic_lr
    )

    try:
        # training
        train_model(
            env_params=('small', small_params, num_samll_instances),
            env_generator=BatchVRPEnvs,
            encoder=graph_encoder,
            action_selector=action_selector,
            actor_optimizer=actor_optimizer,
            num_agents=num_agents,
            critic_estimator=critic_estimator,
            target_critic_estimator=target_critic_estimator,
            critic_optimizer=critic_optimizer,
            epochs=epochs,
            replay_buffer_batch_size=replay_buffer_batch_size,
            replay_buffer_capacity=replay_buffer_capacity,
            alpha=alpha,
            gamma=gamma,
            tau=tau,
            device=device,
            max_workers=max_workers,
            record_gradient=record_gradient,
            window_size=reward_window_size,
            log_interval=log_interval
        )
    except Exception as e:
        # 捕获手动中断，进行清理操作
        print("训练过程中检测到异常，正在清理资源并关闭...")
        traceback.print_exc()
        print(f"异常信息: {e}")
        # 清理深度学习相关资源，例如释放GPU显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # 释放GPU缓存
    finally:
        # 最后保障措施，确保资源释放
        print("所有资源清理完成。训练已中止。")
        # 保存模型 (假设你有4个模型)
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        try:
            torch.save(graph_encoder.state_dict(), f'models/encoder_{current_time}.pth')
            torch.save(action_selector.state_dict(), f'models/action_selector_{current_time}.pth')
            torch.save(critic_estimator.state_dict(), f'models/critic_estimator_{current_time}.pth')
            torch.save(target_critic_estimator.state_dict(), f'models/target_critic_estimator_{current_time}.pth')
            print("模型已成功保存。")
        except Exception as e:
            print(f"保存模型时出错: {e}")
        try:
            # 确保 wandb 会话关闭
            wandb.finish()
        except Exception as e:
            print(f"finally 中结束 wandb 时出错: {e}")
        print("资源已完全清理。")
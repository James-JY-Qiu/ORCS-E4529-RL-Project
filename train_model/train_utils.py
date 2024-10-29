import numpy as np
import torch


def env_encode(encoder, batch_customer_data, batch_company_data, wait_times):
    """
    编码环境数据
    """
    encoder.encode(
        batch_customer_data=batch_customer_data,
        batch_company_data=batch_company_data,
        wait_times=wait_times
    )


def replace_baseline_model(baseline_model, current_model):
    """
    替换基线模型
    """
    with torch.no_grad():
        baseline_model.load_state_dict(current_model.state_dict())


def soft_update_target_model(target_model, current_model, tau):
    """
    软更新目标模型
    """
    with torch.no_grad():
        for target_param, current_param in zip(target_model.parameters(), current_model.parameters()):
            target_param.data.copy_(tau * current_param.data + (1.0 - tau) * target_param.data)


def return_batch_neg_inf_masks(env):
    """
    返回整个 batch 的 neg_inf_masks (batch_size, M, N)
    """
    batch_neg_inf_mask = []
    for vrp_env in env.envs:
        instance_neg_inf_mask = []
        vehicle_count = vrp_env.vehicle_count  # 获取当前实例的车辆数量

        # 将每辆车辆的掩码添加到实例中
        for vehicle_id in range(vehicle_count):
            neg_inf_mask = vrp_env.return_neg_inf_mask(vehicle_id)
            instance_neg_inf_mask.append(neg_inf_mask)

        # 将该实例的所有车辆掩码堆叠在一起，形成 (num_vehicles, num_customers) 的矩阵
        batch_neg_inf_mask.append(np.array(instance_neg_inf_mask))

    # 最后，将所有实例的掩码堆叠在一起，形成 (batch_size, num_vehicles, num_customers) 的三维张量
    batch_neg_inf_mask = torch.tensor(np.array(batch_neg_inf_mask), dtype=torch.float)

    return batch_neg_inf_mask


def batch_steps(env, actions, instance_status, device, record_buffer=False, num_agents=-1):
    """
    对整个 batch 执行一次 step。
    """
    if record_buffer and num_agents == -1:
        raise ValueError("num_agents must be provided when record_buffer is True")
    batch_size = env.batch_size
    batch_step_rewards = torch.zeros(batch_size, dtype=torch.float, device=device)

    agent_rewards = None
    last_batch_status = None
    current_batch_status = None

    if record_buffer:
        agent_rewards = torch.zeros(batch_size, num_agents, dtype=torch.float, device=device)
        last_batch_status = env.get_current_batch_status(return_neg_inf_mask=True)

    for instance_id, vrp_env in enumerate(env.envs):
        if instance_status[instance_id]:
            continue

        step_reward = 0.0  # 初始 step_reward
        for vehicle_id in range(vrp_env.vehicle_count):
            action = actions[instance_id, vehicle_id]  # 获取 action
            reward = vrp_env.step(vehicle_id, action)  # 计算reward并且更新状态
            step_reward += reward
            if record_buffer:
                agent_rewards[instance_id, vehicle_id] = reward

        # 检查该实例是否完成
        done = vrp_env.judge_finish()
        batch_step_rewards[instance_id] = step_reward
        instance_status[instance_id] = done

    if record_buffer:
        current_batch_status = env.get_current_batch_status(return_neg_inf_mask=True)

    return batch_step_rewards, last_batch_status, current_batch_status, agent_rewards


def record_gradients_and_weights(network, explosion_threshold=1e3):
    # 记录梯度和权重
    network_stats = {}

    for name, param in network.named_parameters():
        if param.grad is not None:
            mean_grad = param.grad.mean().item()
            std_grad = param.grad.std().item()
            mean_weight = param.data.mean().item()
            std_weight = param.data.std().item()

            network_stats[name] = {
                "grad_mean": mean_grad,
                "grad_std": std_grad,
                "weight_mean": mean_weight,
                "weight_std": std_weight
            }

            # 检查梯度是否接近 0
            if np.isclose(mean_grad, 0.0):
                print(f"Parameter: {name} has a small gradient ({mean_grad}).")

            # 检查梯度是否过大
            if abs(mean_grad) > explosion_threshold or std_grad > explosion_threshold:
                print(
                    f"Warning: Parameter: {name} has a large gradient (mean: {mean_grad}, std: {std_grad}), indicating possible gradient explosion.")

            # 检查权重是否过大或过小
            if abs(mean_weight) > explosion_threshold or std_weight > explosion_threshold:
                print(
                    f"Warning: Parameter: {name} has a large weight (mean: {mean_weight}, std: {std_weight}), which may indicate instability.")
            elif np.isclose(mean_weight, 0.0):
                print(f"Parameter: {name} has a small weight ({mean_weight}), which may lead to vanishing gradients.")

        else:
            print(f"Parameter: {name} has no gradient.")

    return network_stats


def set_model_status(model_list, training):
    """
    设置模型的训练状态
    """
    for model in model_list:
        model.train(training)

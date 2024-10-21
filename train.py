import numpy as np
import torch

from torch.optim import Adam
import scipy.stats as stats

from tqdm.auto import tqdm
import copy
from datetime import datetime

from Encoder import Encoder, MultiLayerEdgeGAT
from Decoder import BatchVRPEnvs
from Action import ActionSelector

from params import project_name, max_workers, device
from params import embedding_dim, action_extra_embedding_dims, action_heads
from params import epochs, lr
from params import small_params, num_samll_instances
from params import MultiLayerEdgeGATParams
from params import record_gradient, reward_window_size

import wandb


def return_batch_neg_inf_masks(env, max_num_vehicles):
    """
    返回整个 batch 的 neg_inf_masks (batch_size, M, N)
    """
    # 使用 ProcessPoolExecutor 进行多进程并行化，使用 map 保证顺序
    batch_neg_inf_mask = []
    for vrp_env in env.envs:
        instance_neg_inf_mask = []
        vehicle_count = vrp_env.vehicle_count  # 获取当前实例的车辆数量

        # 将每辆真实车辆的掩码添加到实例中
        for vehicle_id in range(vehicle_count):
            neg_inf_mask = vrp_env.return_neg_inf_mask(vehicle_id)
            instance_neg_inf_mask.append(neg_inf_mask)

        if max_num_vehicles - vehicle_count > 0:
            # 为每辆虚拟车辆添加掩码，使其一直在depot处
            depot_mask = torch.full((vrp_env.num_nodes,), float('-inf'))
            depot_mask[0] = 0.0
            for vehicle_id in range(max_num_vehicles - vehicle_count):
                instance_neg_inf_mask.append(depot_mask)

        # 将该实例的所有车辆掩码堆叠在一起，形成 (num_vehicles, num_customers) 的矩阵
        instance_neg_inf_mask = torch.stack(instance_neg_inf_mask, dim=0)
        batch_neg_inf_mask.append(instance_neg_inf_mask)

    # 最后，将所有实例的掩码堆叠在一起，形成 (batch_size, num_vehicles, num_customers) 的三维张量
    batch_neg_inf_mask = torch.stack(batch_neg_inf_mask, dim=0)

    return batch_neg_inf_mask


def batch_steps(env, actions, instance_status, device):
    """
    对整个 batch 执行一次 step，使用多进程处理。
    """
    batch_size = env.batch_size
    step_rewards = torch.zeros(batch_size, dtype=torch.float, device=device)  # 初始化 step_rewards
    actions = actions.detach().cpu().numpy()  # 确保不计算梯度并且返回cpu

    for instance_id, vrp_env in enumerate(env.envs):
        if instance_status[instance_id]:
            continue

        step_reward = 0.0  # 初始 step_reward
        for vehicle_id in range(vrp_env.vehicle_count):
            action = actions[instance_id, vehicle_id]  # 获取 action
            reward = vrp_env.step(vehicle_id, action)  # 计算reward并且更新状态
            step_reward += reward

        # 检查该实例是否完成
        done = vrp_env.judge_finish()
        step_rewards[instance_id] = step_reward
        instance_status[instance_id] = done

    return step_rewards


def env_encode(encoder, batch_customer_data, batch_company_data, num_customers, wait_times):
    """
    编码环境数据
    """
    encoder.encode(
        batch_customer_data=batch_customer_data,
        batch_company_data=batch_company_data,
        num_customers=num_customers,
        wait_times=wait_times
    )


def run_batch(env, encoder, action_selector, mode, generate, device):
    """
    执行一次完整的 sampling batch (即跑完一遍所有的 instances)
    """
    env.reset(generate=generate)
    batch_size = env.batch_size

    # 编码环境数据
    env_encode(encoder, env.batch_customer_data, env.batch_company_data, env.num_customers, env.wait_times)
    env.update_parameters(encoder.batch_distance_matrices, encoder.batch_num_nodes)

    # 记录 instance 是否结束
    instance_status = np.zeros(batch_size, dtype=bool)
    # 记录 reward
    reward_info = torch.zeros(batch_size, dtype=torch.float, device=device)
    # 记录 log_probs
    log_probs_info = torch.zeros(batch_size, dtype=torch.float, device=device)
    # 记录时间步
    t = 0
    while not instance_status.all():
        current_batch_status = env.get_current_batch_status()
        current_vehicle_embeddings, current_customer_embeddings = encoder.get_current_batch_state(**current_batch_status)
        max_num_vehicles = current_vehicle_embeddings.size(1) - 1
        batch_neg_inf_mask = return_batch_neg_inf_masks(env, max_num_vehicles).to(device)
        actions, log_probs = action_selector(
            current_vehicle_embeddings,
            current_customer_embeddings,
            batch_neg_inf_mask,
            mode=mode
        )
        if log_probs is not None:
            log_probs_info = log_probs_info + log_probs.sum(dim=1)

        # 执行选择的动作，并更新环境
        step_rewards = batch_steps(env, actions, instance_status, device)
        reward_info = reward_info + step_rewards

        # print(f"{t} finished number: {instance_status.sum().item()}")
        t += 1

    return reward_info, log_probs_info


def replace_baseline_model(baseline_model, current_model):
    # 在复制参数时禁用梯度计算
    with torch.no_grad():
        baseline_model.load_state_dict(current_model.state_dict())

    # 再次确保 baseline_model 的所有参数的 requires_grad 为 False
    for param in baseline_model.parameters():
        param.requires_grad = False


def record_gradieents(encoder, action_selector):
    # 记录梯度
    encoder_gradients = {}
    action_selector_gradients = {}

    # print("\nEncoder Parameter Gradients:")
    for name, param in encoder.named_parameters():
        if param.grad is not None:
            encoder_gradients[name] = {
                "mean": param.grad.mean().item(),
                "std": param.grad.std().item()
            }
            # print(
            #     f"Parameter: {name}, Gradient Mean: {param.grad.mean().item()}, Gradient Std: {param.grad.std().item()}")
        else:
            print(f"Parameter: {name} has no gradient.")

    # print("\nAction Selector Parameter Gradients:")
    for name, param in action_selector.named_parameters():
        if param.grad is not None:
            action_selector_gradients[name] = {
                "mean": param.grad.mean().item(),
                "std": param.grad.std().item()
            }
            # print(
            #     f"Parameter: {name}, Gradient Mean: {param.grad.mean().item()}, Gradient Std: {param.grad.std().item()}")
        else:
            print(f"Parameter: {name} has no gradient.")

    return encoder_gradients, action_selector_gradients


# Define the training function based on Algorithm 1
def train_model(
        env_params,
        env_generator,
        encoder,
        baseline_encoder,
        action_selector,
        baseline_action_selector,
        optimizer,
        epochs,
        device,
        max_workers,
        record_gradient,
        reward_window_size
):
    """
    Training loop for the MAAM model using GAT encoder.

    Args:
        env_params: Env名称，CVRPSTW 环境参数, 数量
        env_generator: 环境生成器
        encoder: 编码节点特征
        baseline_encoder: 编码节点特征 (baseline)
        action_selector: 选择器
        baseline_action_selector: 选择器 (baseline)
        optimizer: 优化器
        epochs: epochs 数量
        device: CPU or GPU,
        max_workers: 最大工作线程数
    """
    env_name, env_params, total_num_instances = env_params
    batch_size = env_params['batch_size']
    batch_times = total_num_instances // batch_size
    assert total_num_instances % batch_size == 0, f"The total number of every env must be divided by its corresponding batch size"

    # wandb
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    wandb.init(
        project=project_name,
        name= f"{env_name}_{env_params['index']}_{current_time}",
        config={
            "env_name": env_name,
            "env_params": env_params,
            "total_num_instances": total_num_instances,
            "epochs": epochs,
            "encoder": str(encoder),
            "action_selector": str(action_selector),
            "device": device,
            "optimizer": str(optimizer),
            "max_workers": max_workers
        }
    )

    for epoch in tqdm(range(epochs)):
        env = env_generator(max_workers=max_workers, **env_params)
        sampling_reward_list = []
        greedy_reward_list = []
        loss_list = []

        for batch_id in tqdm(range(batch_times)):
            # 1. sampling run
            sampling_reward, log_probs_info = run_batch(
                env, encoder, action_selector,
                mode='sampling', generate=True, device=device
            )
            # 2. greedy run
            with torch.no_grad():
                greedy_reward, _ = run_batch(
                    env, baseline_encoder, baseline_action_selector,
                    mode='greedy', generate=False, device=device
                )
            # 3. 计算差值
            advantage = sampling_reward - greedy_reward
            # 4. 计算loss
            loss = -(log_probs_info * advantage).mean()
            # 5. 清除上次计算的梯度
            optimizer.zero_grad()
            # 6. 反向传播计算当前 batch 的梯度
            loss.backward()
            if record_gradient:
                encoder_gradients, action_selector_gradients = record_gradieents(encoder, action_selector)
            else:
                encoder_gradients, action_selector_gradients = None, None
            # 7. 使用优化器更新参数
            optimizer.step()

            # 8. 进行 Wilcoxon 检验，检验当前策略与基线策略是否有显著差异
            w_stat, p_value = stats.wilcoxon(sampling_reward.cpu().numpy(), greedy_reward.cpu().numpy())

            # 9. 记录数据
            sampling_reward_mean = sampling_reward.mean().item()
            greedy_reward_mean = greedy_reward.mean().item()

            sampling_reward_list.append(sampling_reward_mean)
            greedy_reward_list.append(greedy_reward_mean)
            loss_list.append(loss.item())

            log_dict = {
                "epoch": epoch,
                "batch_id": batch_id,
                "sampling_reward_mean": sampling_reward_mean,
                "greedy_reward_mean": greedy_reward_mean,
                "loss": loss.item(),
                "wilcoxon_stat": w_stat,
                "wilcoxon_p_value": p_value,
                "encoder_gradients": encoder_gradients,
                "action_selector_gradients": action_selector_gradients
            }

            if reward_window_size > 0 and batch_id > reward_window_size:
                log_dict["moving_average_sampling_reward"] = np.mean( sampling_reward_list[-reward_window_size:])
                log_dict["moving_average_greedy_reward"] = np.mean(greedy_reward_list[-reward_window_size:])
                log_dict["moving_average_loss"] = np.mean(loss_list[-reward_window_size:])

            wandb.log(log_dict)

            # 9. 如果 p 值小于 0.05，表示当前策略显著优于基线策略，更新基线策略
            if p_value < 0.05:
                replace_baseline_model(baseline_encoder, encoder)
                replace_baseline_model(baseline_action_selector, action_selector)
                print("更新Baseline策略！")

            print(
                f"Batch ID: {batch_id}, Sampling Reward: {sampling_reward.mean().item()}, Greedy Reward: {greedy_reward.mean().item()}, Loss: {loss.item()}, Wilcoxon Stat: {w_stat}, P-value: {p_value}"
            )


if __name__ == '__main__':
    # encoder
    encoder = Encoder(
        encoder_model=MultiLayerEdgeGAT,
        encoder_params=MultiLayerEdgeGATParams,
        device=device
    )
    encoder.to(device)

    # baseline encoder
    baseline_encoder = copy.deepcopy(encoder)
    for param in baseline_encoder.parameters():
        param.requires_grad = False
    baseline_encoder.to(device)

    # action selector
    action_selector = ActionSelector(
        embedding_dim=embedding_dim + action_extra_embedding_dims,
        heads=action_heads
    )
    action_selector.to(device)

    # baseline action selector
    baseline_action_selector = copy.deepcopy(action_selector)
    for param in baseline_action_selector.parameters():
        param.requires_grad = False
    baseline_action_selector.to(device)

    # optimizer
    optimizer = Adam(
        list(encoder.parameters()) + list(action_selector.parameters()),
        lr=lr  # 设置学习率
    )

    try:
        # training
        train_model(
            env_params=('small', small_params, num_samll_instances),
            env_generator=BatchVRPEnvs,
            encoder=encoder,
            baseline_encoder=baseline_encoder,
            action_selector=action_selector,
            baseline_action_selector=baseline_action_selector,
            optimizer=optimizer,
            epochs=epochs,
            device=device,
            max_workers=max_workers,
            record_gradient=record_gradient,
            reward_window_size=reward_window_size
        )
    except Exception:
        # 捕获手动中断，进行清理操作
        print("训练过程中检测到异常，正在清理资源并关闭...")
        # 清理深度学习相关资源，例如释放GPU显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # 释放GPU缓存
    finally:
        # 最后保障措施，确保资源释放
        print("所有资源清理完成。训练已中止。")
        # 保存模型 (假设你有4个模型)
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        try:
            torch.save(encoder.state_dict(), f'models/encoder_{current_time}.pth')
            torch.save(baseline_encoder.state_dict(), f'models/baseline_encoder_{current_time}.pth')
            torch.save(action_selector.state_dict(), f'models/action_selector_{current_time}.pth')
            torch.save(baseline_action_selector.state_dict(), f'models/baseline_action_selector_{current_time}.pth')
            print("模型已成功保存。")
        except Exception as e:
            print(f"保存模型时出错: {e}")

        try:
            # 确保 wandb 会话关闭
            wandb.finish()
        except Exception as e:
            print(f"finally 中结束 wandb 时出错: {e}")
        print("资源已完全清理。")

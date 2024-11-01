from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from torch.optim import Adam
import scipy.stats as stats

from tqdm.auto import tqdm
import copy
from datetime import datetime

from Encoder import Encoder, MultiLayerEdgeGAT
from Env import BatchVRPEnvs
from Action import ActionSelector

from train_model.train_utils import return_batch_neg_inf_masks, batch_steps, record_gradients_and_weights, env_encode, replace_baseline_model, set_model_status

from params import project_name, max_workers, device
from params import small_params, num_samll_instances
from params import k_distance_nearest_neighbors_percent, k_time_nearest_neighbors_percent
from params import record_gradient, reward_window_size

import wandb
import traceback


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

# train
epochs = 100

# optimizer
lr = 1e-3


def run_batch(env, encoder, action_selector, mode, generate, device):
    """
    执行一次完整的 sampling batch (即跑完一遍所有的 instances)
    """
    env.reset(generate=generate)
    batch_size = env.batch_size

    # 编码环境数据
    env_encode(encoder, env.batch_customer_data, env.batch_company_data, env.wait_times)
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
        current_vehicle_embeddings, current_customer_embeddings = encoder.get_current_batch_state(
            include_global=False, **current_batch_status
        )
        batch_neg_inf_mask = return_batch_neg_inf_masks(env).to(device)
        actions, log_probs = action_selector(
            current_vehicle_embeddings,
            current_customer_embeddings,
            batch_neg_inf_mask,
            mode=mode
        )
        if log_probs is not None:
            log_probs_info = log_probs_info + log_probs.sum(dim=-1)

        # 执行选择的动作，并更新环境
        actions = actions.detach().cpu().numpy()
        step_rewards, _, _, _ = batch_steps(env, actions, instance_status, device)
        reward_info = reward_info + step_rewards

        # print(f"{t} finished number: {instance_status.sum().item()}")
        t += 1

    return reward_info, log_probs_info


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
        record_gradient: 是否记录梯度
        reward_window_size: 奖励窗口大小
    """
    env_name, env_params, total_num_instances = env_params
    batch_size = env_params['batch_size']
    batch_times = total_num_instances // batch_size
    assert total_num_instances % batch_size == 0, f"The total number of every env must be divided by its corresponding batch size"

    # wandb
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    wandb.init(
        project=f"{project_name}_REINFORCE",
        name=f"{env_name}_{env_params['index']}_{current_time}",
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

    sampling_reward_list = deque(maxlen=reward_window_size)
    greedy_reward_list = deque(maxlen=reward_window_size)
    loss_list = deque(maxlen=reward_window_size)

    env = env_generator(
        max_workers=max_workers,
        **env_params
    )

    for epoch in tqdm(range(epochs)):
        for batch_id in tqdm(range(batch_times)):
            # ============================= Strategy Gradient =============================
            # 1. sampling run
            set_model_status([encoder, action_selector], training=True)
            sampling_reward, log_probs_info = run_batch(
                env, encoder, action_selector,
                mode='sampling', generate=True, device=device
            )
            # 2. greedy run
            with torch.no_grad():
                set_model_status([encoder, action_selector], training=False)
                greedy_reward, _, = run_batch(
                    env, baseline_encoder, baseline_action_selector,
                    mode='greedy', generate=False, device=device
                )
            # 3. 计算差值
            advantage = sampling_reward - greedy_reward
            advantage_std = advantage.std()
            if advantage_std != 0:
                standardize_advantage = (advantage - advantage.mean()) / advantage_std
            else:
                standardize_advantage = advantage
            # 4. 计算loss
            expected_rewards = (log_probs_info * standardize_advantage).mean()
            loss = -expected_rewards
            # 5. 清除上次计算的梯度
            optimizer.zero_grad()
            # 6. 反向传播计算当前 batch 的梯度
            loss.backward()
            if record_gradient:
                encoder_gradients = record_gradients_and_weights(encoder)
                action_selector_gradients = record_gradients_and_weights(action_selector)
            else:
                encoder_gradients = action_selector_gradients = None

            # 7. 使用优化器更新参数
            optimizer.step()

            # 8. 进行 Wilcoxon 检验，检验当前策略与基线策略是否有显著差异
            sampling_reward = sampling_reward.detach().cpu().numpy()
            greedy_reward = greedy_reward.detach().cpu().numpy()
            wilcoxon_results = stats.wilcoxon(sampling_reward, greedy_reward)

            # 9. 记录数据
            sampling_reward_mean = sampling_reward.mean()
            sampling_reward_max = sampling_reward.max()
            sampling_reward_min = sampling_reward.min()
            sampling_reward_std = sampling_reward.std()
            greedy_reward_mean = greedy_reward.mean()
            greedy_reward_max = greedy_reward.max()
            greedy_reward_min = greedy_reward.min()
            greedy_reward_std = greedy_reward.std()

            sampling_reward_list.append(sampling_reward_mean)
            greedy_reward_list.append(greedy_reward_mean)
            loss_list.append(loss.item())

            log_dict = {
                "epoch": epoch,
                "batch_id": batch_id,
                "sampling_reward_mean": sampling_reward_mean,
                "sampling_reward_max": sampling_reward_max,
                "sampling_reward_min": sampling_reward_min,
                "sampling_reward_std": sampling_reward_std,
                "greedy_reward_mean": greedy_reward_mean,
                "greedy_reward_max": greedy_reward_max,
                "greedy_reward_min": greedy_reward_min,
                "greedy_reward_std": greedy_reward_std,
                "sampling_greedy_diff_percent": (sampling_reward_mean - greedy_reward_mean) / greedy_reward_mean,
                "expected_rewards": expected_rewards.item(),
                "loss": loss.item(),
                "wilcoxon_stat": wilcoxon_results.statistic,
                "wilcoxon_p_value": wilcoxon_results.pvalue,
                "encoder_gradients": encoder_gradients,
                "action_selector_gradients": action_selector_gradients
            }

            if reward_window_size > 0 and batch_id > reward_window_size:
                log_dict["moving_average_sampling_reward"] = np.mean(sampling_reward_list)
                log_dict["moving_average_greedy_reward"] = np.mean(greedy_reward_list)
                log_dict["moving_average_loss"] = np.mean(loss_list)

            wandb.log(log_dict)

            # 10. 如果 p 值小于 0.05，表示当前策略显著优于基线策略，更新基线策略
            if sampling_reward_mean >= greedy_reward_mean and wilcoxon_results.pvalue < 0.05:
                replace_baseline_model(baseline_encoder, encoder)
                replace_baseline_model(baseline_action_selector, action_selector)
                print("更新Baseline策略！")

            # 11. 保存模型
            if batch_id % 100 == 0:
                torch.save(encoder.state_dict(), f'models/check_point_encoder.pth')
                torch.save(baseline_encoder.state_dict(), f'models/check_point_baseline_encoder.pth')
                torch.save(action_selector.state_dict(), f'models/check_point_action_selector.pth')
                torch.save(baseline_action_selector.state_dict(), f'models/check_point_baseline_action_selector.pth')
                print("模型已保存！")

            print(
                f"Batch ID: {batch_id}, "
                f"Sampling Reward: {sampling_reward.mean().item()}, "
                f"Greedy Reward: {greedy_reward.mean().item()}, "
                f"Loss: {loss.item()}, "
                f"Wilcoxon Stat: {wilcoxon_results.statistic}, P-value: {wilcoxon_results.pvalue}"
            )

            # ============================= End Strategy Gradient =============================


if __name__ == '__main__':
    ENV_PARAMS = ('small', small_params, num_samll_instances)

    # encoder
    encoder = Encoder(
        encoder_model=MultiLayerEdgeGAT,
        encoder_params=MultiLayerEdgeGATParams,
        k_distance_nearest_neighbors_percent=k_distance_nearest_neighbors_percent,
        k_time_nearest_neighbors_percent=k_time_nearest_neighbors_percent,
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
        embedding_dim=embedding_dim,
        heads=action_heads,
        dynamic_vehicle_dim=dynamic_vehicle_dim,
        dynamic_customer_dim=dynamic_customer_dim
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
            env_params=ENV_PARAMS,
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
    except Exception as e:
        # 捕获手动中断，进行清理操作
        print("训练过程中检测到异常，正在清理资源并关闭...")
        # 打印完整的异常堆栈信息
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

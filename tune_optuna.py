import optuna
import torch
import torch.nn.functional as F
import numpy as np
import os

# 从你的项目中导入必要的模块
from env import SamplingEnv2_0
from agent import TransformerEncoder, MADDPGAgent
from replay_buffer import ReplayBuffer
import config as default_config

# 确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

def get_local_obs(obs, agent_id, env):
    """
    辅助函数：为单个智能体提取并归一化局部观测。
    （从 main.py 复制而来）
    """
    agent_pos_normalized = obs['agent_positions'][agent_id] / env.size
    agent_load_normalized = obs['agent_loads'][agent_id] / (env.agent_capacity[agent_id] + 1e-7)
    agent_energy_normalized = obs['agent_energy'][agent_id] / (env.agent_energy_max[agent_id] + 1e-7)
    
    max_charge_time = env.fast_charge_time if agent_id < 3 else env.slow_charge_time
    agent_charging_status_normalized = obs['agent_charging_status'][agent_id] / max_charge_time
    
    agent_obs_local_normalized = np.concatenate([
        agent_pos_normalized,
        [agent_load_normalized, agent_energy_normalized, obs['agent_task_status'][agent_id], agent_charging_status_normalized]
    ])
    return agent_obs_local_normalized

def objective(trial):
    """
    Optuna 的目标函数。
    每次 trial 都会用一组新的超参数来运行一次完整的训练，并返回最终的性能评估。
    """
    # --- 1. 定义超参数的搜索空间 ---
    # 使用 trial.suggest_... 方法来定义每个超参数的范围
    # 对于学习率等跨度大的参数，建议使用对数均匀分布 (log=True)
    lr_actor = trial.suggest_float("lr_actor", 1e-5, 1e-3, log=True)
    lr_critic = trial.suggest_float("lr_critic", 1e-4, 1e-2, log=True)
    gamma = trial.suggest_float("gamma", 0.99, 0.9999)
    tau = trial.suggest_float("tau", 0.001, 0.01, log=True)
    noise_decay = trial.suggest_float("noise_decay", 0.99, 0.9999)
    hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512])
    
    # Transformer 参数
    emb_dim = trial.suggest_categorical("emb_dim", [64, 128])
    n_heads = trial.suggest_categorical("n_heads", [2, 4, 8])
    n_layers = trial.suggest_categorical("n_layers", [1, 2, 3])

    # --- 2. 设置环境和智能体 ---
    # 使用 trial 建议的超参数来配置环境和智能体
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 为了加速调参，我们可以在每次 trial 中使用较少的回合数
    # 原配置是 1500，这里我们用 500
    TUNING_NUM_EPISODES = 500

    env = SamplingEnv2_0(num_points=default_config.NUM_POINTS, num_agents=default_config.NUM_AGENTS)
    num_agents = default_config.NUM_AGENTS
    num_points = default_config.NUM_POINTS
    action_dim = env.action_dim
    
    agent_local_obs_dim = 6
    point_feature_dim = 5
    agent_obs_dim = agent_local_obs_dim + num_points + (num_agents - 1) * action_dim
    
    transformer = TransformerEncoder(
        point_input_dim=point_feature_dim, 
        agent_local_obs_dim=agent_local_obs_dim, 
        emb_dim=emb_dim, 
        n_heads=n_heads, 
        n_layers=n_layers
    ).to(device)
    
    agents = [MADDPGAgent(
        agent_obs_dim=agent_obs_dim, 
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        gamma=gamma,
        tau=tau
    ).to(device) for _ in range(num_agents)]

    replay_buffers = [ReplayBuffer(capacity=default_config.CAPACITY) for _ in range(num_agents)]

    episode_rewards_log = []
    step_counter = 0
    noise_std = default_config.NOISE_STD_START

    # --- 3. 运行训练循环 (与 main.py 基本相同) ---
    for episode in range(TUNING_NUM_EPISODES):
        obs = env.reset()
        done = False
        total_rewards = np.zeros(num_agents)
        last_actions_one_hot = F.one_hot(torch.full((num_agents,), num_points), num_classes=action_dim).numpy()
        num_steps = 0

        while (not done) and (num_steps < default_config.MAX_NUM_STEPS):
            num_steps += 1
            
            point_features = np.hstack([
                obs['points'], obs['samples'].reshape(-1, 1),
                obs['time_windows'].reshape(-1, 1), obs['priority'].reshape(-1, 1)
            ])
            point_tensor = torch.FloatTensor(point_features).to(device)

            actions = []
            list_states_with_comm = []
            action_masks = obs['action_masks']

            all_attn_probs, all_local_obs = [], []
            for i in range(num_agents):
                agent_obs_local_normalized = get_local_obs(obs, i, env)
                all_local_obs.append(agent_obs_local_normalized)
                agent_obs_local_tensor = torch.FloatTensor(agent_obs_local_normalized).to(device)
                with torch.no_grad():
                    attn_probs, _ = transformer(point_tensor, agent_obs_local_tensor)
                all_attn_probs.append(attn_probs.squeeze(0).cpu().detach().numpy())

            for i in range(num_agents):
                agent_obs_local_normalized = all_local_obs[i]
                attn_probs = all_attn_probs[i]
                agent_obs_full_normalized = np.concatenate([agent_obs_local_normalized, attn_probs])
                other_agents_last_actions = np.delete(last_actions_one_hot, i, axis=0).flatten()
                state_with_comm = np.concatenate([agent_obs_full_normalized, other_agents_last_actions])
                list_states_with_comm.append(state_with_comm)
                
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(state_with_comm).to(device)
                    mask_tensor = torch.BoolTensor(action_masks[i]).to(device)
                    action = agents[i].act(obs_tensor, noise_std=noise_std, action_mask=mask_tensor)
                actions.append(action)

            current_actions_one_hot = F.one_hot(torch.tensor(actions), num_classes=action_dim).numpy()
            next_obs, rewards, done = env.step(actions, num_steps, default_config.MAX_NUM_STEPS)
            
            next_point_features = np.hstack([
                next_obs['points'], next_obs['samples'].reshape(-1, 1),
                next_obs['time_windows'].reshape(-1, 1), next_obs['priority'].reshape(-1, 1)
            ])
            next_point_tensor = torch.FloatTensor(next_point_features).to(device)
            
            all_next_attn_probs = []
            for i in range(num_agents):
                next_agent_obs_local_normalized = get_local_obs(next_obs, i, env)
                next_agent_obs_local_tensor = torch.FloatTensor(next_agent_obs_local_normalized).to(device)
                with torch.no_grad():
                    next_attn_probs, _ = transformer(next_point_tensor, next_agent_obs_local_tensor)
                all_next_attn_probs.append(next_attn_probs.squeeze(0).cpu().detach().numpy())

            for i in range(num_agents):
                state_to_store = list_states_with_comm[i]
                next_state_local_normalized = get_local_obs(next_obs, i, env)
                next_attn_probs = all_next_attn_probs[i]
                next_state_full_normalized = np.concatenate([next_state_local_normalized, next_attn_probs])
                next_other_agents_actions = np.delete(current_actions_one_hot, i, axis=0).flatten()
                next_state_with_comm = np.concatenate([next_state_full_normalized, next_other_agents_actions])
                replay_buffers[i].push(state_to_store, actions[i], rewards[i], next_state_with_comm, float(done))

            last_actions_one_hot = current_actions_one_hot

            if step_counter > default_config.BATCH_SIZE and step_counter % default_config.UPDATE_EVERY == 0:
                for agent_id in range(num_agents):
                    if len(replay_buffers[agent_id]) > default_config.BATCH_SIZE:
                        states, actions_b, rewards_b, next_states, dones_b = replay_buffers[agent_id].sample(default_config.BATCH_SIZE)
                        rewards_b = (rewards_b - rewards_b.mean()) / (rewards_b.std() + 1e-7)
                        states, actions_b, rewards_b, next_states, dones_b = states.to(device), actions_b.to(device), rewards_b.to(device), next_states.to(device), dones_b.to(device)
                        agents[agent_id].update(states, actions_b, rewards_b, next_states, dones_b, action_dim)

            step_counter += 1
            total_rewards += rewards
            obs = next_obs
        
        noise_std = max(default_config.NOISE_STD_END, noise_std * noise_decay) # 使用 trial 建议的 noise_decay
        episode_rewards_log.append(total_rewards.sum()) # 记录所有智能体的奖励总和
        
        # 打印进度，便于监控
        print(f"Trial {trial.number}, Episode {episode}/{TUNING_NUM_EPISODES}, Total Reward: {total_rewards.sum():.2f}, Noise: {noise_std:.4f}")
    
    # --- 4. 返回评估指标 ---
    # 我们返回最后 50 个回合的平均奖励作为这次 trial 的最终得分
    # 这样做比只看最后一个回合的奖励更稳定
    final_avg_reward = np.mean(episode_rewards_log[-50:])
    return final_avg_reward


if __name__ == "__main__":
    # 创建一个研究 (study)，方向是 "maximize" (最大化回报)
    # 我们使用 TPE 采样器，这是一种高效的贝叶斯优化算法
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
    
    # 开始优化，n_trials 是要尝试的超参数组合的数量
    # 建议从一个较小的数字开始，例如 20-50，然后根据需要增加
    study.optimize(objective, n_trials=50)
    
    # 打印最佳结果
    print("\n" + "="*40)
    print("            OPTIMIZATION FINISHED            ")
    print("="*40)
    
    print(f"Number of finished trials: {len(study.trials)}")
    
    best_trial = study.best_trial
    
    print("\n--- Best Trial ---")
    print(f"  Value (Avg Reward): {best_trial.value:.4f}")
    
    print("  Parameters: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # 你还可以可视化结果 (需要安装 plotly)
    # pip install plotly
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.show()
        
        fig2 = optuna.visualization.plot_param_importances(study)
        fig2.show()
    except (ImportError, RuntimeError):
        print("\nCould not generate plots. Please install plotly: `pip install plotly`")


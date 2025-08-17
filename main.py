import torch
import torch.nn.functional as F
import numpy as np
from env import SamplingEnv2_0
from agent import TransformerEncoder, MADDPGAgent # Assuming agent.py now contains the optimized versions
from replay_buffer import ReplayBuffer
from utils import render_env, plot_rewards, plot_reward_curve2, plot_agent_trajectories
import config
import os

'''
后台运行命令：
CUDA_VISIBLE_DEVICES=3 nohup python main.py > output.txt 2>&1 &
'''

def get_local_obs(obs, agent_id, env):
    """
    Helper function to extract and normalize local observations for a single agent.
    """
    agent_pos_normalized = obs['agent_positions'][agent_id] / env.size
    agent_load_normalized = obs['agent_loads'][agent_id] / (env.agent_capacity[agent_id] + 1e-7)
    agent_energy_normalized = obs['agent_energy'][agent_id] / (env.agent_energy_max[agent_id] + 1e-7)
    
    # 根据智能体类型使用不同的最大充电时间进行归一化
    max_charge_time = env.fast_charge_time if agent_id < 3 else env.slow_charge_time
    agent_charging_status_normalized = obs['agent_charging_status'][agent_id] / max_charge_time
    
    agent_obs_local_normalized = np.concatenate([
        agent_pos_normalized,
        [agent_load_normalized, agent_energy_normalized, obs['agent_task_status'][agent_id], agent_charging_status_normalized]
    ])
    return agent_obs_local_normalized


def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建文件夹
    if not os.path.exists("trajectories"):
        os.makedirs("trajectories")
    if not os.path.exists("reward_plots"):
        os.makedirs("reward_plots")

    env = SamplingEnv2_0(num_points=config.NUM_POINTS, num_agents=config.NUM_AGENTS)
    num_agents = config.NUM_AGENTS
    num_points = config.NUM_POINTS
    action_dim = env.action_dim
    
    # 定义状态和观测维度
    agent_local_obs_dim = 6  # [pos(2), load(1), energy(1), task_status(1), charge_status(1)]
    point_feature_dim = 5  # [pos(2), samples(1), time_windows(1), priority(1)]
    
    # 智能体的完整观测维度 = 本地观测 + 注意力权重 + 其他智能体动作
    agent_obs_dim = agent_local_obs_dim + num_points + (num_agents - 1) * action_dim
    
    # 实例化支持动态Query的Transformer
    transformer = TransformerEncoder(point_input_dim=point_feature_dim, agent_local_obs_dim=agent_local_obs_dim, emb_dim=64).to(device)
    
    # 实例化智能体 (假设 agent.py 已更新为优化后的 Actor 和 Critic)
    agents = [MADDPGAgent(agent_obs_dim=agent_obs_dim, action_dim=action_dim).to(device) for _ in range(num_agents)]

    replay_buffers = [ReplayBuffer(capacity=config.CAPACITY) for _ in range(num_agents)]

    episode_rewards_log = []
    step_counter = 0
    noise_std = config.NOISE_STD_START

    for episode in range(config.NUM_EPISODES):
        obs = env.reset()
        done = False
        total_rewards = np.zeros(num_agents)

        # 初始化上一时刻的动作为"等待"
        last_actions_one_hot = F.one_hot(torch.full((num_agents,), num_points), num_classes=action_dim).numpy()

        num_steps = 0
        while (not done) and (num_steps < config.MAX_NUM_STEPS):
            num_steps += 1
            
            # 准备任务点特征
            point_features = np.hstack([
                obs['points'],
                obs['samples'].reshape(-1, 1),
                obs['time_windows'].reshape(-1, 1),
                obs['priority'].reshape(-1, 1)
            ])
            point_tensor = torch.FloatTensor(point_features).to(device)

            actions = []
            list_states_with_comm = [] # 存储包含通信信息的完整状态
            action_masks = obs['action_masks']

            # --- 核心改动: 为每个智能体独立计算注意力 ---
            all_attn_probs = []
            all_local_obs = []
            for i in range(num_agents):
                agent_obs_local_normalized = get_local_obs(obs, i, env)
                all_local_obs.append(agent_obs_local_normalized)
                agent_obs_local_tensor = torch.FloatTensor(agent_obs_local_normalized).to(device)
                
                with torch.no_grad():
                    # 传入任务点和个体化的局部观测以生成注意力
                    attn_probs, _ = transformer(point_tensor, agent_obs_local_tensor)
                all_attn_probs.append(attn_probs.squeeze(0).cpu().detach().numpy())

            # --- 基于个体化的注意力进行决策 ---
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
            next_obs, rewards, done = env.step(actions, num_steps, config.MAX_NUM_STEPS)
            
            # --- 核心改动: 为每个智能体独立计算下一时刻的注意力 ---
            next_point_features = np.hstack([
                next_obs['points'],
                next_obs['samples'].reshape(-1, 1),
                next_obs['time_windows'].reshape(-1, 1),
                next_obs['priority'].reshape(-1, 1)
            ])
            next_point_tensor = torch.FloatTensor(next_point_features).to(device)
            
            all_next_attn_probs = []
            for i in range(num_agents):
                next_agent_obs_local_normalized = get_local_obs(next_obs, i, env)
                next_agent_obs_local_tensor = torch.FloatTensor(next_agent_obs_local_normalized).to(device)
                with torch.no_grad():
                    next_attn_probs, _ = transformer(next_point_tensor, next_agent_obs_local_tensor)
                all_next_attn_probs.append(next_attn_probs.squeeze(0).cpu().detach().numpy())


            # --- 存储包含个体化注意力的经验 ---
            for i in range(num_agents):
                state_to_store = list_states_with_comm[i]
                
                # 构造下一时刻的完整状态 s'
                next_state_local_normalized = get_local_obs(next_obs, i, env)
                next_attn_probs = all_next_attn_probs[i]
                next_state_full_normalized = np.concatenate([next_state_local_normalized, next_attn_probs])
                
                next_other_agents_actions = np.delete(current_actions_one_hot, i, axis=0).flatten()
                next_state_with_comm = np.concatenate([next_state_full_normalized, next_other_agents_actions])

                replay_buffers[i].push(state_to_store, actions[i], rewards[i], next_state_with_comm, float(done))

            last_actions_one_hot = current_actions_one_hot

            # --- 训练步骤 (无需改动) ---
            if step_counter > config.BATCH_SIZE and step_counter % config.UPDATE_EVERY == 0:
                for agent_id in range(num_agents):
                    if len(replay_buffers[agent_id]) > config.BATCH_SIZE:
                        states, actions_b, rewards_b, next_states, dones_b = replay_buffers[agent_id].sample(config.BATCH_SIZE)
                        rewards_b = (rewards_b - rewards_b.mean()) / (rewards_b.std() + 1e-7)
                        states, actions_b, rewards_b, next_states, dones_b = states.to(device), actions_b.to(device), rewards_b.to(device), next_states.to(device), dones_b.to(device)
                        agents[agent_id].update(states, actions_b, rewards_b, next_states, dones_b, action_dim)

            step_counter += 1
            total_rewards += rewards
            obs = next_obs

        noise_std = max(config.NOISE_STD_END, noise_std * config.NOISE_DECAY)
        charge_counts = env.agent_charge_counts
        print(f"Episode {episode}: total_rewards = {total_rewards}, noise = {noise_std:.4f}, charge_counts = {charge_counts}")
        episode_rewards_log.append(total_rewards.tolist())
        
        # 记录每个智能体的最终路径点
        for i in range(num_agents):
            final_position = obs['agent_positions'][i]
            if env.agent_paths[i]:
                 env.agent_paths[i].append(final_position)

        if (episode + 1) % 50 == 0:
            plot_agent_trajectories(env.agent_paths, env.points, env.central_station, episode_id=episode+1, foldername="trajectories")

        if (episode + 1) % 100 == 0 and episode > 0:
            print(f"\n--- Plotting rewards at episode {episode + 1} ---\n")
            plot_rewards(episode_rewards_log, filename=f"reward_plots/reward_curve_ep{episode+1}.png")
            plot_reward_curve2(episode_rewards_log, filename=f"reward_plots/reward_curve2_ep{episode+1}.png")

    print("\n--- Plotting final rewards ---\n")
    plot_rewards(episode_rewards_log, filename="reward_plots/reward_curve_final.png")
    plot_reward_curve2(episode_rewards_log, filename="reward_plots/reward_curve2_final.png")

    all_rewards = np.array(episode_rewards_log)
    np.save('rewards_log.npy', all_rewards)
    
if __name__ == "__main__":
    main()

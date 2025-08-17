import torch
import torch.nn.functional as F
import numpy as np
from env import SamplingEnv2_0
from env_simplified import SamplingEnvSimplified
from env_configurable import SamplingEnvConfigurable
from agent import TransformerEncoder, MADDPGAgent # Assuming agent.py now contains the optimized versions
from agent_enhanced import MADDPGAgentEnhanced  # 增强版Critic支持
from replay_buffer import PrioritizedReplayBuffer
from utils import render_env, plot_rewards, plot_reward_curve2, plot_agent_trajectories, debug_print
from lr_scheduler import LearningRateScheduler
import config
import os
import random

'''
后台运行命令：
CUDA_VISIBLE_DEVICES=3 nohup python main.py > output.txt 2>&1 &
'''
def set_seed(seed):
    """
    设置项目中所有相关的随机数种子以确保实验的可复现性。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 适用于多GPU情况
    
    # 关键步骤：确保cuDNN的确定性行为
    # 这可能会对性能有轻微影响，但对于可复现性至关重要
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"✅ 所有随机数种子已设置为: {seed}")


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
    
    # 添加每个任务点的处理状态信息（同步给所有智能体）
    done_points_info = obs['done_points'].copy()  # 0表示未处理，1表示已处理（采集或超时）
    
    # 添加全局任务完成信息
    agent_obs_local_normalized = np.concatenate([
        agent_pos_normalized,
        [agent_load_normalized, agent_energy_normalized, obs['agent_task_status'][agent_id], 
         agent_charging_status_normalized, obs['task_completion_ratio'], 
         obs['all_tasks_completed'], obs['total_loaded_agents']],
        done_points_info  # 新增：每个任务点的处理状态（采集或超时）
    ])
    return agent_obs_local_normalized

def main():
    set_seed(config.RANDOM_SEED)
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    debug_print(f"Using device: {device}")
    
    # 输出通信模式配置
    comm_mode = "即时通信(t)" if config.ENABLE_IMMEDIATE_COMMUNICATION else "延迟通信(t-1)"
    debug_print(f"🔄 智能体通信模式: {comm_mode}")
    
    # 创建文件夹 - 支持配置的输出目录
    trajectories_dir = getattr(config, 'TRAJECTORIES_DIR', "trajectories")
    reward_plots_dir = getattr(config, 'REWARD_PLOTS_DIR', "reward_plots")
    
    if not os.path.exists(trajectories_dir):
        os.makedirs(trajectories_dir)
    if not os.path.exists(reward_plots_dir):
        os.makedirs(reward_plots_dir)

    # 根据配置选择环境版本
    if config.ENVIRONMENT_TYPE == "simplified":
        env = SamplingEnvSimplified(num_points=config.NUM_POINTS, num_agents=config.NUM_AGENTS)
        debug_print("🔧 使用简化版环境 (基础调试模式)")
        env_type = "简化版"
    elif config.ENVIRONMENT_TYPE == "configurable":
        env = SamplingEnvConfigurable(num_points=config.NUM_POINTS, num_agents=config.NUM_AGENTS)
        debug_print("🧪 使用可配置奖励环境 (增量式实验模式)")
        env_type = f"可配置版-{config.REWARD_EXPERIMENT_LEVEL}"
    else:  # "full" or fallback
        env = SamplingEnv2_0(num_points=config.NUM_POINTS, num_agents=config.NUM_AGENTS)
        debug_print("🚀 使用完整版环境 (生产模式)")
        env_type = "完整版"
    num_agents = config.NUM_AGENTS
    num_points = config.NUM_POINTS
    action_dim = env.action_dim
    
    # 定义状态和观测维度
    agent_local_obs_dim = 9 + num_points  # [pos(2), load(1), energy(1), task_status(1), charge_status(1), task_completion_ratio(1), all_tasks_completed(1), total_loaded_agents(1)] + [done_points(30):任务点处理状态]
    point_feature_dim = 5  # [pos(2), samples(1), time_windows(1), priority(1)]
    
    # 智能体的完整观测维度 = 本地观测 + transformer特征 + 其他智能体局部观测
    # 这个维度现在由Agent内部管理
    
    # 实例化智能体 - 根据配置选择标准或增强版架构
    if config.USE_ENHANCED_CRITIC:
        debug_print("🔬 使用增强版Critic架构 (接收所有智能体动作信息)")
        AgentClass = MADDPGAgentEnhanced
    else:
        debug_print("🎯 使用标准MADDPG架构")
        AgentClass = MADDPGAgent
    
    agents = [AgentClass(
        agent_local_obs_dim=agent_local_obs_dim,
        point_feature_dim=point_feature_dim, 
        num_points=num_points,
        num_agents=num_agents,
        action_dim=action_dim,
        lr_actor=config.LR_ACTOR, 
        lr_critic=config.LR_CRITIC
    ).to(device) for _ in range(num_agents)]
    
    debug_print(f"🧠 已创建{num_agents}个集成Transformer的智能体，支持端到端训练")

    replay_buffers = [PrioritizedReplayBuffer(
        capacity=config.CAPACITY, 
        alpha=config.PER_ALPHA, 
        beta=config.PER_BETA, 
        beta_increment=config.PER_BETA_INCREMENT
    ) for _ in range(num_agents)]
    
    # 初始化学习率调度器
    lr_scheduler = LearningRateScheduler(config.LR_ACTOR, config.LR_CRITIC)

    episode_rewards_log = []
    step_counter = 0
    noise_std = config.NOISE_STD_START

    for episode in range(config.NUM_EPISODES):
        # 更新学习率
        if config.ENABLE_LR_SCHEDULING:
            actor_lr, critic_lr = lr_scheduler.get_learning_rates(episode)
            for agent in agents:
                agent.update_learning_rate(actor_lr, critic_lr)
            
            # 输出学习率信息
            if episode % 200 == 0:
                debug_print(f"Episode {episode}: Actor LR={actor_lr:.2e}, Critic LR={critic_lr:.2e}")
        
        obs = env.reset()
        done = False
        total_rewards = np.zeros(num_agents)

        # 初始化上一时刻的局部观测（在第一步时会被实际观测覆盖）
        last_local_obs = None

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

            # --- 新架构: 准备智能体决策所需的原始数据 ---
            all_local_obs = []
            for i in range(num_agents):
                agent_obs_local_normalized = get_local_obs(obs, i, env)
                all_local_obs.append(agent_obs_local_normalized)
            
            # 调试信息：验证任务点状态同步（每100步打印一次）
            if num_steps % 100 == 0:
                done_points_status = obs['done_points']
                processed_tasks = np.sum(done_points_status)
                total_tasks = len(done_points_status)
                debug_print(f"🔄 步骤 {num_steps}: 任务点状态同步 - 已处理: {processed_tasks}/{total_tasks} 个任务点")
                debug_print(f"   已处理的任务点ID: {np.where(done_points_status == 1)[0].tolist()}")
                debug_print(f"   所有智能体都能看到相同的任务点状态: {done_points_status.tolist()}")

            # --- 使用新的Agent接口进行决策 (内置Transformer) ---
            for i in range(num_agents):
                agent_local_obs_tensor = torch.FloatTensor(all_local_obs[i]).to(device)
                
                # 获取其他智能体的局部观测 - 可配置的通信模式
                if config.ENABLE_IMMEDIATE_COMMUNICATION:
                    # 即时通信模式：使用当前时刻其他智能体的状态
                    other_agents_obs = np.concatenate([all_local_obs[j] for j in range(num_agents) if j != i])
                else:
                    # 延迟通信模式：使用t-1时刻其他智能体的状态（更现实）
                    if last_local_obs is None:
                        # 第一步：使用当前所有智能体的局部观测
                        other_agents_obs = np.concatenate([all_local_obs[j] for j in range(num_agents) if j != i])
                    else:
                        # 后续步骤：使用上一时刻其他智能体的局部观测
                        other_agents_obs = np.concatenate([last_local_obs[j] for j in range(num_agents) if j != i])
                
                other_agents_obs_tensor = torch.FloatTensor(other_agents_obs).to(device)
                mask_tensor = torch.BoolTensor(action_masks[i]).to(device)
                
                # 调用新的Agent接口 - 内部会调用Transformer并进行端到端学习
                with torch.no_grad():
                    action = agents[i].act(
                        agent_local_obs=agent_local_obs_tensor,
                        point_features=point_tensor, 
                        other_agents_obs=other_agents_obs_tensor,
                        noise_std=noise_std, 
                        action_mask=mask_tensor
                    )
                actions.append(action)
                
                # 为了兼容现有的经验回放，我们仍需要构建完整的状态表示
                # 使用Agent的辅助方法生成完整状态
                full_state = agents[i]._process_state(agent_local_obs_tensor, point_tensor, other_agents_obs_tensor)
                list_states_with_comm.append(full_state.cpu().numpy())

            # 保存当前时刻的局部观测供下一时刻使用
            current_local_obs = all_local_obs.copy()
            next_obs, rewards, done = env.step(actions, num_steps, config.MAX_NUM_STEPS)
            
            # --- 构建下一状态表示用于经验存储 ---
            next_point_features = np.hstack([
                next_obs['points'],
                next_obs['samples'].reshape(-1, 1),
                next_obs['time_windows'].reshape(-1, 1),
                next_obs['priority'].reshape(-1, 1)
            ])
            next_point_tensor = torch.FloatTensor(next_point_features).to(device)
            
            # 获取下一时刻所有智能体的局部观测
            all_next_local_obs = []
            for i in range(num_agents):
                next_agent_obs_local_normalized = get_local_obs(next_obs, i, env)
                all_next_local_obs.append(next_agent_obs_local_normalized)

            # --- 存储经验到回放缓冲区 ---
            for i in range(num_agents):
                state_to_store = list_states_with_comm[i]
                
                # 构造下一时刻的完整状态表示
                next_agent_local_obs_tensor = torch.FloatTensor(all_next_local_obs[i]).to(device)
                
                # 获取下一状态的通信信息 - 确保状态转移的时间一致性
                if config.ENABLE_IMMEDIATE_COMMUNICATION:
                    # 即时通信模式：使用t+1时刻其他智能体的状态 (all_next_local_obs)
                    next_other_agents_obs = np.concatenate([all_next_local_obs[j] for j in range(num_agents) if j != i])
                else:
                    # 延迟通信模式：使用t时刻其他智能体的状态 (current_local_obs)
                    next_other_agents_obs = np.concatenate([current_local_obs[j] for j in range(num_agents) if j != i])
                
                next_other_agents_obs_tensor = torch.FloatTensor(next_other_agents_obs).to(device)
                
                # 使用Agent的辅助方法生成下一状态的完整表示
                with torch.no_grad():
                    next_state_with_comm = agents[i]._process_state(
                        next_agent_local_obs_tensor, 
                        next_point_tensor, 
                        next_other_agents_obs_tensor
                    ).cpu().numpy()

                # 为增强版Critic准备全局动作信息
                all_current_actions = np.array(actions) if config.USE_ENHANCED_CRITIC else None
                
                replay_buffers[i].push(
                    state_to_store, 
                    actions[i], 
                    rewards[i], 
                    next_state_with_comm, 
                    float(done),
                    all_current_actions,    # 所有智能体当前动作
                    None                    # next_all_actions将在训练时动态计算
                )

            # 更新上一时刻的局部观测
            last_local_obs = current_local_obs

            # --- 训练步骤 (使用优先经验回放) ---
            if step_counter > config.BATCH_SIZE and step_counter % config.UPDATE_EVERY == 0:
                for agent_id in range(num_agents):
                    if len(replay_buffers[agent_id]) > config.BATCH_SIZE:
                        # 从优先经验回放池采样 - 适配增强版Critic的新格式
                        sample_result = replay_buffers[agent_id].sample(config.BATCH_SIZE)
                        states, actions_b, rewards_b, next_states, dones_b, all_actions_b, next_all_actions_b, indices, is_weights = sample_result
                        
                        # 标准化奖励
                        rewards_b = (rewards_b - rewards_b.mean()) / (rewards_b.std() + 1e-7)
                        
                        # 转移到设备
                        states = states.to(device)
                        actions_b = actions_b.to(device) 
                        rewards_b = rewards_b.to(device)
                        next_states = next_states.to(device)
                        dones_b = dones_b.to(device)
                        is_weights = is_weights.to(device)
                        
                        # 根据配置选择不同的更新方式
                        if config.USE_ENHANCED_CRITIC:
                            # 增强版Critic：传递全局动作信息
                            all_actions_b = all_actions_b.to(device)
                            
                            # 动态计算所有智能体在下一状态的动作（使用目标网络）
                            next_all_actions_list = []
                            with torch.no_grad():
                                for temp_agent_id in range(num_agents):
                                    # 使用每个智能体的目标actor网络计算下一动作
                                    next_action_logits = agents[temp_agent_id].actor_target(next_states)
                                    next_action = torch.argmax(next_action_logits, dim=1)
                                    next_all_actions_list.append(next_action)
                            
                            # 将列表转换为Tensor [batch_size, num_agents]
                            next_all_actions_b = torch.stack(next_all_actions_list, dim=1).to(device)
                            
                            # Debug打印（仅在第一次更新时）
                            if step_counter == config.BATCH_SIZE + 1 and agent_id == 0:
                                debug_print(f"🔧 增强版Critic修复生效：动态计算next_all_actions shape: {next_all_actions_b.shape}")
                            
                            td_errors = agents[agent_id].update(
                                agent_id, states, actions_b, rewards_b, next_states, dones_b, 
                                all_actions_b, next_all_actions_b, is_weights
                            )
                        else:
                            # 标准MADDPG：原有更新方式
                            td_errors = agents[agent_id].update(states, actions_b, rewards_b, next_states, dones_b, action_dim, is_weights)
                        
                        # 更新优先级
                        replay_buffers[agent_id].update_priorities(indices, td_errors)

            step_counter += 1
            total_rewards += rewards
            obs = next_obs

        noise_std = max(config.NOISE_STD_END, noise_std * config.NOISE_DECAY)
        charge_counts = env.agent_charge_counts
        
        # 获取当前学习率用于显示
        lr_info = ""
        if config.ENABLE_LR_SCHEDULING:
            current_actor_lr, current_critic_lr = lr_scheduler.get_current_rates()
            lr_info = f", actor_lr = {current_actor_lr:.2e}, critic_lr = {current_critic_lr:.2e}"
        
        debug_print(f"Episode {episode}: total_rewards = {total_rewards}, noise = {noise_std:.4f}, charge_counts = {charge_counts}{lr_info}")
        
        # 输出任务点状态同步的最终结果
        final_done_points = obs['done_points']
        processed_count = np.sum(final_done_points)
        total_count = len(final_done_points)
        debug_print(f"   📊 [{env_type}环境] 任务点处理状态: {processed_count}/{total_count} 个任务点已处理 (处理率: {processed_count/total_count*100:.1f}%)")
        if processed_count == total_count:
            debug_print(f"   🎉 所有任务点均已处理！Transformer端到端学习已激活，任务点状态已同步")
        
        episode_rewards_log.append(total_rewards.tolist())
        
        # 定期输出协作分析摘要
        if (episode + 1) % 100 == 0:
            env.debug_print_collaboration_summary()
        
        # 记录每个智能体的最终路径点
        for i in range(num_agents):
            final_position = obs['agent_positions'][i]
            if env.agent_paths[i]:
                 env.agent_paths[i].append(final_position)

        if (episode + 1) % 50 == 0:
            plot_agent_trajectories(env.agent_paths, env.points, env.central_station, episode_id=episode+1, foldername=trajectories_dir)

        if (episode + 1) % 100 == 0 and episode > 0:
            debug_print(f"\n--- Plotting rewards at episode {episode + 1} ---\n")
            plot_rewards(episode_rewards_log, filename=f"{reward_plots_dir}/reward_curve_ep{episode+1}.png")
            plot_reward_curve2(episode_rewards_log, filename=f"{reward_plots_dir}/reward_curve2_ep{episode+1}.png")

    debug_print("\n--- Plotting final rewards ---") # 输出最终奖励曲线
    plot_rewards(episode_rewards_log, filename=f"{reward_plots_dir}/reward_curve_final.png")
    plot_reward_curve2(episode_rewards_log, filename=f"{reward_plots_dir}/reward_curve2_final.png")

    # 保存数据文件到输出目录
    output_dir = getattr(config, 'OUTPUT_DIR', ".")
    
    all_rewards = np.array(episode_rewards_log)
    rewards_log_path = os.path.join(output_dir, 'rewards_log.npy')
    np.save(rewards_log_path, all_rewards)
    debug_print(f"📊 奖励数据已保存: {rewards_log_path}")
    
    # 输出最终协作分析报告
    debug_print("\n" + "🎯" * 20)
    debug_print("最终协作分析报告")
    debug_print("🎯" * 20)
    env.debug_print_collaboration_summary()
    
    # 保存协作分析数据
    final_analytics = env.get_collaboration_analytics()
    collaboration_path = os.path.join(output_dir, 'collaboration_analytics.npy')
    np.save(collaboration_path, final_analytics)
    debug_print(f"📊 协作分析数据已保存: {collaboration_path}")
    
    # === 增量式实验结果保存 ===
    if config.ENVIRONMENT_TYPE == "configurable":
        from experiment_analyzer import ExperimentAnalyzer
        
        analyzer = ExperimentAnalyzer()
        
        # 准备训练配置信息
        training_config = {
            'num_episodes': config.NUM_EPISODES,
            'num_agents': num_agents,
            'num_points': num_points,
            'batch_size': config.BATCH_SIZE,
            'lr_actor': config.LR_ACTOR,
            'lr_critic': config.LR_CRITIC,
            'use_enhanced_critic': config.USE_ENHANCED_CRITIC,
            'communication_mode': "immediate" if config.ENABLE_IMMEDIATE_COMMUNICATION else "delayed"
        }
        
        # 计算最终性能指标
        final_rewards = np.array(episode_rewards_log)
        if final_rewards.ndim > 1:
            final_performance = np.mean(np.sum(final_rewards[-100:], axis=1))
            convergence_episode = len(final_rewards)
        else:
            final_performance = np.mean(final_rewards[-100:])
            convergence_episode = len(final_rewards)
        
        final_metrics = {
            'final_performance': final_performance,
            'convergence_episode': convergence_episode,
            'total_episodes': len(episode_rewards_log),
            'final_conflict_rate': final_analytics.get('conflict_rate', 0) if final_analytics else 0,
            'experiment_level': config.REWARD_EXPERIMENT_LEVEL
        }
        
        # 保存实验结果
        result_file = analyzer.save_experiment_result(
            experiment_level=config.REWARD_EXPERIMENT_LEVEL,
            episode_rewards=episode_rewards_log,
            collaboration_analytics=final_analytics,
            training_config=training_config,
            final_metrics=final_metrics
        )
        
        debug_print(f"\n🎯 增量式实验结果已保存，可使用以下命令进行分析:")
        debug_print(f"   python experiment_analyzer.py")
        debug_print(f"   或在Python中：from experiment_analyzer import analyze_latest_experiments; analyze_latest_experiments()")
    
if __name__ == "__main__":
    main()

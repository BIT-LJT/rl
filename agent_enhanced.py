"""
增强版MADDPG智能体 - 支持Critic接收所有智能体动作信息

基于经典MADDPG理论的Critic设计优化：
- Critic输入: (state, action, other_agents_actions) 而不是 (state, action)
- 这样Critic可以更容易学习Q函数，因为减少了对其他智能体意图的推断

使用方法：
在main.py中将MADDPGAgent替换为MADDPGAgentEnhanced
并修改经验回放以存储所有智能体的动作信息
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from agent import TransformerEncoder, Actor

class CriticEnhanced(nn.Module):
    """增强版Critic - 接收所有智能体的动作信息"""
    def __init__(self, agent_obs_dim, action_dim, num_agents, hidden_dim=256):
        super(CriticEnhanced, self).__init__()
        self.num_agents = num_agents
        
        # 第一个网络分支：处理state
        self.state_fc1 = nn.Linear(agent_obs_dim, hidden_dim)
        self.state_fc2 = nn.Linear(hidden_dim, hidden_dim // 2)

        # 第二个网络分支：合并state特征和所有智能体的动作
        all_actions_dim = action_dim * num_agents  # 所有智能体的动作
        self.concat_fc1 = nn.Linear(hidden_dim // 2 + all_actions_dim, hidden_dim)
        self.concat_fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, state, all_actions):
        """
        Args:
            state: 单个智能体的状态 [batch_size, agent_obs_dim]
            all_actions: 所有智能体的动作 [batch_size, num_agents * action_dim]
        """
        # 处理state
        state_feature = F.relu(self.state_fc1(state))
        state_feature = F.relu(self.state_fc2(state_feature))

        # 拼接state特征和所有智能体的动作
        x = torch.cat([state_feature, all_actions], dim=1)

        # 处理拼接后的向量
        x = F.relu(self.concat_fc1(x))
        return self.concat_fc2(x)

class MADDPGAgentEnhanced(nn.Module):
    """增强版MADDPG智能体 - Critic接收所有智能体动作信息"""
    
    def __init__(self, agent_local_obs_dim, point_feature_dim, num_points, num_agents, action_dim, hidden_dim=256, lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, tau=0.005):
        super().__init__()
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.agent_local_obs_dim = agent_local_obs_dim
        
        # 集成Transformer到Agent中
        self.transformer = TransformerEncoder(point_input_dim=point_feature_dim, agent_local_obs_dim=agent_local_obs_dim, emb_dim=64)
        
        # 计算Actor和Critic的输入维度
        transformer_output_dim = 64
        other_agents_obs_dim = (num_agents - 1) * agent_local_obs_dim
        full_obs_dim = agent_local_obs_dim + transformer_output_dim + other_agents_obs_dim
        
        self.actor = Actor(full_obs_dim, action_dim, hidden_dim)
        
        # 使用增强版Critic
        self.critic = CriticEnhanced(full_obs_dim, action_dim, num_agents, hidden_dim)
        self.actor_target = Actor(full_obs_dim, action_dim, hidden_dim)
        self.critic_target = CriticEnhanced(full_obs_dim, action_dim, num_agents, hidden_dim)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Transformer参数包含在Actor优化器中
        actor_params = list(self.actor.parameters()) + list(self.transformer.parameters())
        self.actor_optimizer = optim.Adam(actor_params, lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.gamma = gamma
        self.tau = tau
        
        self.initial_lr_actor = lr_actor
        self.initial_lr_critic = lr_critic

    def act(self, agent_local_obs, point_features, other_agents_obs, noise_std=0.1, action_mask=None):
        """决策接口 - 与原版本相同"""
        # 调用Transformer生成任务点注意力权重和特征
        attn_probs, transformer_features = self.transformer(point_features, agent_local_obs)
        
        # 使用注意力权重对任务点特征进行加权聚合
        weighted_features = torch.matmul(attn_probs, transformer_features).squeeze(0)
        
        # 拼接所有特征
        full_obs = torch.cat([agent_local_obs, weighted_features, other_agents_obs], dim=0)
        
        # 生成动作logits
        logits = self.actor(full_obs)

        if action_mask is not None:
            logits[~action_mask] = -1e10
        
        if noise_std > 0:
            gumbel_noise = torch.distributions.Gumbel(0, 1).sample(logits.size()).to(logits.device) * noise_std
            noisy_logits = logits + gumbel_noise
        else:
            noisy_logits = logits
            
        action = torch.argmax(noisy_logits).item()
        return action
    
    def _process_state(self, agent_local_obs, point_features, other_agents_obs):
        """状态处理辅助方法 - 与原版本相同"""
        with torch.no_grad():
            attn_probs, transformer_features = self.transformer(point_features, agent_local_obs)
            
        weighted_features = torch.matmul(attn_probs, transformer_features).squeeze(0)
        full_obs = torch.cat([agent_local_obs, weighted_features, other_agents_obs], dim=0)
        return full_obs

    def update(self, agent_id, states, actions, rewards, next_states, dones, all_actions, next_all_actions, is_weights=None):
        """
        增强版更新方法 - Critic接收所有智能体的动作信息
        
        Args:
            agent_id: 当前智能体的ID索引 [0, num_agents)
            states: 状态批次 [batch_size, full_obs_dim]
            actions: 当前智能体动作批次 [batch_size]
            rewards: 奖励批次 [batch_size]
            next_states: 下一状态批次 [batch_size, full_obs_dim]
            dones: 结束标志批次 [batch_size]
            all_actions: 所有智能体当前动作 [batch_size, num_agents]
            next_all_actions: 所有智能体下一动作 [batch_size, num_agents]
            is_weights: 重要性采样权重
        """
        # --- 更新 Critic ---
        next_actions_logits = self.actor_target(next_states)
        next_actions = torch.argmax(next_actions_logits, dim=1)
        
        # 构建所有智能体的下一动作（这里简化处理，实际应该是所有智能体的目标动作）
        # 这需要在训练循环中传入所有智能体的下一动作
        next_all_actions_one_hot = F.one_hot(next_all_actions, num_classes=self.action_dim).float()
        next_all_actions_flat = next_all_actions_one_hot.view(next_all_actions_one_hot.size(0), -1)

        with torch.no_grad():
            target_q = self.critic_target(next_states, next_all_actions_flat)
            target_q = rewards.unsqueeze(1) + self.gamma * target_q * (1 - dones.unsqueeze(1))

        # 当前所有智能体动作
        all_actions_one_hot = F.one_hot(all_actions, num_classes=self.action_dim).float()
        all_actions_flat = all_actions_one_hot.view(all_actions_one_hot.size(0), -1)
        
        current_q = self.critic(states, all_actions_flat)

        # TD误差
        td_errors = target_q - current_q
        
        if is_weights is not None:
            critic_loss = (is_weights.unsqueeze(1) * (td_errors ** 2)).mean()
        else:
            critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # --- 更新 Actor ---
        actor_actions_logits = self.actor(states)
        actor_actions_one_hot = F.gumbel_softmax(actor_actions_logits, hard=True)
        
        # 构建包含当前智能体新动作的所有动作
        # 动态替换当前agent_id的动作，修复硬编码bug
        modified_all_actions = all_actions_flat.clone()
        start_index = agent_id * self.action_dim
        end_index = (agent_id + 1) * self.action_dim
        modified_all_actions[:, start_index:end_index] = actor_actions_one_hot
        
        actor_loss = -self.critic(states, modified_all_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        
        all_actor_params = list(self.actor.parameters()) + list(self.transformer.parameters())
        torch.nn.utils.clip_grad_norm_(all_actor_params, 1.0)
        
        self.actor_optimizer.step()

        # 软更新目标网络
        self._soft_update(self.critic, self.critic_target)
        self._soft_update(self.actor, self.actor_target)
        
        return td_errors.detach().cpu().numpy().flatten()
    
    def _soft_update(self, local_model, target_model):
        """软更新目标网络"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
    
    def update_learning_rate(self, actor_lr, critic_lr):
        """更新学习率"""
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = actor_lr
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = critic_lr
    
    def get_learning_rates(self):
        """获取当前学习率"""
        actor_lr = self.actor_optimizer.param_groups[0]['lr']
        critic_lr = self.critic_optimizer.param_groups[0]['lr']
        return actor_lr, critic_lr

# 配置选项：是否使用增强版Agent
USE_ENHANCED_CRITIC = False  # 可在config.py中配置

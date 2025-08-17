import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# --- Transformer Encoder ---

class TransformerEncoder(nn.Module):
    def __init__(self, point_input_dim, agent_local_obs_dim, emb_dim=64, n_heads=4, n_layers=2):
        super().__init__()
        self.embedding = nn.Linear(point_input_dim, emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 移除固定的 self.query
        # self.query = nn.Parameter(torch.randn(1, emb_dim))
        
        # 新增一个网络，用于从智能体局部观测生成 query
        self.query_generator = nn.Sequential(
            nn.Linear(agent_local_obs_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, points_x, agent_local_obs):
        # agent_local_obs: [batch_size, agent_local_obs_dim]
        # points_x: [batch_size, num_points, point_input_dim]
        # 注意：为了适配 main.py 中的调用，我们假设批处理大小为1
        points_x = points_x.unsqueeze(0)  # Add batch dimension
        
        embedded_points = self.embedding(points_x)
        transformer_out = self.transformer(embedded_points)
        transformer_out = transformer_out.squeeze(0) # [num_points, emb_dim]

        # 从智能体局部观测动态生成 query
        # agent_local_obs 也需要增加一个 batch 维度
        agent_local_obs = agent_local_obs.unsqueeze(0)
        dynamic_query = self.query_generator(agent_local_obs) # [1, emb_dim]
        
        attn_scores = torch.matmul(dynamic_query, transformer_out.T)
        attn_probs = F.softmax(attn_scores, dim=-1)
        return attn_probs, transformer_out

# --- MADDPG Agent ---

class Actor(nn.Module):
    def __init__(self, agent_obs_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(agent_obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        # 增加 Tanh 激活函数来约束输出范围
        self.tanh = nn.Tanh()

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # 将 logits 通过 tanh 约束，然后可以乘以一个常数（例如5.0）来调整范围
        # 这个常数可以根据实验效果调整
        logits = self.fc3(x)
        scaled_logits = self.tanh(logits) * 5.0 
        return scaled_logits

class Critic(nn.Module):
    def __init__(self, agent_obs_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        # 增加网络的深度和宽度，可以适当增加 hidden_dim
        # 第一个网络分支，用于处理 state
        self.state_fc1 = nn.Linear(agent_obs_dim, hidden_dim)
        self.state_fc2 = nn.Linear(hidden_dim, hidden_dim // 2)

        # 第二个网络分支，用于合并 state 特征和 action
        self.concat_fc1 = nn.Linear(hidden_dim // 2 + action_dim, hidden_dim)
        self.concat_fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        # 先单独处理 state
        state_feature = F.relu(self.state_fc1(state))
        state_feature = F.relu(self.state_fc2(state_feature))

        # 然后将 state 特征与 action 拼接
        x = torch.cat([state_feature, action], dim=1)

        # 处理拼接后的向量
        x = F.relu(self.concat_fc1(x))
        return self.concat_fc2(x)

class MADDPGAgent(nn.Module):
    def __init__(self, agent_local_obs_dim, point_feature_dim, num_points, num_agents, action_dim, hidden_dim=256, lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, tau=0.005):
        super().__init__()
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.agent_local_obs_dim = agent_local_obs_dim
        
        # 集成Transformer到Agent中，实现端到端训练
        self.transformer = TransformerEncoder(point_input_dim=point_feature_dim, agent_local_obs_dim=agent_local_obs_dim, emb_dim=64)
        
        # 计算Actor和Critic的输入维度
        # 包含：agent本地观测 + transformer输出特征 + 其他智能体观测
        transformer_output_dim = 64  # transformer的emb_dim
        other_agents_obs_dim = (num_agents - 1) * agent_local_obs_dim
        full_obs_dim = agent_local_obs_dim + transformer_output_dim + other_agents_obs_dim
        
        self.actor = Actor(full_obs_dim, action_dim, hidden_dim)
        self.critic = Critic(full_obs_dim, action_dim, hidden_dim)
        self.actor_target = Actor(full_obs_dim, action_dim, hidden_dim)
        self.critic_target = Critic(full_obs_dim, action_dim, hidden_dim)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # 关键修复：将Transformer参数包含在优化器中，实现端到端训练
        actor_params = list(self.actor.parameters()) + list(self.transformer.parameters())
        self.actor_optimizer = optim.Adam(actor_params, lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.gamma = gamma
        self.tau = tau
        
        # 保存初始学习率用于学习率调度
        self.initial_lr_actor = lr_actor
        self.initial_lr_critic = lr_critic

    def act(self, agent_local_obs, point_features, other_agents_obs, noise_std=0.1, action_mask=None):
        """
        根据观测选择动作，内部调用Transformer进行端到端学习
        
        Args:
            agent_local_obs: 智能体本地观测 [agent_local_obs_dim]
            point_features: 任务点特征 [num_points, point_feature_dim]  
            other_agents_obs: 其他智能体观测 [other_agents_obs_dim]
            noise_std: 噪声标准差
            action_mask: 动作掩码
        """
        # 调用Transformer生成任务点注意力权重和特征
        attn_probs, transformer_features = self.transformer(point_features, agent_local_obs)
        
        # 使用注意力权重对任务点特征进行加权聚合，生成高级特征表示
        # transformer_features: [num_points, emb_dim]
        # attn_probs: [1, num_points]
        weighted_features = torch.matmul(attn_probs, transformer_features)  # [1, emb_dim]
        weighted_features = weighted_features.squeeze(0)  # [emb_dim]
        
        # 拼接所有特征：本地观测 + Transformer特征 + 其他智能体观测
        full_obs = torch.cat([agent_local_obs, weighted_features, other_agents_obs], dim=0)
        
        # 生成动作logits
        logits = self.actor(full_obs)

        # 应用动作掩码
        if action_mask is not None:
            # 将掩码中为False的位置对应的logits设置为一个非常小的数
            logits[~action_mask] = -1e10
        
        if noise_std > 0:
            gumbel_noise = torch.distributions.Gumbel(0, 1).sample(logits.size()).to(logits.device) * noise_std
            noisy_logits = logits + gumbel_noise
        else:
            noisy_logits = logits
            
        action = torch.argmax(noisy_logits).item()
        return action
    
    def _process_state(self, agent_local_obs, point_features, other_agents_obs):
        """
        处理状态的辅助方法，生成完整的状态表示
        
        Args:
            agent_local_obs: 智能体本地观测
            point_features: 任务点特征
            other_agents_obs: 其他智能体观测
            
        Returns:
            full_obs: 完整的状态表示
        """
        # 调用Transformer生成特征
        with torch.no_grad():
            attn_probs, transformer_features = self.transformer(point_features, agent_local_obs)
            
        # 加权聚合特征
        weighted_features = torch.matmul(attn_probs, transformer_features).squeeze(0)
        
        # 拼接完整状态
        full_obs = torch.cat([agent_local_obs, weighted_features, other_agents_obs], dim=0)
        return full_obs

    def update(self, states, actions, rewards, next_states, dones, action_dim, is_weights=None):
        """
        更新智能体网络 - 支持Transformer端到端训练
        
        注意：为了简化实现，假设states和next_states是已经处理过的完整状态表示
        Transformer的训练通过actor的反向传播实现
        
        Args:
            states: 状态批次 [batch_size, full_obs_dim]
            actions: 动作批次
            rewards: 奖励批次
            next_states: 下一状态批次 [batch_size, full_obs_dim]
            dones: 结束标志批次
            action_dim: 动作维度
            is_weights: 重要性采样权重（用于优先经验回放）
        
        Returns:
            td_errors: TD误差（用于更新优先级）
        """
        # --- 更新 Critic ---
        next_actions_logits = self.actor_target(next_states)
        next_actions = torch.argmax(next_actions_logits, dim=1)
        next_actions_one_hot = F.one_hot(next_actions, num_classes=action_dim).float()

        with torch.no_grad():
            target_q = self.critic_target(next_states, next_actions_one_hot)
            target_q = rewards.unsqueeze(1) + self.gamma * target_q * (1 - dones.unsqueeze(1))

        current_actions_one_hot = F.one_hot(actions.long(), num_classes=action_dim).float()
        current_q = self.critic(states, current_actions_one_hot)

        # 计算TD误差（用于优先级更新）
        td_errors = target_q - current_q
        
        # 如果使用优先经验回放，应用重要性采样权重
        if is_weights is not None:
            critic_loss = (is_weights.unsqueeze(1) * (td_errors ** 2)).mean()
        else:
            critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # --- 更新 Actor（包含Transformer的端到端训练）---
        actor_actions_logits = self.actor(states)
        actor_actions_one_hot = F.gumbel_softmax(actor_actions_logits, hard=True)
        actor_loss = -self.critic(states, actor_actions_one_hot).mean()

        # 关键：这里的反向传播会同时更新Actor和Transformer的参数
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        
        # 对所有参数进行梯度裁剪（包括Transformer）
        all_actor_params = list(self.actor.parameters()) + list(self.transformer.parameters())
        torch.nn.utils.clip_grad_norm_(all_actor_params, 1.0)
        
        self.actor_optimizer.step()

        # --- 软更新目标网络 ---
        self._soft_update(self.critic, self.critic_target)
        self._soft_update(self.actor, self.actor_target)
        
        # 返回TD误差用于优先级更新
        return td_errors.detach().cpu().numpy().flatten()
    
    def _soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
    
    def update_learning_rate(self, actor_lr, critic_lr):
        """
        更新学习率
        """
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = actor_lr
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = critic_lr
    
    def get_learning_rates(self):
        """
        获取当前学习率
        """
        actor_lr = self.actor_optimizer.param_groups[0]['lr']
        critic_lr = self.critic_optimizer.param_groups[0]['lr']
        return actor_lr, critic_lr

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
    def __init__(self, agent_obs_dim, action_dim, hidden_dim=256, lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, tau=0.005):
        super().__init__()
        self.action_dim = action_dim
        self.actor = Actor(agent_obs_dim, action_dim, hidden_dim)
        self.critic = Critic(agent_obs_dim, action_dim, hidden_dim)
        self.actor_target = Actor(agent_obs_dim, action_dim, hidden_dim)
        self.critic_target = Critic(agent_obs_dim, action_dim, hidden_dim)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.gamma = gamma
        self.tau = tau

    def act(self, obs, noise_std=0.1, action_mask=None):
        """
        根据观测选择动作, 并应用动作掩码.
        """
        logits = self.actor(obs)

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

    def update(self, states, actions, rewards, next_states, dones, action_dim):
        # --- 更新 Critic ---
        next_actions_logits = self.actor_target(next_states)
        # 注意: 在训练 critic 时, 我们不需要应用动作掩码, 因为我们是根据 actor_target 的输出来计算目标Q值.
        # 动作掩码主要用于“实际执行”阶段, 保证智能体不会做出无效动作.
        next_actions = torch.argmax(next_actions_logits, dim=1)
        next_actions_one_hot = F.one_hot(next_actions, num_classes=action_dim).float()

        with torch.no_grad():
            target_q = self.critic_target(next_states, next_actions_one_hot)
            target_q = rewards.unsqueeze(1) + self.gamma * target_q * (1 - dones.unsqueeze(1))

        current_actions_one_hot = F.one_hot(actions.long(), num_classes=action_dim).float()
        current_q = self.critic(states, current_actions_one_hot)

        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # --- 更新 Actor ---
        actor_actions_logits = self.actor(states)
        # 在更新 actor 时, 我们使用 Gumbel-Softmax 进行采样, 此时也不直接应用硬掩码.
        # Actor 的目标是最大化 Critic 的输出, 它会自然地学会避免那些导致低Q值的动作.
        # 由于我们已经在执行阶段避免了无效动作, replay buffer 中不会有这些“坏”的经验,
        # Actor 也不会被引导去学习它们.
        actor_actions_one_hot = F.gumbel_softmax(actor_actions_logits, hard=True)
        actor_loss = -self.critic(states, actor_actions_one_hot).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # --- 软更新目标网络 ---
        self._soft_update(self.critic, self.critic_target)
        self._soft_update(self.actor, self.actor_target)
    
    def _soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

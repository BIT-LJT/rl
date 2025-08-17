import random
import numpy as np
import torch
from collections import deque

class SumTree:
    """
    SumTree数据结构用于高效的优先级采样
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # 存储优先级的完全二叉树
        self.data = np.zeros(capacity, dtype=object)  # 存储实际经验数据
        self.write = 0  # 写入指针
        self.n_entries = 0  # 当前存储的经验数量

    def _propagate(self, idx, change):
        """向上传播优先级变化"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        """根据累积优先级检索叶子节点"""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        """返回所有优先级的总和"""
        return self.tree[0]

    def add(self, p, data):
        """添加经验和优先级"""
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        """更新优先级"""
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        """根据累积优先级获取经验"""
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

class PrioritizedReplayBuffer:
    """
    优先经验回放池
    """
    def __init__(self, capacity=100000, alpha=0.6, beta=0.4, beta_increment=0.001):
        """
        初始化优先经验回放池
        
        Args:
            capacity: 经验池容量
            alpha: 优先级指数（0=均匀采样，1=完全按优先级采样）
            beta: 重要性采样权重指数（0=无权重，1=完全补偿）
            beta_increment: beta的增长率
        """
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = 1e-6  # 避免零优先级
        self.max_priority = 1.0  # 最大优先级

    def push(self, state, action, reward, next_state, done, all_actions=None, next_all_actions=None):
        """
        添加经验到回放池
        
        Args:
            state: 当前状态
            action: 当前智能体的动作
            reward: 奖励
            next_state: 下一状态
            done: 结束标志
            all_actions: 所有智能体当前动作 (为增强版Critic准备)
            next_all_actions: 所有智能体下一动作 (为增强版Critic准备)
        """
        experience = (state, action, reward, next_state, done, all_actions, next_all_actions)
        # 新经验使用最大优先级
        self.tree.add(self.max_priority, experience)

    def sample(self, batch_size):
        """
        采样一个批次的经验
        
        Returns:
            states, actions, rewards, next_states, dones, indices, weights
        """
        batch_idx = []
        batch_experiences = []
        priorities = []
        
        # 计算采样区间
        priority_segment = self.tree.total() / batch_size
        
        # 更新beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        for i in range(batch_size):
            a = priority_segment * i
            b = priority_segment * (i + 1)
            s = random.uniform(a, b)
            
            idx, priority, experience = self.tree.get(s)
            batch_idx.append(idx)
            batch_experiences.append(experience)
            priorities.append(priority)
        
        # 计算重要性采样权重
        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()  # 归一化
        
        # 解包经验 - 支持增强版Critic的全局动作信息
        states, actions, rewards, next_states, dones, all_actions, next_all_actions = map(list, zip(*batch_experiences))
        
        # 转换为numpy数组，但先处理None值
        states = np.array(states)
        actions = np.array(actions) 
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        
        # 检查all_actions是否为None，如果是则创建占位符以保持兼容性
        if all_actions[0] is None:
            all_actions = np.zeros((len(batch_experiences), 1), dtype=np.int64)  # 占位符
            next_all_actions = np.zeros((len(batch_experiences), 1), dtype=np.int64)  # 占位符
        else:
            all_actions = np.array(all_actions)
            # 对于next_all_actions，如果是None则在训练时动态计算，这里先创建占位符
            if next_all_actions[0] is None:
                next_all_actions = np.zeros_like(all_actions)  # 占位符，将在训练时被替换
            else:
                next_all_actions = np.array(next_all_actions)
        
        return (torch.FloatTensor(states),
                torch.LongTensor(actions),
                torch.FloatTensor(rewards),
                torch.FloatTensor(next_states),
                torch.FloatTensor(dones),
                torch.LongTensor(all_actions),        # 新增：所有智能体当前动作
                torch.LongTensor(next_all_actions),   # 新增：所有智能体下一动作
                batch_idx,
                torch.FloatTensor(is_weights))

    def update_priorities(self, indices, priorities):
        """
        更新经验的优先级
        
        Args:
            indices: 经验在树中的索引
            priorities: 新的优先级（通常是TD误差）
        """
        for idx, priority in zip(indices, priorities):
            priority = abs(priority) + self.epsilon  # 确保优先级为正
            priority = priority ** self.alpha  # 应用alpha指数
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        """返回当前经验池中的经验数量"""
        return self.tree.n_entries

# 保持原有的ReplayBuffer以兼容性（如果需要的话）
class ReplayBuffer:
    def __init__(self, capacity=100000):
        """
        经验回放池的构造函数.
        
        Args:
            capacity (int): 经验池的最大容量.
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, all_actions=None, next_all_actions=None):
        """
        向经验池中添加一条经验.
        
        Args:
            state: 当前状态.
            action: 执行的动作.
            reward: 获得的奖励.
            next_state: 下一个状态.
            done: 是否结束.
            all_actions: 所有智能体当前动作 (为增强版Critic准备, 兼容性参数)
            next_all_actions: 所有智能体下一动作 (为增强版Critic准备, 兼容性参数)
        """
        self.buffer.append((state, action, reward, next_state, done, all_actions, next_all_actions))

    def sample(self, batch_size):
        """
        从经验池中随机采样一个批次的经验.
        
        Args:
            batch_size (int): 采样的批次大小.
            
        Returns:
            一个包含状态、动作、奖励、下一状态、完成标志和全局动作信息的元组 (tensors).
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, all_actions, next_all_actions = map(np.array, zip(*batch))
        
        # 检查all_actions是否为None，如果是则创建占位符以保持兼容性
        if all_actions[0] is None:
            all_actions = np.zeros((len(batch), 1), dtype=np.int64)  # 占位符
            next_all_actions = np.zeros((len(batch), 1), dtype=np.int64)  # 占位符
        
        return (torch.FloatTensor(state),
                torch.LongTensor(action),
                torch.FloatTensor(reward),
                torch.FloatTensor(next_state),
                torch.FloatTensor(done),
                torch.LongTensor(all_actions),        # 新增：兼容性支持
                torch.LongTensor(next_all_actions))   # 新增：兼容性支持

    def __len__(self):
        """
        返回当前经验池中的经验数量.
        这是解决 TypeError 的关键新增方法.
        """
        return len(self.buffer)

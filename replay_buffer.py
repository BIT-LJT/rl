import random
import numpy as np
import torch
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=100000):
        """
        经验回放池的构造函数.
        
        Args:
            capacity (int): 经验池的最大容量.
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        向经验池中添加一条经验.
        
        Args:
            state: 当前状态.
            action: 执行的动作.
            reward: 获得的奖励.
            next_state: 下一个状态.
            done: 是否结束.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        从经验池中随机采样一个批次的经验.
        
        Args:
            batch_size (int): 采样的批次大小.
            
        Returns:
            一个包含状态、动作、奖励、下一状态和完成标志的元组 (tensors).
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return (torch.FloatTensor(state),
                torch.LongTensor(action),
                torch.FloatTensor(reward),
                torch.FloatTensor(next_state),
                torch.FloatTensor(done))

    def __len__(self):
        """
        返回当前经验池中的经验数量.
        这是解决 TypeError 的关键新增方法.
        """
        return len(self.buffer)

"""
学习率调度器
"""
import math
import config
from utils import debug_print

class LearningRateScheduler:
    """
    学习率调度器，支持多种衰减策略
    """
    
    def __init__(self, initial_lr_actor, initial_lr_critic):
        self.initial_lr_actor = initial_lr_actor
        self.initial_lr_critic = initial_lr_critic
        self.current_lr_actor = initial_lr_actor
        self.current_lr_critic = initial_lr_critic
        
        # 最小学习率
        self.min_lr_actor = initial_lr_actor * config.LR_MIN_RATIO
        self.min_lr_critic = initial_lr_critic * config.LR_MIN_RATIO
        
    def get_learning_rates(self, episode):
        """
        根据当前回合数计算学习率
        """
        if not config.ENABLE_LR_SCHEDULING:
            return self.initial_lr_actor, self.initial_lr_critic
        
        # 预热阶段：线性增长到初始学习率
        if episode < config.LR_WARMUP_EPISODES:
            warmup_factor = (episode + 1) / config.LR_WARMUP_EPISODES
            actor_lr = self.initial_lr_actor * warmup_factor
            critic_lr = self.initial_lr_critic * warmup_factor
        else:
            # 正常训练阶段
            adjusted_episode = episode - config.LR_WARMUP_EPISODES
            
            if config.LR_DECAY_TYPE == "exponential":
                actor_lr, critic_lr = self._exponential_decay(adjusted_episode)
            elif config.LR_DECAY_TYPE == "step":
                actor_lr, critic_lr = self._step_decay(adjusted_episode)
            elif config.LR_DECAY_TYPE == "cosine":
                actor_lr, critic_lr = self._cosine_decay(adjusted_episode)
            else:
                # 默认使用指数衰减
                actor_lr, critic_lr = self._exponential_decay(adjusted_episode)
        
        # 确保不低于最小学习率
        actor_lr = max(actor_lr, self.min_lr_actor)
        critic_lr = max(critic_lr, self.min_lr_critic)
        
        self.current_lr_actor = actor_lr
        self.current_lr_critic = critic_lr
        
        return actor_lr, critic_lr
    
    def _exponential_decay(self, episode):
        """
        指数衰减：lr = initial_lr * (decay_rate ^ episode)
        """
        decay_factor = config.LR_DECAY_RATE ** episode
        actor_lr = self.initial_lr_actor * decay_factor
        critic_lr = self.initial_lr_critic * decay_factor
        return actor_lr, critic_lr
    
    def _step_decay(self, episode):
        """
        阶梯衰减：每隔一定回合降低学习率
        """
        decay_times = episode // config.LR_DECAY_EPISODES
        decay_factor = config.LR_DECAY_RATE ** decay_times
        actor_lr = self.initial_lr_actor * decay_factor
        critic_lr = self.initial_lr_critic * decay_factor
        return actor_lr, critic_lr
    
    def _cosine_decay(self, episode):
        """
        余弦衰减：平滑的余弦曲线衰减
        """
        total_episodes = config.NUM_EPISODES - config.LR_WARMUP_EPISODES
        cosine_factor = 0.5 * (1 + math.cos(math.pi * episode / total_episodes))
        
        actor_lr = self.min_lr_actor + (self.initial_lr_actor - self.min_lr_actor) * cosine_factor
        critic_lr = self.min_lr_critic + (self.initial_lr_critic - self.min_lr_critic) * cosine_factor
        return actor_lr, critic_lr
    
    def get_current_rates(self):
        """
        获取当前学习率
        """
        return self.current_lr_actor, self.current_lr_critic
    
    def log_lr_info(self, episode):
        """
        输出学习率信息
        """
        if episode % 100 == 0:  # 每100回合输出一次
            debug_print(f"Episode {episode}: LR_Actor={self.current_lr_actor:.2e}, LR_Critic={self.current_lr_critic:.2e}")

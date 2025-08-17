# Environment Settings
NUM_POINTS = 30  #任务点数量
NUM_AGENTS = 5

# Training Settings
NUM_EPISODES = 3000  # 训练回合数
MAX_NUM_STEPS = 1500 # 最大步数

BATCH_SIZE = 512 # 批量大小 - 针对11G显存优化
GAMMA = 0.999
UPDATE_EVERY = 1
CAPACITY =100000 # 经验回放容量

# Learning Rates
LR_ACTOR = 1e-6 # 演员初始学习率
LR_CRITIC = 5e-4 # 批评家初始学习率

# Learning Rate Scheduling Settings
ENABLE_LR_SCHEDULING = True  # 是否启用学习率调度
LR_DECAY_TYPE = "exponential"  # 学习率衰减类型: "exponential", "step", "cosine"
LR_DECAY_RATE = 0.995  # 指数衰减率 (每次更新乘以此值)
LR_DECAY_EPISODES = 100  # 每多少回合进行一次学习率衰减
LR_MIN_RATIO = 0.1  # 最小学习率比例 (最小学习率 = 初始学习率 * 此比例)
LR_WARMUP_EPISODES = 50  # 学习率预热回合数

# Exploration Noise Settings
NOISE_STD_START = 1.0 # 初始噪声标准差
NOISE_STD_END = 0.01 # 最终噪声标准差
NOISE_DECAY = 0.998 # 让噪声衰减得更慢，给予更多探索时间

# Prioritized Experience Replay Settings  
PER_ALPHA = 0.6  # 优先级指数（0=均匀采样，1=完全按优先级采样）
PER_BETA = 0.4   # 重要性采样权重指数（0=无权重，1=完全补偿）
PER_BETA_INCREMENT = 0.001  # beta的增长率
PER_EPSILON = 1e-6  # 避免零优先级的小值

# Target Policy Smoothing Settings (目标策略平滑设置)
TARGET_POLICY_NOISE_STD = 0.2  # 目标策略噪声标准差
TARGET_POLICY_NOISE_CLIP = 0.5  # 目标策略噪声裁剪范围

# Continuous Energy Penalty Settings (连续能量惩罚设置)
LOW_ENERGY_THRESHOLD = 0.3  # 低能量阈值（30%）
ENERGY_PENALTY_MULTIPLIER = 5.0  # 能量惩罚倍数

# Heavy Agent Load Efficiency Settings (重载智能体载重效率设置)
R_LOAD_EFFICIENCY = 20.0  # 重载智能体载重效率奖励系数（用于非线性奖励）

# Agent Intention Observation Settings (智能体意图观测设置)
ENABLE_AGENT_INTENTION_OBS = True  # 是否在观测中包含其他智能体的意图（上一时刻动作）

# TensorBoard Logging Settings (TensorBoard日志设置)
ENABLE_TENSORBOARD = True  # 是否启用TensorBoard记录
TENSORBOARD_LOG_INTERVAL = 1  # 每多少个episode记录一次基础指标
TENSORBOARD_DIAGNOSTIC_INTERVAL = 1   # 每多少次更新记录一次诊断信息（更频繁的诊断记录）
TENSORBOARD_COLLABORATION_INTERVAL = 1  # 每多少个episode记录一次协作分析指标（更频繁的协作分析）

# Model Checkpointing Settings (模型检查点设置)
ENABLE_MODEL_CHECKPOINTING = True  # 是否启用模型检查点保存
MODEL_SAVE_INTERVAL = 200  # 每多少个episode保存一次模型
CHECKPOINTS_DIR = "checkpoints"  # 模型检查点保存目录

# Delayed Policy Updates Settings (延迟策略更新设置 - TD3核心改进)
ENABLE_DELAYED_POLICY_UPDATES = True  # 是否启用延迟策略更新
ACTOR_UPDATE_FREQUENCY = 5 #Critic每更新N次，Actor才更新1次（推荐值：2-3）

# Intelligent Conflict Resolution Settings
ICR_PRIORITY_WEIGHT = 1.0      # 任务优先级权重
ICR_AGENT_TYPE_WEIGHT = 1.0    # 智能体类型匹配权重
ICR_DISTANCE_WEIGHT = 0.6      # 距离效率权重
ICR_LOAD_WEIGHT = 0.4          # 载重利用率权重
ICR_URGENCY_WEIGHT = 0.8       # 时间紧迫度权重
ICR_ENERGY_WEIGHT = 0.2        # 能量效率权重

# Visualization Settings
RENDER_INTERVAL = 1000 # 渲染间隔

# Debug Settings
DEBUG_PRINT = False  # True=启用打印输出, False=关闭打印输出

# Random Seed Settings
RANDOM_SEED = 123  # 随机种子，用于确保实验可重现性
# 推荐的实验种子集: [42, 123, 2024, 888, 1337]

# Communication Settings
ENABLE_IMMEDIATE_COMMUNICATION = True  # True=即时通信(t), False=延迟通信(t-1)

# Environment Settings
ENVIRONMENT_TYPE = "full"  # "configurable"=可配置奖励环境, "simplified"=简化环境, "full"=完整环境 
# USE_SIMPLIFIED_ENVIRONMENT = False  # 已弃用，请使用ENVIRONMENT_TYPE

# Agent Architecture Settings
USE_ENHANCED_CRITIC = True  # True=Critic接收所有智能体动作信息(实验性), False=标准MADDPG

# Incremental Reward Experiment Settings
from reward_config import RewardExperimentConfig
REWARD_EXPERIMENT_LEVEL = RewardExperimentConfig.LOAD_EFFICIENCY  # 增量式奖励实验等级
# 可选等级: BASIC, LOAD_EFFICIENCY, ROLE_SPECIALIZATION, COLLABORATION, BEHAVIOR_SHAPING, FULL

# 智能体参数 - 重新平衡设计
agent_capacity = [8, 8, 8, 24, 24]  # 智能体载重容量 (更大差距：3倍)
agent_speed = [30, 30, 30, 15, 15]  # 智能体移动速度 (缩小差距：2倍)
agent_energy_max = [60000, 60000, 60000, 150000, 150000]  # 智能体最大能量（重载智能体更多能量）
fast_charge_time = 3 * 60#快速充电时间
slow_charge_time = 8 * 60#慢速充电时间

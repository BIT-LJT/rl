# Environment Settings
NUM_POINTS = 30  #任务点数量
NUM_AGENTS = 5

# Training Settings
NUM_EPISODES = 3000  # 训练回合数
MAX_NUM_STEPS = 1500 # 最大步数

BATCH_SIZE = 512 # 批量大小 - 针对11G显存优化
GAMMA = 0.999
UPDATE_EVERY = 1
CAPACITY = 100000 # 经验回放容量

# Learning Rates
LR_ACTOR = 5e-5 # 演员初始学习率
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
NOISE_DECAY = 0.995 # 让噪声衰减得更慢，给予更多探索时间

# Prioritized Experience Replay Settings  
PER_ALPHA = 0.6  # 优先级指数（0=均匀采样，1=完全按优先级采样）
PER_BETA = 0.4   # 重要性采样权重指数（0=无权重，1=完全补偿）
PER_BETA_INCREMENT = 0.001  # beta的增长率
PER_EPSILON = 1e-6  # 避免零优先级的小值

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
ENABLE_IMMEDIATE_COMMUNICATION = False  # True=即时通信(t), False=延迟通信(t-1)

# Environment Settings
ENVIRONMENT_TYPE = "configurable"  # "configurable"=可配置奖励环境, "simplified"=简化环境, "full"=完整环境
# USE_SIMPLIFIED_ENVIRONMENT = False  # 已弃用，请使用ENVIRONMENT_TYPE

# Agent Architecture Settings
USE_ENHANCED_CRITIC = True  # True=Critic接收所有智能体动作信息(实验性), False=标准MADDPG

# Incremental Reward Experiment Settings
from reward_config import RewardExperimentConfig
REWARD_EXPERIMENT_LEVEL = RewardExperimentConfig.LOAD_EFFICIENCY  # 增量式奖励实验等级
# 可选等级: BASIC, LOAD_EFFICIENCY, ROLE_SPECIALIZATION, COLLABORATION, BEHAVIOR_SHAPING, FULL

# 智能体参数
agent_capacity = [10, 10, 10, 20, 20]  # 智能体载重容量
agent_speed = [40, 40, 40, 8, 8]  # 智能体移动速度 - 扩大到5倍差距
agent_energy_max = [80000, 80000, 80000, 120000, 120000]  # 智能体最大能量（更严格的设置）
fast_charge_time = 3 * 60#快速充电时间
slow_charge_time = 8 * 60#慢速充电时间

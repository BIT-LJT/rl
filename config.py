# Environment Settings
NUM_POINTS = 30  # 提升难度: 任务点增加到15个
NUM_AGENTS = 5

# Training Settings
NUM_EPISODES = 1500  # 增加训练回合数以适应更复杂的任务
MAX_NUM_STEPS = 28800 # 增加最大步数

BATCH_SIZE = 256
GAMMA = 0.999
UPDATE_EVERY = 1
CAPACITY = 100000

# Learning Rates
LR_ACTOR = 5e-5
LR_CRITIC = 5e-4

# Exploration Noise Settings
NOISE_STD_START = 1.0
NOISE_STD_END = 0.01
NOISE_DECAY = 0.999 # 让噪声衰减得更慢，给予更多探索时间

# Visualization Settings
RENDER_INTERVAL = 1000

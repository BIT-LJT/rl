"""
批量实验脚本 - 多种子增量式奖励实验

这个脚本自动运行所有实验等级的多种子实验，确保结果的统计可靠性。

使用方法：
1. 配置实验参数
2. 运行脚本：python batch_experiment.py
3. 等待所有实验完成
4. 使用多种子分析器分析结果

注意：完整实验可能需要数小时或数天时间，建议在性能良好的机器上运行。
"""

import os
import subprocess
import time
import json
from datetime import datetime, timedelta
from pathlib import Path

from reward_config import RewardExperimentConfig
from utils import debug_print

class BatchExperimentRunner:
    """批量实验运行器"""
    
    def __init__(self, 
                 seeds=[42, 123, 2024, 888, 1337],
                 experiment_levels=None,
                 episodes_per_experiment=2000,
                 results_dir="multi_seed_results"):
        """
        初始化批量实验运行器
        
        Args:
            seeds: 随机种子列表
            experiment_levels: 实验等级列表，None表示运行所有等级
            episodes_per_experiment: 每个实验的训练回合数
            results_dir: 结果存储目录
        """
        self.seeds = seeds
        self.episodes_per_experiment = episodes_per_experiment
        self.results_dir = results_dir
        
        # 默认运行所有实验等级
        if experiment_levels is None:
            self.experiment_levels = [
                RewardExperimentConfig.BASIC,
                RewardExperimentConfig.LOAD_EFFICIENCY,
                RewardExperimentConfig.ROLE_SPECIALIZATION,
                RewardExperimentConfig.COLLABORATION,
                RewardExperimentConfig.BEHAVIOR_SHAPING,
                RewardExperimentConfig.FULL
            ]
        else:
            self.experiment_levels = experiment_levels
        
        # 创建结果目录
        os.makedirs(results_dir, exist_ok=True)
        
        # 实验状态跟踪
        self.experiment_log = []
        self.start_time = None
        self.total_experiments = len(self.seeds) * len(self.experiment_levels)
        
    def run_all_experiments(self):
        """运行所有实验"""
        debug_print("🚀 开始批量增量式奖励实验")
        debug_print("=" * 80)
        debug_print(f"📊 实验配置:")
        debug_print(f"   种子数量: {len(self.seeds)}")
        debug_print(f"   实验等级: {len(self.experiment_levels)}")
        debug_print(f"   总实验数: {self.total_experiments}")
        debug_print(f"   每实验回合数: {self.episodes_per_experiment}")
        debug_print(f"   结果目录: {self.results_dir}")
        debug_print("=" * 80)
        
        self.start_time = datetime.now()
        completed_experiments = 0
        
        for level in self.experiment_levels:
            debug_print(f"\n🧪 开始实验等级: {level.upper()}")
            debug_print("-" * 60)
            
            for seed in self.seeds:
                debug_print(f"\n🎲 运行种子: {seed}")
                
                try:
                    # 记录实验开始
                    exp_start_time = datetime.now()
                    
                    # 运行单个实验
                    success = self._run_single_experiment(level, seed)
                    
                    # 记录实验结果
                    exp_end_time = datetime.now()
                    duration = exp_end_time - exp_start_time
                    
                    if success:
                        debug_print(f"✅ 实验完成 (耗时: {duration})")
                        completed_experiments += 1
                    else:
                        debug_print(f"❌ 实验失败 (耗时: {duration})")
                    
                    # 记录到日志
                    self.experiment_log.append({
                        'level': level,
                        'seed': seed,
                        'success': success,
                        'start_time': exp_start_time.isoformat(),
                        'end_time': exp_end_time.isoformat(),
                        'duration_seconds': duration.total_seconds()
                    })
                    
                    # 显示进度
                    progress = (completed_experiments / self.total_experiments) * 100
                    remaining = self.total_experiments - completed_experiments
                    if completed_experiments > 0:
                        avg_time_per_exp = (datetime.now() - self.start_time) / completed_experiments
                        estimated_remaining_time = avg_time_per_exp * remaining
                        debug_print(f"📈 进度: {completed_experiments}/{self.total_experiments} ({progress:.1f}%)")
                        debug_print(f"⏱️ 预计剩余时间: {estimated_remaining_time}")
                    
                except Exception as e:
                    debug_print(f"💥 实验异常: {e}")
                    self.experiment_log.append({
                        'level': level,
                        'seed': seed,
                        'success': False,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    })
        
        # 保存实验日志
        self._save_experiment_log()
        
        # 输出总结
        total_time = datetime.now() - self.start_time
        success_rate = (completed_experiments / self.total_experiments) * 100
        
        debug_print("\n" + "🎉" * 80)
        debug_print("批量实验完成总结")
        debug_print("🎉" * 80)
        debug_print(f"✅ 成功完成: {completed_experiments}/{self.total_experiments} ({success_rate:.1f}%)")
        debug_print(f"⏱️ 总耗时: {total_time}")
        debug_print(f"📁 结果目录: {self.results_dir}")
        debug_print(f"📊 下一步: 运行 python multi_seed_analyzer.py 分析结果")
        debug_print("🎉" * 80)
        
        return completed_experiments, self.total_experiments
    
    def _run_single_experiment(self, level, seed):
        """
        运行单个实验
        
        Args:
            level: 实验等级
            seed: 随机种子
            
        Returns:
            bool: 是否成功
        """
        try:
            # 预先创建独立的实验目录
            exp_output_dir = self._prepare_experiment_directory(level, seed)
            
            # 修改配置文件，指定输出目录
            self._update_config(level, seed, exp_output_dir)
            
            # 运行主程序
            debug_print(f"   📋 配置: 等级={level}, 种子={seed}, 回合={self.episodes_per_experiment}")
            debug_print(f"   📁 输出目录: {exp_output_dir}")
            
            result = subprocess.run(
                ["python", "main.py"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                debug_print(f"   ✅ 训练完成")
                # 清理临时文件（如果有的话）
                self._cleanup_temp_files()
                return True
            else:
                debug_print(f"   ❌ 训练失败: {result.stderr}")
                return False
                

        except Exception as e:
            debug_print(f"   💥 运行异常: {e}")
            return False
    
    def _prepare_experiment_directory(self, level, seed):
        """
        为实验预先创建独立的目录结构
        
        Args:
            level: 实验等级
            seed: 随机种子
            
        Returns:
            str: 实验输出目录路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建实验特定目录
        exp_name = f"{level}_seed{seed}_{timestamp}"
        exp_dir = os.path.join(self.results_dir, level, exp_name)
        
        # 创建所有必要的子目录
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(os.path.join(exp_dir, "reward_plots"), exist_ok=True)
        os.makedirs(os.path.join(exp_dir, "trajectories"), exist_ok=True)
        
        debug_print(f"   📁 已创建实验目录: {exp_dir}")
        
        return exp_dir
    
    def _cleanup_temp_files(self):
        """清理可能残留的临时文件"""
        temp_files = [
            "rewards_log.npy",
            "collaboration_analytics.npy"
        ]
        
        for filename in temp_files:
            if os.path.exists(filename):
                try:
                    os.remove(filename)
                    debug_print(f"   🧹 清理临时文件: {filename}")
                except Exception as e:
                    debug_print(f"   ⚠️ 清理文件失败 {filename}: {e}")
    
    def _update_config(self, level, seed, output_dir):
        """更新配置文件"""
        config_content = f"""# Environment Settings
NUM_POINTS = 30  #任务点数量
NUM_AGENTS = 5

# Training Settings
NUM_EPISODES = {self.episodes_per_experiment}  # 训练回合数
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
RANDOM_SEED = {seed}  # 随机种子，用于确保实验可重现性
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
REWARD_EXPERIMENT_LEVEL = RewardExperimentConfig.{level.upper()}  # 增量式奖励实验等级
# 可选等级: BASIC, LOAD_EFFICIENCY, ROLE_SPECIALIZATION, COLLABORATION, BEHAVIOR_SHAPING, FULL

# Output Directory Settings (用于批量实验)
OUTPUT_DIR = r"{output_dir}"  # 实验输出目录
REWARD_PLOTS_DIR = r"{os.path.join(output_dir, 'reward_plots')}"  # 奖励曲线图目录
TRAJECTORIES_DIR = r"{os.path.join(output_dir, 'trajectories')}"  # 轨迹图目录

# 智能体参数
agent_capacity = [10, 10, 10, 20, 20]  # 智能体载重容量
agent_speed = [40, 40, 40, 8, 8]  # 智能体移动速度 - 扩大到5倍差距
agent_energy_max = [80000, 80000, 80000, 120000, 120000]  # 智能体最大能量（更严格的设置）
fast_charge_time = 3 * 60#快速充电时间
slow_charge_time = 8 * 60#慢速充电时间
"""
        
        with open("config.py", "w", encoding="utf-8") as f:
            f.write(config_content)
    

    def _save_experiment_log(self):
        """保存实验日志"""
        log_file = os.path.join(self.results_dir, "experiment_log.json")
        
        log_data = {
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'total_experiments': self.total_experiments,
            'seeds': self.seeds,
            'experiment_levels': self.experiment_levels,
            'episodes_per_experiment': self.episodes_per_experiment,
            'experiments': self.experiment_log
        }
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        
        debug_print(f"📝 实验日志已保存: {log_file}")

def run_quick_test():
    """运行快速测试（少数种子和回合）"""
    debug_print("🧪 运行快速测试模式")
    
    runner = BatchExperimentRunner(
        seeds=[42, 123],  # 只用2个种子
        experiment_levels=[
            RewardExperimentConfig.BASIC,
            RewardExperimentConfig.LOAD_EFFICIENCY
        ],  # 只测试2个等级
        episodes_per_experiment=500,  # 较少的回合数
        results_dir="quick_test_results"
    )
    
    return runner.run_all_experiments()

def run_full_experiments():
    """运行完整实验"""
    debug_print("🚀 运行完整实验模式")
    
    runner = BatchExperimentRunner(
        seeds=[42, 123, 2024, 888, 1337],  # 5个种子
        experiment_levels=None,  # 所有等级
        episodes_per_experiment=3000,  # 完整回合数
        results_dir="full_experiment_results"
    )
    
    return runner.run_all_experiments()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="批量增量式奖励实验")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick",
                       help="实验模式: quick=快速测试, full=完整实验")
    
    args = parser.parse_args()
    
    if args.mode == "quick":
        run_quick_test()
    else:
        run_full_experiments()

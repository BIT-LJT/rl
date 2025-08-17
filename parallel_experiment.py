#!/usr/bin/env python3
"""
并行实验脚本 - 同时运行6个实验等级
支持多进程并行执行，每个等级独立运行，避免相互干扰

使用方法：
python parallel_experiment.py --seeds 42,123,2024 --episodes 3000
"""

import os
import subprocess
import multiprocessing as mp
import time
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path
import argparse
import psutil

from reward_config import RewardExperimentConfig
from utils import debug_print

class ParallelExperimentRunner:
    """并行实验运行器"""
    
    def __init__(self, 
                 seeds=[42, 123, 2024, 888, 1337],
                 episodes_per_experiment=3000,
                 results_dir="parallel_experiment_results",
                 gpu_memory_limit=0.85):
        """
        初始化并行实验运行器
        
        Args:
            seeds: 随机种子列表
            episodes_per_experiment: 每个实验的训练回合数
            results_dir: 结果存储目录
            gpu_memory_limit: GPU显存使用限制（0-1之间）
        """
        self.seeds = seeds
        self.episodes_per_experiment = episodes_per_experiment
        self.results_dir = results_dir
        self.gpu_memory_limit = gpu_memory_limit
        
        # 所有实验等级
        self.experiment_levels = [
            RewardExperimentConfig.BASIC,
            RewardExperimentConfig.LOAD_EFFICIENCY,
            RewardExperimentConfig.ROLE_SPECIALIZATION,
            RewardExperimentConfig.COLLABORATION,
            RewardExperimentConfig.BEHAVIOR_SHAPING,
            RewardExperimentConfig.FULL
        ]
        
        # 创建结果目录和临时文件目录
        os.makedirs(results_dir, exist_ok=True)
        self.temp_dir = os.path.join(results_dir, "temp_files")
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # 状态跟踪
        self.start_time = None
        self.process_status = {}
        
    def run_single_level_all_seeds(self, level):
        """
        运行单个等级的所有种子实验
        
        Args:
            level: 实验等级
            
        Returns:
            dict: 运行结果统计
        """
        process_id = os.getpid()
        debug_print(f"🚀 进程 {process_id}: 开始运行 {level.upper()} 等级实验")
        
        results = {
            'level': level,
            'process_id': process_id,
            'start_time': datetime.now().isoformat(),
            'seeds_results': [],
            'total_success': 0,
            'total_failed': 0
        }
        
        for seed in self.seeds:
            debug_print(f"   🎲 进程 {process_id}: 运行种子 {seed}")
            
            try:
                # 创建独立的实验目录
                exp_dir = self._prepare_experiment_directory(level, seed, process_id)
                
                # 生成配置文件（每个进程使用独立的配置文件）
                config_file = os.path.join(self.temp_dir, f"config_{level}_{process_id}.py")
                self._create_config_file(level, seed, exp_dir, config_file)
                
                # 运行实验
                success = self._run_experiment(config_file, process_id, level, seed)
                
                seed_result = {
                    'seed': seed,
                    'success': success,
                    'experiment_dir': exp_dir,
                    'timestamp': datetime.now().isoformat()
                }
                
                results['seeds_results'].append(seed_result)
                
                if success:
                    results['total_success'] += 1
                    debug_print(f"   ✅ 进程 {process_id}: 种子 {seed} 完成")
                else:
                    results['total_failed'] += 1
                    debug_print(f"   ❌ 进程 {process_id}: 种子 {seed} 失败")
                
                # 清理配置文件
                if os.path.exists(config_file):
                    os.remove(config_file)
                    
            except Exception as e:
                debug_print(f"   💥 进程 {process_id}: 种子 {seed} 异常: {e}")
                results['total_failed'] += 1
        
        results['end_time'] = datetime.now().isoformat()
        results['duration'] = (datetime.fromisoformat(results['end_time']) - 
                             datetime.fromisoformat(results['start_time'])).total_seconds()
        
        debug_print(f"🏁 进程 {process_id}: {level.upper()} 等级完成 - 成功: {results['total_success']}, 失败: {results['total_failed']}")
        return results
    
    def _prepare_experiment_directory(self, level, seed, process_id):
        """为实验创建独立目录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{level}_seed{seed}_proc{process_id}_{timestamp}"
        exp_dir = os.path.join(self.results_dir, level, exp_name)
        
        # 创建目录结构
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(os.path.join(exp_dir, "reward_plots"), exist_ok=True)
        os.makedirs(os.path.join(exp_dir, "trajectories"), exist_ok=True)
        
        return exp_dir
    
    def _create_config_file(self, level, seed, output_dir, config_filename):
        """创建独立的配置文件"""
        config_content = f"""# Environment Settings
NUM_POINTS = 30
NUM_AGENTS = 5

# Training Settings
NUM_EPISODES = {self.episodes_per_experiment}
MAX_NUM_STEPS = 1500

BATCH_SIZE = 512  # 针对11G显存优化
GAMMA = 0.999
UPDATE_EVERY = 1
CAPACITY = 100000

# Learning Rates
LR_ACTOR = 5e-5
LR_CRITIC = 5e-4

# Learning Rate Scheduling Settings
ENABLE_LR_SCHEDULING = True
LR_DECAY_TYPE = "exponential"
LR_DECAY_RATE = 0.995
LR_DECAY_EPISODES = 100
LR_MIN_RATIO = 0.1
LR_WARMUP_EPISODES = 50

# Exploration Noise Settings
NOISE_STD_START = 1.0
NOISE_STD_END = 0.01
NOISE_DECAY = 0.995

# Prioritized Experience Replay Settings
PER_ALPHA = 0.6
PER_BETA = 0.4
PER_BETA_INCREMENT = 0.001
PER_EPSILON = 1e-6

# Intelligent Conflict Resolution Settings
ICR_PRIORITY_WEIGHT = 1.0
ICR_AGENT_TYPE_WEIGHT = 1.0
ICR_DISTANCE_WEIGHT = 0.6
ICR_LOAD_WEIGHT = 0.4
ICR_URGENCY_WEIGHT = 0.8
ICR_ENERGY_WEIGHT = 0.2

# Visualization Settings
RENDER_INTERVAL = 1000

# Debug Settings
DEBUG_PRINT = False  # 并行实验中启用打印以便监控

# Random Seed Settings
RANDOM_SEED = {seed}

# Communication Settings
ENABLE_IMMEDIATE_COMMUNICATION = False

# Environment Settings
ENVIRONMENT_TYPE = "configurable"

# Agent Architecture Settings
USE_ENHANCED_CRITIC = True

# Incremental Reward Experiment Settings
from reward_config import RewardExperimentConfig
REWARD_EXPERIMENT_LEVEL = RewardExperimentConfig.{level.upper()}

# Output Directory Settings
OUTPUT_DIR = r"{output_dir}"
REWARD_PLOTS_DIR = r"{os.path.join(output_dir, 'reward_plots')}"
TRAJECTORIES_DIR = r"{os.path.join(output_dir, 'trajectories')}"

# 智能体参数
agent_capacity = [10, 10, 10, 20, 20]
agent_speed = [40, 40, 40, 8, 8]
agent_energy_max = [80000, 80000, 80000, 120000, 120000]
fast_charge_time = 3 * 60
slow_charge_time = 8 * 60
"""
        
        with open(config_filename, "w", encoding="utf-8") as f:
            f.write(config_content)
    
    def _run_experiment(self, config_file, process_id, level, seed):
        """运行单个实验"""
        try:
            # 设置环境变量，指定使用的配置文件
            env = os.environ.copy()
            env['PYTHONPATH'] = os.getcwd()
            
            # 创建运行脚本
            run_script = f"""
import sys
import importlib.util
import os

# 动态加载指定的配置文件
config_path = r"{config_file}"
spec = importlib.util.spec_from_file_location("config", config_path)
config = importlib.util.module_from_spec(spec)
sys.modules["config"] = config
spec.loader.exec_module(config)

# 运行主程序
from main import main
main()
"""
            
            script_file = os.path.join(self.temp_dir, f"run_{level}_{process_id}.py")
            with open(script_file, "w", encoding="utf-8") as f:
                f.write(run_script)
            
            # 运行实验
            result = subprocess.run(
                ["python", script_file],
                capture_output=True,
                text=True,
                env=env
            )
            
            # 清理运行脚本
            if os.path.exists(script_file):
                os.remove(script_file)
            
            return result.returncode == 0
            

        except Exception as e:
            debug_print(f"   💥 进程 {process_id}: 运行异常: {e}")
            return False
    
    def run_parallel_experiments(self, max_processes=6):
        """并行运行所有实验等级"""
        debug_print("🚀 开始并行实验")
        debug_print("=" * 80)
        debug_print(f"📊 实验配置:")
        debug_print(f"   实验等级: {len(self.experiment_levels)}")
        debug_print(f"   种子数量: {len(self.seeds)}")
        debug_print(f"   并行进程: {min(max_processes, len(self.experiment_levels))}")
        debug_print(f"   每实验回合数: {self.episodes_per_experiment}")
        debug_print(f"   结果目录: {self.results_dir}")
        debug_print("=" * 80)
        
        self.start_time = datetime.now()
        
        # 检查系统资源
        self._check_system_resources()
        
        # 创建进程池并运行
        actual_processes = min(max_processes, len(self.experiment_levels))
        
        with mp.Pool(processes=actual_processes) as pool:
            debug_print(f"🔄 启动 {actual_processes} 个并行进程...")
            
            # 提交所有任务
            results = pool.map(self.run_single_level_all_seeds, self.experiment_levels)
        
        # 保存结果
        self._save_parallel_results(results)
        
        # 清理临时文件
        self._cleanup_temp_directory()
        
        # 输出总结
        self._print_summary(results)
        
        return results
    
    def _check_system_resources(self):
        """检查系统资源"""
        # CPU核心数
        cpu_count = mp.cpu_count()
        debug_print(f"💻 系统资源:")
        debug_print(f"   CPU核心数: {cpu_count}")
        
        # 内存使用情况
        memory = psutil.virtual_memory()
        debug_print(f"   内存使用: {memory.percent:.1f}% ({memory.available // (1024**3):.1f}GB 可用)")
        
        # GPU信息（如果可用）
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                debug_print(f"   GPU数量: {gpu_count}")
                for i in range(gpu_count):
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory
                    debug_print(f"   GPU {i}: {gpu_memory // (1024**3):.1f}GB")
        except ImportError:
            debug_print("   GPU: 未安装PyTorch，无法检测")
    
    def _save_parallel_results(self, results):
        """保存并行实验结果"""
        summary = {
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'total_duration': (datetime.now() - self.start_time).total_seconds(),
            'experiment_levels': self.experiment_levels,
            'seeds': self.seeds,
            'episodes_per_experiment': self.episodes_per_experiment,
            'results': results
        }
        
        results_file = os.path.join(self.results_dir, "parallel_experiment_summary.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        debug_print(f"📝 并行实验结果已保存: {results_file}")
    
    def _cleanup_temp_directory(self):
        """清理临时文件目录"""
        try:
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                debug_print(f"🧹 已清理临时文件目录: {self.temp_dir}")
        except Exception as e:
            debug_print(f"⚠️ 清理临时目录失败: {e}")
    
    def _print_summary(self, results):
        """打印实验总结"""
        total_time = datetime.now() - self.start_time
        total_experiments = len(self.experiment_levels) * len(self.seeds)
        total_success = sum(r['total_success'] for r in results)
        
        debug_print("\n" + "🎉" * 80)
        debug_print("并行实验完成总结")
        debug_print("🎉" * 80)
        debug_print(f"✅ 成功完成: {total_success}/{total_experiments}")
        debug_print(f"⏱️ 总耗时: {total_time}")
        debug_print(f"📁 结果目录: {self.results_dir}")
        
        debug_print(f"\n📊 各等级结果:")
        for result in results:
            level = result['level']
            success = result['total_success']
            total = len(self.seeds)
            duration = timedelta(seconds=result['duration'])
            debug_print(f"   {level.upper()}: {success}/{total} ({duration})")
        
        debug_print(f"\n📊 下一步: 使用 python multi_seed_analyzer.py 分析结果")
        debug_print("🎉" * 80)

def main():
    parser = argparse.ArgumentParser(description="并行运行6个实验等级")
    parser.add_argument("--seeds", type=str, default="42,123,2024,888,1337",
                       help="随机种子列表，用逗号分隔")
    parser.add_argument("--episodes", type=int, default=3000,
                       help="每个实验的训练回合数")
    parser.add_argument("--processes", type=int, default=6,
                       help="最大并行进程数")
    parser.add_argument("--output", type=str, default="parallel_experiment_results",
                       help="结果输出目录")
    
    args = parser.parse_args()
    
    # 解析种子列表
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    
    runner = ParallelExperimentRunner(
        seeds=seeds,
        episodes_per_experiment=args.episodes,
        results_dir=args.output
    )
    
    runner.run_parallel_experiments(max_processes=args.processes)

if __name__ == "__main__":
    main()

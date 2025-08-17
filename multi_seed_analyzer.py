"""
增强版多种子实验结果分析器

这个工具专门用于分析多种子批量实验的结果，提供统计可靠的性能对比和深度改进建议。

功能特性：
- 加载多种子实验结果（支持experiment_results和parallel_experiment_results）
- 计算平均值和标准差
- 生成带置信区间的可视化图表
- 提供科学的统计分析报告
- 支持显著性检验
- 学习曲线深度分析（收敛速度、稳定性、震荡程度）
- 奖励组件贡献分析
- 超参数敏感性分析
- 生成具体的代码改进建议

使用方法：
python multi_seed_analyzer.py --results_dir parallel_experiment_results
python multi_seed_analyzer.py --results_dir experiment_results
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import glob
from pathlib import Path
from datetime import datetime
from scipy import stats
import seaborn as sns

from reward_config import RewardExperimentConfig, EXPERIMENT_CONFIGS

# 独立的调试打印函数
def debug_print(*args, **kwargs):
    """独立的打印函数，不依赖config"""
    print(*args, **kwargs)

# 设置中文字体 - 支持Windows/Mac/Linux多平台
import platform
import matplotlib.font_manager as fm

def setup_chinese_font():
    """设置中文字体"""
    system = platform.system()
    
    if system == "Windows":
        # Windows系统优先字体
        chinese_fonts = ['Microsoft YaHei', 'SimHei', 'KaiTi', 'FangSong']
    elif system == "Darwin":  # macOS
        chinese_fonts = ['PingFang SC', 'Hiragino Sans GB', 'STSong']
    else:  # Linux
        chinese_fonts = ['WenQuanYi Micro Hei', 'DejaVu Sans']
    
    # 查找系统中可用的中文字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    for font in chinese_fonts:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font]
            debug_print(f"🔤 使用字体: {font}")
            break
    else:
        # 如果没有中文字体，使用英文标题
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
        debug_print("⚠️ 未找到中文字体，将使用英文标题")
        return False
    
    plt.rcParams['axes.unicode_minus'] = False
    return True

# 设置字体
USE_CHINESE = setup_chinese_font()

class MultiSeedAnalyzer:
    """增强版多种子实验结果分析器"""
    
    def __init__(self, results_dir="parallel_experiment_results"):
        """
        初始化分析器
        
        Args:
            results_dir: 实验结果目录（支持experiment_results或parallel_experiment_results）
        """
        self.results_dir = results_dir
        self.experiment_data = {}  # {level: {seed: data}}
        self.summary_stats = {}  # {level: {metric: {mean, std, ci}}}
        self.learning_analysis = {}  # {level: learning_curve_metrics}
        self.improvement_suggestions = {}  # {level: suggestions}
        
    def load_all_results(self):
        """加载所有多种子实验结果"""
        debug_print("📊 加载多种子实验结果...")
        
        if not os.path.exists(self.results_dir):
            debug_print(f"❌ 结果目录不存在: {self.results_dir}")
            return
        
        debug_print(f"   📁 结果目录: {self.results_dir}")
        
        # 检测结果目录类型
        if self._is_experiment_results_dir():
            total_loaded = self._load_experiment_results()
        elif self._is_parallel_experiment_results_dir():
            total_loaded = self._load_parallel_experiment_results()
        else:
            debug_print(f"❌ 无法识别的结果目录结构: {self.results_dir}")
            return
        
        debug_print(f"📊 总计加载 {total_loaded} 个实验结果")
        
        # 显示数据摘要
        self._print_data_summary()
    
    def _is_experiment_results_dir(self):
        """检测是否为experiment_results目录结构"""
        # 查找.json和.npy文件
        json_files = glob.glob(os.path.join(self.results_dir, "*.json"))
        npy_files = glob.glob(os.path.join(self.results_dir, "*_rewards.npy"))
        return len(json_files) > 0 and len(npy_files) > 0
    
    def _is_parallel_experiment_results_dir(self):
        """检测是否为parallel_experiment_results目录结构"""
        # 查找子目录中的rewards_log.npy文件
        pattern = os.path.join(self.results_dir, "*/*/rewards_log.npy")
        files = glob.glob(pattern)
        return len(files) > 0
    
    def _load_experiment_results(self):
        """加载experiment_results目录中的结果"""
        debug_print("   📁 检测到experiment_results目录结构")
        total_loaded = 0
        
        # 获取所有json文件
        json_files = glob.glob(os.path.join(self.results_dir, "*.json"))
        
        for json_file in json_files:
            try:
                # 解析文件名获取实验等级
                filename = os.path.basename(json_file)
                parts = filename.replace('.json', '').split('_')
                level = parts[0]
                timestamp = '_'.join(parts[1:])
                
                # 加载JSON数据
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 使用时间戳作为伪种子（单次实验）
                seed = hash(timestamp) % 10000
                
                if level not in self.experiment_data:
                    self.experiment_data[level] = {}
                
                # 提取奖励数据
                episode_rewards = np.array(data['episode_rewards'])
                
                # 提取协作分析数据
                collaboration_analytics = data.get('collaboration_analytics', {})
                
                self.experiment_data[level][seed] = {
                    'rewards': episode_rewards,
                    'analytics': collaboration_analytics,
                    'file_path': json_file,
                    'timestamp': timestamp,
                    'config': data.get('reward_config', {}),
                    'final_metrics': data.get('final_metrics', {})
                }
                
                total_loaded += 1
                debug_print(f"     ✅ {level} ({timestamp}): 已加载 ({episode_rewards.shape})")
                
            except Exception as e:
                debug_print(f"     ❌ 加载失败 {json_file}: {e}")
        
        return total_loaded
    
    def _load_parallel_experiment_results(self):
        """加载parallel_experiment_results目录中的结果"""
        debug_print("   📁 检测到parallel_experiment_results目录结构")
        total_loaded = 0
        
        # 遍历所有实验等级目录
        for level_dir in os.listdir(self.results_dir):
            level_path = os.path.join(self.results_dir, level_dir)
            
            if not os.path.isdir(level_path):
                continue
                
            if level_dir not in self.experiment_data:
                self.experiment_data[level_dir] = {}
            
            debug_print(f"   📁 处理等级: {level_dir}")
            
            # 查找所有奖励文件（在子目录中）
            search_pattern = os.path.join(level_path, "*/rewards_log.npy")
            debug_print(f"     🔍 搜索模式: {search_pattern}")
            reward_files = glob.glob(search_pattern)
            debug_print(f"     📄 找到 {len(reward_files)} 个奖励文件")
            
            for reward_file in reward_files:
                try:
                    # 从目录名提取种子信息
                    dir_name = os.path.basename(os.path.dirname(reward_file))
                    # 格式: level_seedXXX_procXXX_timestamp
                    parts = dir_name.split('_')
                    seed_part = [p for p in parts if p.startswith('seed')]
                    
                    if seed_part:
                        seed = int(seed_part[0].replace('seed', ''))
                        
                        # 加载奖励数据
                        try:
                            rewards = np.load(reward_file, allow_pickle=True)
                        except Exception as e:
                            debug_print(f"     ❌ 奖励数据加载失败: {e}")
                            continue
                        
                        # 查找对应的协作分析文件
                        analytics_file = reward_file.replace('rewards_log.npy', 'collaboration_analytics.npy')
                        analytics = None
                        if os.path.exists(analytics_file):
                            try:
                                analytics = np.load(analytics_file, allow_pickle=True).item()
                            except Exception as e:
                                debug_print(f"     ⚠️ 协作数据加载失败: {e}")
                        
                        self.experiment_data[level_dir][seed] = {
                            'rewards': rewards,
                            'analytics': analytics,
                            'file_path': reward_file
                        }
                        
                        total_loaded += 1
                        debug_print(f"     ✅ 种子 {seed}: 已加载 ({rewards.shape})")
                        
                except Exception as e:
                    debug_print(f"     ❌ 加载失败 {reward_file}: {e}")
        
        return total_loaded
        
    def _print_data_summary(self):
        """打印数据摘要"""
        debug_print("\n📋 数据摘要:")
        debug_print("-" * 60)
        
        for level, seeds_data in self.experiment_data.items():
            debug_print(f"   {level}: {len(seeds_data)} 个种子")
            if seeds_data:
                example_data = next(iter(seeds_data.values()))
                debug_print(f"     奖励维度: {example_data['rewards'].shape}")
                debug_print(f"     协作数据: {'有' if example_data['analytics'] else '无'}")
        
        debug_print("-" * 60)
    
    def calculate_statistics(self):
        """计算统计指标"""
        debug_print("\n📊 计算统计指标...")
        
        for level, seeds_data in self.experiment_data.items():
            if not seeds_data:
                continue
                
            debug_print(f"   🔍 分析等级: {level}")
            
            # 收集所有种子的数据
            all_rewards = []
            all_final_performance = []
            all_conflict_rates = []
            
            for seed, data in seeds_data.items():
                rewards = data['rewards']
                
                # 计算总奖励
                if rewards.ndim > 1:
                    total_rewards = np.sum(rewards, axis=1)
                else:
                    total_rewards = rewards
                
                all_rewards.append(total_rewards)
                
                # 最终性能（最后100轮的平均）
                final_perf = np.mean(total_rewards[-100:])
                all_final_performance.append(final_perf)
                
                # 冲突率
                if data['analytics'] and 'conflict_rate' in data['analytics']:
                    all_conflict_rates.append(data['analytics']['conflict_rate'])
            
            # 计算统计指标
            stats_dict = {}
            
            # 最终性能统计
            if all_final_performance:
                final_array = np.array(all_final_performance)
                stats_dict['final_performance'] = {
                    'mean': np.mean(final_array),
                    'std': np.std(final_array, ddof=1),  # 使用样本标准差
                    'ci_95': self._calculate_confidence_interval(final_array),
                    'n': len(final_array)
                }
            
            # 冲突率统计
            if all_conflict_rates:
                conflict_array = np.array(all_conflict_rates)
                stats_dict['conflict_rate'] = {
                    'mean': np.mean(conflict_array),
                    'std': np.std(conflict_array, ddof=1),
                    'ci_95': self._calculate_confidence_interval(conflict_array),
                    'n': len(conflict_array)
                }
            
            # 奖励曲线统计
            if all_rewards:
                # 计算每个时间点的平均值和标准差
                min_length = min(len(r) for r in all_rewards)
                truncated_rewards = [r[:min_length] for r in all_rewards]
                rewards_matrix = np.array(truncated_rewards)
                
                stats_dict['reward_curve'] = {
                    'mean': np.mean(rewards_matrix, axis=0),
                    'std': np.std(rewards_matrix, axis=0, ddof=1),
                    'ci_95_upper': np.percentile(rewards_matrix, 97.5, axis=0),
                    'ci_95_lower': np.percentile(rewards_matrix, 2.5, axis=0),
                    'n': len(all_rewards)
                }
            
            self.summary_stats[level] = stats_dict
            
            # 打印关键统计信息
            if 'final_performance' in stats_dict:
                fp = stats_dict['final_performance']
                debug_print(f"     最终性能: {fp['mean']:.1f} ± {fp['std']:.1f} (n={fp['n']})")
            
            if 'conflict_rate' in stats_dict:
                cr = stats_dict['conflict_rate']
                debug_print(f"     冲突率: {cr['mean']:.3f} ± {cr['std']:.3f} (n={cr['n']})")
    
    def analyze_learning_curves(self):
        """分析学习曲线特征"""
        debug_print("\n📈 分析学习曲线特征...")
        
        for level, seeds_data in self.experiment_data.items():
            if not seeds_data:
                continue
                
            debug_print(f"   🔍 分析等级: {level}")
            
            # 收集所有种子的奖励曲线
            all_curves = []
            for seed, data in seeds_data.items():
                rewards = data['rewards']
                if rewards.ndim > 1:
                    total_rewards = np.sum(rewards, axis=1)
                else:
                    total_rewards = rewards
                all_curves.append(total_rewards)
            
            if not all_curves:
                continue
            
            # 计算学习曲线指标
            curve_metrics = self._calculate_learning_metrics(all_curves)
            self.learning_analysis[level] = curve_metrics
            
            # 打印关键指标
            debug_print(f"     收敛速度: {curve_metrics['convergence_episode']:.0f} 轮")
            debug_print(f"     学习稳定性: {curve_metrics['stability_score']:.3f}")
            debug_print(f"     收敛后震荡: {curve_metrics['oscillation_level']:.3f}")
    
    def _calculate_learning_metrics(self, reward_curves):
        """计算学习曲线指标"""
        # 统一长度
        min_length = min(len(curve) for curve in reward_curves)
        truncated_curves = [curve[:min_length] for curve in reward_curves]
        curves_matrix = np.array(truncated_curves)
        
        # 平均曲线
        mean_curve = np.mean(curves_matrix, axis=0)
        
        # 1. 收敛速度分析 (达到最终性能90%的轮数)
        final_performance = np.mean(mean_curve[-100:])  # 最后100轮的平均性能
        threshold = final_performance * 0.9
        
        # 使用移动平均来平滑曲线
        window_size = 50
        smoothed_curve = np.convolve(mean_curve, np.ones(window_size)/window_size, mode='valid')
        
        convergence_episode = len(mean_curve)  # 默认值
        for i, value in enumerate(smoothed_curve):
            if value >= threshold:
                convergence_episode = i + window_size//2
                break
        
        # 2. 学习稳定性 (后期方差的倒数)
        late_phase = mean_curve[len(mean_curve)//2:]  # 后半段
        stability_score = 1.0 / (1.0 + np.var(late_phase))
        
        # 3. 收敛后震荡水平 (最后1/3阶段的标准差)
        final_third = curves_matrix[:, -len(mean_curve)//3:]
        oscillation_level = np.mean(np.std(final_third, axis=0))
        
        # 4. 初始学习速度 (前20%阶段的改进幅度)
        initial_phase_length = len(mean_curve) // 5
        initial_improvement = mean_curve[initial_phase_length] - mean_curve[0]
        
        # 5. 学习效率 (单位时间内的性能提升)
        learning_efficiency = (final_performance - mean_curve[0]) / len(mean_curve)
        
        return {
            'convergence_episode': convergence_episode,
            'stability_score': stability_score,
            'oscillation_level': oscillation_level,
            'initial_improvement': initial_improvement,
            'learning_efficiency': learning_efficiency,
            'final_performance': final_performance,
            'mean_curve': mean_curve
        }
    
    def generate_improvement_suggestions(self):
        """基于分析结果生成改进建议"""
        debug_print("\n💡 生成改进建议...")
        
        for level, stats in self.summary_stats.items():
            suggestions = []
            
            # 基于学习曲线分析的建议
            if level in self.learning_analysis:
                learning = self.learning_analysis[level]
                
                # 收敛速度建议
                if learning['convergence_episode'] > 2000:
                    suggestions.append({
                        'category': '收敛速度',
                        'issue': '收敛速度较慢',
                        'suggestion': '考虑提高学习率或优化奖励函数设计',
                        'code_change': 'config.LEARNING_RATE *= 1.5  # 提高学习率'
                    })
                
                # 稳定性建议
                if learning['stability_score'] < 0.7:
                    suggestions.append({
                        'category': '学习稳定性',
                        'issue': '学习过程不够稳定',
                        'suggestion': '减少学习率或增加经验回放缓冲区大小',
                        'code_change': 'config.LEARNING_RATE *= 0.8  # 降低学习率\nconfig.BUFFER_SIZE *= 2  # 增加缓冲区'
                    })
                
                # 震荡建议
                if learning['oscillation_level'] > 50:
                    suggestions.append({
                        'category': '训练震荡',
                        'issue': '训练后期震荡较大',
                        'suggestion': '实施学习率衰减或目标网络软更新',
                        'code_change': 'config.LR_DECAY = 0.995  # 学习率衰减\nconfig.TAU = 0.01  # 软更新参数'
                    })
            
            # 基于性能对比的建议
            if 'final_performance' in stats:
                fp = stats['final_performance']
                
                # 如果有基准比较
                if RewardExperimentConfig.BASIC in self.summary_stats:
                    basic_perf = self.summary_stats[RewardExperimentConfig.BASIC]['final_performance']['mean']
                    improvement = fp['mean'] - basic_perf
                    
                    if improvement < 0:
                        suggestions.append({
                            'category': '性能回退',
                            'issue': f'相比基准性能下降了 {abs(improvement):.1f}',
                            'suggestion': '检查奖励函数权重配置，可能某些奖励组件产生负面影响',
                            'code_change': f'# 检查 {level} 配置中的奖励权重\n# 考虑降低或移除表现不佳的奖励组件'
                        })
                    elif improvement < 50:
                        suggestions.append({
                            'category': '微弱改进',
                            'issue': f'相比基准仅提升 {improvement:.1f}',
                            'suggestion': '增加该等级特有奖励组件的权重',
                            'code_change': f'# 在 {level} 配置中增加关键奖励权重\n# 例如: COLLABORATION_REWARD_WEIGHT *= 1.5'
                        })
            
            # 基于冲突率的建议
            if 'conflict_rate' in stats:
                cr = stats['conflict_rate']
                if cr['mean'] > 0.3:
                    suggestions.append({
                        'category': '协作优化',
                        'issue': f'冲突率过高 ({cr["mean"]:.3f})',
                        'suggestion': '增强协作奖励机制或改进冲突检测算法',
                        'code_change': 'config.CONFLICT_PENALTY_WEIGHT *= 2  # 增加冲突惩罚\nconfig.COLLABORATION_BONUS_WEIGHT *= 1.5  # 增加协作奖励'
                    })
            
            self.improvement_suggestions[level] = suggestions
            
            if suggestions:
                debug_print(f"   🎯 {level}: 生成了 {len(suggestions)} 条改进建议")
    
    def _calculate_confidence_interval(self, data, confidence=0.95):
        """计算置信区间"""
        n = len(data)
        if n < 2:
            return (np.mean(data), np.mean(data))
        
        mean = np.mean(data)
        se = stats.sem(data)  # 标准误差
        t_value = stats.t.ppf((1 + confidence) / 2, df=n-1)
        margin_error = t_value * se
        
        return (mean - margin_error, mean + margin_error)
    
    def generate_statistical_report(self, output_file="multi_seed_report.txt"):
        """生成统计分析报告"""
        debug_print(f"\n📝 生成统计报告: {output_file}")
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("多种子增量式奖励实验统计分析报告")
        report_lines.append("=" * 80)
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"结果目录: {self.results_dir}")
        report_lines.append("")
        
        # 实验概述
        report_lines.append("📊 实验概述")
        report_lines.append("-" * 60)
        total_experiments = sum(len(seeds) for seeds in self.experiment_data.values())
        report_lines.append(f"实验等级数: {len(self.experiment_data)}")
        report_lines.append(f"总实验次数: {total_experiments}")
        report_lines.append("")
        
        # 各等级详细分析
        experiment_order = [
            RewardExperimentConfig.BASIC,
            RewardExperimentConfig.LOAD_EFFICIENCY,
            RewardExperimentConfig.ROLE_SPECIALIZATION,
            RewardExperimentConfig.COLLABORATION,
            RewardExperimentConfig.BEHAVIOR_SHAPING,
            RewardExperimentConfig.FULL
        ]
        
        performance_data = {}  # 用于增量效果分析
        
        for level in experiment_order:
            if level not in self.summary_stats:
                continue
                
            stats = self.summary_stats[level]
            config = EXPERIMENT_CONFIGS[level]
            
            report_lines.append(f"🧪 实验等级: {level.upper()}")
            report_lines.append("-" * 60)
            report_lines.append(f"描述: {config.get_experiment_description()}")
            report_lines.append(f"种子数量: {len(self.experiment_data[level])}")
            report_lines.append("")
            
            # 性能指标
            if 'final_performance' in stats:
                fp = stats['final_performance']
                ci_lower, ci_upper = fp['ci_95']
                report_lines.append("📈 最终性能分析:")
                report_lines.append(f"   平均值: {fp['mean']:.2f}")
                report_lines.append(f"   标准差: {fp['std']:.2f}")
                report_lines.append(f"   95%置信区间: [{ci_lower:.2f}, {ci_upper:.2f}]")
                report_lines.append(f"   样本数: {fp['n']}")
                
                performance_data[level] = fp
            
            # 协作指标
            if 'conflict_rate' in stats:
                cr = stats['conflict_rate']
                ci_lower, ci_upper = cr['ci_95']
                report_lines.append("")
                report_lines.append("🤝 协作分析:")
                report_lines.append(f"   冲突率均值: {cr['mean']:.4f}")
                report_lines.append(f"   冲突率标准差: {cr['std']:.4f}")
                report_lines.append(f"   95%置信区间: [{ci_lower:.4f}, {ci_upper:.4f}]")
                report_lines.append(f"   样本数: {cr['n']}")
            
            report_lines.append("")
            report_lines.append("")
        
        # 增量效果分析
        if len(performance_data) > 1:
            report_lines.append("📊 增量效果分析")
            report_lines.append("-" * 60)
            
            prev_level = None
            prev_data = None
            
            for level in experiment_order:
                if level not in performance_data:
                    continue
                    
                current_data = performance_data[level]
                
                if prev_data is not None:
                    # 计算性能提升
                    improvement = current_data['mean'] - prev_data['mean']
                    improvement_pct = (improvement / abs(prev_data['mean'])) * 100
                    
                    # 简单的统计显著性检验（基于置信区间是否重叠）
                    prev_ci_upper = prev_data['ci_95'][1]
                    current_ci_lower = current_data['ci_95'][0]
                    is_significant = current_ci_lower > prev_ci_upper
                    
                    significance_text = "显著" if is_significant else "不显著"
                    
                    report_lines.append(f"{prev_level.upper()} → {level.upper()}:")
                    report_lines.append(f"   性能变化: {improvement:+.2f} ({improvement_pct:+.1f}%)")
                    report_lines.append(f"   统计显著性: {significance_text}")
                    report_lines.append("")
                
                prev_level = level
                prev_data = current_data
        
        # 学习曲线分析
        if self.learning_analysis:
            report_lines.append("📈 学习曲线分析")
            report_lines.append("-" * 60)
            
            for level in experiment_order:
                if level not in self.learning_analysis:
                    continue
                    
                learning = self.learning_analysis[level]
                report_lines.append(f"🔍 {level.upper()} 学习特征:")
                report_lines.append(f"   收敛速度: {learning['convergence_episode']:.0f} 轮")
                report_lines.append(f"   学习稳定性: {learning['stability_score']:.3f}")
                report_lines.append(f"   震荡水平: {learning['oscillation_level']:.3f}")
                report_lines.append(f"   初始改进: {learning['initial_improvement']:.2f}")
                report_lines.append(f"   学习效率: {learning['learning_efficiency']:.4f}")
                report_lines.append("")
        
        # 具体改进建议
        if self.improvement_suggestions:
            report_lines.append("🛠️ 具体改进建议")
            report_lines.append("-" * 60)
            
            for level in experiment_order:
                if level not in self.improvement_suggestions:
                    continue
                    
                suggestions = self.improvement_suggestions[level]
                if not suggestions:
                    continue
                    
                report_lines.append(f"🎯 {level.upper()} 改进建议:")
                
                for i, suggestion in enumerate(suggestions, 1):
                    report_lines.append(f"   {i}. {suggestion['category']}: {suggestion['issue']}")
                    report_lines.append(f"      建议: {suggestion['suggestion']}")
                    report_lines.append(f"      代码修改: {suggestion['code_change']}")
                    report_lines.append("")
        
        # 结论和建议
        report_lines.append("💡 结论和建议")
        report_lines.append("-" * 60)
        
        if performance_data:
            best_level = max(performance_data.keys(), key=lambda k: performance_data[k]['mean'])
            best_perf = performance_data[best_level]
            
            report_lines.append(f"🏆 最佳表现等级: {best_level.upper()}")
            report_lines.append(f"   最终性能: {best_perf['mean']:.2f} ± {best_perf['std']:.2f}")
            
            if RewardExperimentConfig.BASIC in performance_data:
                basic_perf = performance_data[RewardExperimentConfig.BASIC]['mean']
                total_improvement = best_perf['mean'] - basic_perf
                total_improvement_pct = (total_improvement / abs(basic_perf)) * 100
                
                report_lines.append("")
                report_lines.append(f"🚀 总体提升效果:")
                report_lines.append(f"   相比BASIC等级提升: {total_improvement:+.2f} ({total_improvement_pct:+.1f}%)")
        
        # 学习曲线洞察
        if self.learning_analysis:
            report_lines.append("")
            report_lines.append("📊 学习曲线洞察:")
            
            # 找到收敛最快的等级
            fastest_convergence = min(self.learning_analysis.items(), 
                                    key=lambda x: x[1]['convergence_episode'])
            report_lines.append(f"   最快收敛: {fastest_convergence[0]} ({fastest_convergence[1]['convergence_episode']:.0f} 轮)")
            
            # 找到最稳定的等级
            most_stable = max(self.learning_analysis.items(), 
                            key=lambda x: x[1]['stability_score'])
            report_lines.append(f"   最稳定学习: {most_stable[0]} (稳定性: {most_stable[1]['stability_score']:.3f})")
            
            # 找到学习效率最高的等级
            most_efficient = max(self.learning_analysis.items(), 
                               key=lambda x: x[1]['learning_efficiency'])
            report_lines.append(f"   最高效率: {most_efficient[0]} (效率: {most_efficient[1]['learning_efficiency']:.4f})")
        
        report_lines.append("")
        report_lines.append("📋 综合建议:")
        report_lines.append("   1. 对于显著改进的模块，可进一步调优参数")
        report_lines.append("   2. 对于性能下降的模块，需要检查实现或调整权重")
        report_lines.append("   3. 考虑组合最有效的奖励模块构建最优策略")
        report_lines.append("   4. 参考收敛最快的配置来优化学习率设置")
        report_lines.append("   5. 采用最稳定配置的参数来减少训练波动")
        
        # 保存报告
        report_path = os.path.join(self.results_dir, output_file)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        debug_print(f"📊 统计报告已保存: {report_path}")
        
        # 也打印关键结论
        debug_print("\n💡 关键发现:")
        if performance_data:
            for level, data in performance_data.items():
                ci_lower, ci_upper = data['ci_95']
                debug_print(f"   {level}: {data['mean']:.1f} ± {data['std']:.1f} [{ci_lower:.1f}, {ci_upper:.1f}]")
    
    def plot_multi_seed_comparison(self, output_file="multi_seed_comparison.png"):
        """绘制多种子对比图表"""
        debug_print(f"\n📊 生成多种子对比图表: {output_file}")
        
        # 设置图表样式
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 根据字体支持选择标题
        if USE_CHINESE:
            main_title = '多种子增量式奖励实验对比分析'
            curve_title = '训练奖励曲线对比 (95%置信区间)'
            performance_title = '最终性能分布 (最后100轮平均)'
            improvement_title = '相对基准的性能提升'
            conflict_title = '冲突率对比'
            xlabel_episodes = '训练回合'
            ylabel_reward = '总奖励'
            ylabel_avg_reward = '平均总奖励'
            ylabel_improvement = '性能提升'
            ylabel_conflict = '平均冲突率'
        else:
            main_title = 'Multi-Seed Incremental Reward Experiment Comparison'
            curve_title = 'Training Reward Curves (95% CI)'
            performance_title = 'Final Performance Distribution (Last 100 Episodes Avg)'
            improvement_title = 'Performance Improvement vs Baseline'
            conflict_title = 'Conflict Rate Comparison'
            xlabel_episodes = 'Episodes'
            ylabel_reward = 'Total Reward'
            ylabel_avg_reward = 'Average Total Reward'
            ylabel_improvement = 'Performance Improvement'
            ylabel_conflict = 'Average Conflict Rate'
        
        fig.suptitle(main_title, fontsize=16, fontweight='bold')
        
        # 1. 带置信区间的奖励曲线对比
        ax1 = axes[0, 0]
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.summary_stats)))
        
        for i, (level, stats) in enumerate(self.summary_stats.items()):
            if 'reward_curve' in stats:
                curve_stats = stats['reward_curve']
                x = np.arange(len(curve_stats['mean']))
                
                # 绘制平均线
                ax1.plot(x, curve_stats['mean'], color=colors[i], 
                        label=f"{level} (n={curve_stats['n']})", linewidth=2)
                
                # 绘制置信区间
                ax1.fill_between(x, curve_stats['ci_95_lower'], curve_stats['ci_95_upper'],
                               color=colors[i], alpha=0.2)
        
        ax1.set_title(curve_title)
        ax1.set_xlabel(xlabel_episodes)
        ax1.set_ylabel(ylabel_reward)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 最终性能箱线图
        ax2 = axes[0, 1]
        final_performances = []
        level_names = []
        
        for level, seeds_data in self.experiment_data.items():
            level_performances = []
            for seed, data in seeds_data.items():
                rewards = data['rewards']
                if rewards.ndim > 1:
                    total_rewards = np.sum(rewards, axis=1)
                else:
                    total_rewards = rewards
                final_perf = np.mean(total_rewards[-100:])
                level_performances.append(final_perf)
            
            if level_performances:
                final_performances.append(level_performances)
                level_names.append(level)
        
        if final_performances:
            box_plot = ax2.boxplot(final_performances, labels=level_names, patch_artist=True)
            
            # 设置颜色
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax2.set_title(performance_title)
            ax2.set_ylabel(ylabel_avg_reward)
            ax2.tick_params(axis='x', rotation=45)
        
        # 3. 性能提升对比
        ax3 = axes[1, 0]
        if len(self.summary_stats) > 1:
            levels = list(self.summary_stats.keys())
            improvements = []
            improvement_errors = []
            
            # 以第一个等级为基准
            baseline_stats = list(self.summary_stats.values())[0]
            if 'final_performance' in baseline_stats:
                baseline_mean = baseline_stats['final_performance']['mean']
                
                for stats in self.summary_stats.values():
                    if 'final_performance' in stats:
                        current_mean = stats['final_performance']['mean']
                        current_std = stats['final_performance']['std']
                        improvement = current_mean - baseline_mean
                        improvements.append(improvement)
                        improvement_errors.append(current_std)
                
                bars = ax3.bar(range(len(levels)), improvements, yerr=improvement_errors,
                             capsize=5, color=colors[:len(levels)], alpha=0.7)
                ax3.set_title(improvement_title)
                ax3.set_ylabel(ylabel_improvement)
                ax3.set_xticks(range(len(levels)))
                ax3.set_xticklabels(levels, rotation=45)
                ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                ax3.grid(True, alpha=0.3)
        
        # 4. 冲突率对比
        ax4 = axes[1, 1]
        conflict_rates = []
        conflict_errors = []
        conflict_levels = []
        
        for level, stats in self.summary_stats.items():
            if 'conflict_rate' in stats:
                cr_stats = stats['conflict_rate']
                conflict_rates.append(cr_stats['mean'])
                conflict_errors.append(cr_stats['std'])
                conflict_levels.append(level)
        
        if conflict_rates:
            bars = ax4.bar(range(len(conflict_levels)), conflict_rates, 
                          yerr=conflict_errors, capsize=5, 
                          color=colors[:len(conflict_levels)], alpha=0.7)
            ax4.set_title(conflict_title)
            ax4.set_ylabel(ylabel_conflict)
            ax4.set_xticks(range(len(conflict_levels)))
            ax4.set_xticklabels(conflict_levels, rotation=45)
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        output_path = os.path.join(self.results_dir, output_file)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        debug_print(f"📊 多种子对比图表已保存: {output_path}")
    
    def plot_enhanced_analysis(self, output_file="enhanced_analysis.png"):
        """绘制增强分析图表"""
        debug_print(f"\n📊 生成增强分析图表: {output_file}")
        
        if not self.learning_analysis:
            debug_print("❌ 没有学习曲线分析数据")
            return
        
        # 设置图表样式
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 根据字体支持选择标题
        if USE_CHINESE:
            main_title = '增强版学习曲线分析'
            convergence_title = '收敛速度对比'
            stability_title = '学习稳定性对比'
            oscillation_title = '震荡水平对比'
            efficiency_title = '学习效率对比'
            improvement_title = '初始改进幅度对比'
            curve_comparison_title = '标准化学习曲线对比'
        else:
            main_title = 'Enhanced Learning Curve Analysis'
            convergence_title = 'Convergence Speed Comparison'
            stability_title = 'Learning Stability Comparison'
            oscillation_title = 'Oscillation Level Comparison'
            efficiency_title = 'Learning Efficiency Comparison'
            improvement_title = 'Initial Improvement Comparison'
            curve_comparison_title = 'Normalized Learning Curves'
        
        fig.suptitle(main_title, fontsize=16, fontweight='bold')
        
        levels = list(self.learning_analysis.keys())
        colors = plt.cm.Set1(np.linspace(0, 1, len(levels)))
        
        # 1. 收敛速度对比
        ax1 = axes[0, 0]
        convergence_speeds = [self.learning_analysis[level]['convergence_episode'] for level in levels]
        bars1 = ax1.bar(range(len(levels)), convergence_speeds, color=colors, alpha=0.7)
        ax1.set_title(convergence_title)
        ax1.set_ylabel('收敛轮数' if USE_CHINESE else 'Convergence Episodes')
        ax1.set_xticks(range(len(levels)))
        ax1.set_xticklabels(levels, rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 在柱子上显示数值
        for bar, value in zip(bars1, convergence_speeds):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(convergence_speeds)*0.01,
                    f'{value:.0f}', ha='center', va='bottom')
        
        # 2. 学习稳定性对比
        ax2 = axes[0, 1]
        stability_scores = [self.learning_analysis[level]['stability_score'] for level in levels]
        bars2 = ax2.bar(range(len(levels)), stability_scores, color=colors, alpha=0.7)
        ax2.set_title(stability_title)
        ax2.set_ylabel('稳定性分数' if USE_CHINESE else 'Stability Score')
        ax2.set_xticks(range(len(levels)))
        ax2.set_xticklabels(levels, rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 3. 震荡水平对比
        ax3 = axes[0, 2]
        oscillation_levels = [self.learning_analysis[level]['oscillation_level'] for level in levels]
        bars3 = ax3.bar(range(len(levels)), oscillation_levels, color=colors, alpha=0.7)
        ax3.set_title(oscillation_title)
        ax3.set_ylabel('震荡水平' if USE_CHINESE else 'Oscillation Level')
        ax3.set_xticks(range(len(levels)))
        ax3.set_xticklabels(levels, rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # 4. 学习效率对比
        ax4 = axes[1, 0]
        learning_efficiencies = [self.learning_analysis[level]['learning_efficiency'] for level in levels]
        bars4 = ax4.bar(range(len(levels)), learning_efficiencies, color=colors, alpha=0.7)
        ax4.set_title(efficiency_title)
        ax4.set_ylabel('学习效率' if USE_CHINESE else 'Learning Efficiency')
        ax4.set_xticks(range(len(levels)))
        ax4.set_xticklabels(levels, rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # 5. 初始改进幅度对比
        ax5 = axes[1, 1]
        initial_improvements = [self.learning_analysis[level]['initial_improvement'] for level in levels]
        bars5 = ax5.bar(range(len(levels)), initial_improvements, color=colors, alpha=0.7)
        ax5.set_title(improvement_title)
        ax5.set_ylabel('初始改进' if USE_CHINESE else 'Initial Improvement')
        ax5.set_xticks(range(len(levels)))
        ax5.set_xticklabels(levels, rotation=45)
        ax5.grid(True, alpha=0.3)
        
        # 6. 标准化学习曲线对比
        ax6 = axes[1, 2]
        for i, level in enumerate(levels):
            curve = self.learning_analysis[level]['mean_curve']
            # 标准化到0-1范围
            normalized_curve = (curve - curve.min()) / (curve.max() - curve.min())
            x = np.arange(len(normalized_curve))
            ax6.plot(x, normalized_curve, color=colors[i], label=level, linewidth=2)
        
        ax6.set_title(curve_comparison_title)
        ax6.set_xlabel('训练轮数' if USE_CHINESE else 'Training Episodes')
        ax6.set_ylabel('标准化奖励' if USE_CHINESE else 'Normalized Reward')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        output_path = os.path.join(self.results_dir, output_file)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        debug_print(f"📊 增强分析图表已保存: {output_path}")
    
    def save_improvement_suggestions(self, output_file="improvement_suggestions.json"):
        """保存改进建议为JSON文件"""
        debug_print(f"\n💾 保存改进建议: {output_file}")
        
        suggestions_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'results_directory': self.results_dir,
            'total_levels_analyzed': len(self.improvement_suggestions),
            'suggestions_by_level': self.improvement_suggestions
        }
        
        # 添加汇总统计
        total_suggestions = sum(len(suggestions) for suggestions in self.improvement_suggestions.values())
        categories = {}
        for suggestions in self.improvement_suggestions.values():
            for suggestion in suggestions:
                category = suggestion['category']
                categories[category] = categories.get(category, 0) + 1
        
        suggestions_data['summary'] = {
            'total_suggestions': total_suggestions,
            'suggestions_by_category': categories
        }
        
        # 保存文件
        output_path = os.path.join(self.results_dir, output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(suggestions_data, f, ensure_ascii=False, indent=2)
        
        debug_print(f"💾 改进建议已保存: {output_path}")
        debug_print(f"   总计 {total_suggestions} 条建议，涵盖 {len(categories)} 个类别")
    
    def run_complete_analysis(self):
        """运行完整的多种子分析"""
        debug_print("🚀 开始完整的多种子实验分析")
        debug_print("=" * 80)
        
        # 加载数据
        self.load_all_results()
        
        if not self.experiment_data:
            debug_print("❌ 没有找到实验数据，请确保先运行批量实验")
            return
        
        # 计算统计指标
        self.calculate_statistics()
        
        # 分析学习曲线
        self.analyze_learning_curves()
        
        # 生成改进建议
        self.generate_improvement_suggestions()
        
        # 生成报告
        self.generate_statistical_report()
        
        # 生成图表
        self.plot_multi_seed_comparison()
        
        # 生成增强版可视化图表
        self.plot_enhanced_analysis()
        
        # 保存改进建议
        self.save_improvement_suggestions()
        
        debug_print("\n🎉 增强版多种子分析完成！")
        debug_print(f"📁 查看结果目录: {self.results_dir}")
        debug_print("📊 主要输出文件:")
        debug_print("   - multi_seed_report.txt: 详细统计报告（包含改进建议）")
        debug_print("   - multi_seed_comparison.png: 基础对比图表")
        debug_print("   - enhanced_analysis.png: 增强分析图表（学习曲线特征）")
        debug_print("   - improvement_suggestions.json: 结构化改进建议")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="多种子实验结果分析")
    parser.add_argument("--results_dir", default="multi_seed_results",
                       help="实验结果目录路径")
    
    args = parser.parse_args()
    
    analyzer = MultiSeedAnalyzer(args.results_dir)
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()

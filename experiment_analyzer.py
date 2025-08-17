"""
增量式奖励实验分析工具

这个工具帮助您分析和比较不同实验等级的训练结果，
识别每个奖励模块对协作策略的具体影响。

使用方法：
1. 运行不同等级的实验
2. 使用此工具加载和分析结果
3. 生成对比报告和可视化图表
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
from datetime import datetime
from utils import debug_print
from reward_config import RewardExperimentConfig, EXPERIMENT_CONFIGS

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class ExperimentAnalyzer:
    """增量式奖励实验分析器"""
    
    def __init__(self, results_dir="experiment_results"):
        """
        初始化分析器
        
        Args:
            results_dir: 实验结果存储目录
        """
        self.results_dir = results_dir
        self.experiment_data = {}
        os.makedirs(results_dir, exist_ok=True)
    
    def save_experiment_result(self, experiment_level, episode_rewards, collaboration_analytics, 
                             training_config=None, final_metrics=None):
        """
        保存实验结果
        
        Args:
            experiment_level: 实验等级
            episode_rewards: 每轮奖励记录
            collaboration_analytics: 协作分析数据
            training_config: 训练配置信息
            final_metrics: 最终性能指标
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        result_data = {
            'experiment_level': experiment_level,
            'timestamp': timestamp,
            'episode_rewards': np.array(episode_rewards).tolist(),
            'collaboration_analytics': collaboration_analytics,
            'training_config': training_config or {},
            'final_metrics': final_metrics or {},
            'reward_config': EXPERIMENT_CONFIGS[experiment_level].get_reward_constants()
        }
        
        # 保存到文件
        filename = f"{experiment_level}_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        
        debug_print(f"✅ 实验结果已保存: {filepath}")
        
        # 同时保存numpy格式的奖励数据（兼容现有分析工具）
        rewards_array = np.array(episode_rewards)
        np_filename = f"{experiment_level}_{timestamp}_rewards.npy"
        np_filepath = os.path.join(self.results_dir, np_filename)
        np.save(np_filepath, rewards_array)
        
        return filepath
    
    def load_experiment_results(self, experiment_levels=None):
        """
        加载实验结果
        
        Args:
            experiment_levels: 要加载的实验等级列表，None表示加载所有
        """
        if experiment_levels is None:
            experiment_levels = [RewardExperimentConfig.BASIC, RewardExperimentConfig.LOAD_EFFICIENCY,
                               RewardExperimentConfig.ROLE_SPECIALIZATION, RewardExperimentConfig.COLLABORATION,
                               RewardExperimentConfig.BEHAVIOR_SHAPING, RewardExperimentConfig.FULL]
        
        self.experiment_data = {}
        
        for level in experiment_levels:
            # 查找该等级的最新结果文件
            level_files = [f for f in os.listdir(self.results_dir) 
                          if f.startswith(level) and f.endswith('.json')]
            
            if level_files:
                # 选择最新的文件
                latest_file = sorted(level_files)[-1]
                filepath = os.path.join(self.results_dir, latest_file)
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    self.experiment_data[level] = data
                    debug_print(f"✅ 已加载 {level} 实验结果: {latest_file}")
                except Exception as e:
                    debug_print(f"❌ 加载 {level} 实验结果失败: {e}")
            else:
                debug_print(f"⚠️ 未找到 {level} 实验结果")
        
        return len(self.experiment_data)
    
    def generate_comparison_report(self, output_file="experiment_comparison_report.txt"):
        """生成实验对比报告"""
        if not self.experiment_data:
            debug_print("❌ 没有加载任何实验数据，请先调用 load_experiment_results()")
            return
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("增量式奖励实验对比报告")
        report_lines.append("=" * 80)
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"实验数量: {len(self.experiment_data)}")
        report_lines.append("")
        
        # 为每个实验等级生成分析
        for level in sorted(self.experiment_data.keys()):
            data = self.experiment_data[level]
            config = EXPERIMENT_CONFIGS[level]
            
            report_lines.append(f"📊 实验等级: {level.upper()}")
            report_lines.append("-" * 60)
            report_lines.append(f"描述: {config.get_experiment_description()}")
            report_lines.append(f"预期行为: {', '.join(config.get_expected_behaviors())}")
            report_lines.append("")
            
            # 奖励分析
            episode_rewards = np.array(data['episode_rewards'])
            if episode_rewards.ndim > 1:
                total_rewards_per_episode = np.sum(episode_rewards, axis=1)
                avg_reward_per_agent = np.mean(episode_rewards, axis=1)
            else:
                total_rewards_per_episode = episode_rewards
                avg_reward_per_agent = episode_rewards
            
            report_lines.append("🎯 性能指标:")
            report_lines.append(f"   最终平均奖励: {np.mean(total_rewards_per_episode[-100:]):.2f}")
            report_lines.append(f"   奖励标准差: {np.std(total_rewards_per_episode[-100:]):.2f}")
            report_lines.append(f"   训练回合数: {len(total_rewards_per_episode)}")
            
            # 协作分析
            if 'collaboration_analytics' in data:
                analytics = data['collaboration_analytics']
                if analytics:
                    report_lines.append("")
                    report_lines.append("🤝 协作指标:")
                    report_lines.append(f"   冲突率: {analytics.get('conflict_rate', 0):.3f}")
                    report_lines.append(f"   总冲突次数: {analytics.get('total_conflicts', 0)}")
                    report_lines.append(f"   总决策次数: {analytics.get('total_decisions', 0)}")
            
            report_lines.append("")
            report_lines.append("")
        
        # 跨实验对比
        if len(self.experiment_data) > 1:
            report_lines.append("📈 跨实验对比分析")
            report_lines.append("-" * 60)
            
            # 性能提升分析
            performance_data = {}
            for level, data in self.experiment_data.items():
                episode_rewards = np.array(data['episode_rewards'])
                if episode_rewards.ndim > 1:
                    final_performance = np.mean(np.sum(episode_rewards[-100:], axis=1))
                else:
                    final_performance = np.mean(episode_rewards[-100:])
                performance_data[level] = final_performance
            
            # 按性能排序
            sorted_performance = sorted(performance_data.items(), key=lambda x: x[1], reverse=True)
            
            report_lines.append("性能排序 (最终100轮平均总奖励):")
            for i, (level, perf) in enumerate(sorted_performance):
                report_lines.append(f"   {i+1}. {level.upper()}: {perf:.2f}")
            
            # 增量提升分析
            report_lines.append("")
            report_lines.append("增量效果分析:")
            experiment_order = [RewardExperimentConfig.BASIC, RewardExperimentConfig.LOAD_EFFICIENCY,
                              RewardExperimentConfig.ROLE_SPECIALIZATION, RewardExperimentConfig.COLLABORATION,
                              RewardExperimentConfig.BEHAVIOR_SHAPING, RewardExperimentConfig.FULL]
            
            prev_perf = None
            for level in experiment_order:
                if level in performance_data:
                    current_perf = performance_data[level]
                    if prev_perf is not None:
                        improvement = current_perf - prev_perf
                        improvement_pct = (improvement / abs(prev_perf)) * 100 if prev_perf != 0 else 0
                        report_lines.append(f"   {level} vs 前一级: {improvement:+.2f} ({improvement_pct:+.1f}%)")
                    prev_perf = current_perf
        
        # 保存报告
        report_path = os.path.join(self.results_dir, output_file)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        debug_print(f"📊 对比报告已生成: {report_path}")
        
        # 也打印到控制台
        for line in report_lines:
            debug_print(line)
    
    def plot_performance_comparison(self, output_file="performance_comparison.png"):
        """绘制性能对比图"""
        if not self.experiment_data:
            debug_print("❌ 没有加载任何实验数据")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('增量式奖励实验性能对比', fontsize=16, fontweight='bold')
        
        # 1. 奖励曲线对比
        ax1 = axes[0, 0]
        for level, data in self.experiment_data.items():
            episode_rewards = np.array(data['episode_rewards'])
            if episode_rewards.ndim > 1:
                total_rewards = np.sum(episode_rewards, axis=1)
            else:
                total_rewards = episode_rewards
            
            # 平滑处理
            window_size = min(50, len(total_rewards) // 10)
            if window_size > 1:
                smoothed = pd.Series(total_rewards).rolling(window=window_size).mean()
                ax1.plot(smoothed, label=level.upper(), linewidth=2)
            else:
                ax1.plot(total_rewards, label=level.upper(), linewidth=2)
        
        ax1.set_title('训练奖励曲线对比')
        ax1.set_xlabel('训练回合')
        ax1.set_ylabel('总奖励')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 最终性能对比
        ax2 = axes[0, 1]
        levels = []
        final_performances = []
        
        for level, data in self.experiment_data.items():
            episode_rewards = np.array(data['episode_rewards'])
            if episode_rewards.ndim > 1:
                final_perf = np.mean(np.sum(episode_rewards[-100:], axis=1))
            else:
                final_perf = np.mean(episode_rewards[-100:])
            
            levels.append(level.upper())
            final_performances.append(final_perf)
        
        bars = ax2.bar(levels, final_performances, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
        ax2.set_title('最终性能对比 (最后100轮平均)')
        ax2.set_ylabel('平均总奖励')
        ax2.tick_params(axis='x', rotation=45)
        
        # 在柱状图上添加数值标签
        for bar, perf in zip(bars, final_performances):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(final_performances)*0.01,
                    f'{perf:.1f}', ha='center', va='bottom')
        
        # 3. 冲突率对比
        ax3 = axes[1, 0]
        conflict_levels = []
        conflict_rates = []
        
        for level, data in self.experiment_data.items():
            if 'collaboration_analytics' in data and data['collaboration_analytics']:
                analytics = data['collaboration_analytics']
                conflict_rate = analytics.get('conflict_rate', 0)
                conflict_levels.append(level.upper())
                conflict_rates.append(conflict_rate)
        
        if conflict_rates:
            bars = ax3.bar(conflict_levels, conflict_rates, color='red', alpha=0.7)
            ax3.set_title('冲突率对比')
            ax3.set_ylabel('冲突率')
            ax3.tick_params(axis='x', rotation=45)
            
            for bar, rate in zip(bars, conflict_rates):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + max(conflict_rates)*0.01,
                        f'{rate:.3f}', ha='center', va='bottom')
        else:
            ax3.text(0.5, 0.5, '暂无冲突数据', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('冲突率对比')
        
        # 4. 模块启用状态
        ax4 = axes[1, 1]
        modules = ['载重效率', '角色专业化', '协作模块', '行为塑造', '高级惩罚']
        module_keys = ['enable_load_efficiency', 'enable_role_specialization', 
                      'enable_collaboration', 'enable_behavior_shaping', 'enable_advanced_penalties']
        
        experiment_order = [RewardExperimentConfig.BASIC, RewardExperimentConfig.LOAD_EFFICIENCY,
                          RewardExperimentConfig.ROLE_SPECIALIZATION, RewardExperimentConfig.COLLABORATION,
                          RewardExperimentConfig.BEHAVIOR_SHAPING, RewardExperimentConfig.FULL]
        
        # 创建模块启用矩阵
        module_matrix = []
        level_labels = []
        
        for level in experiment_order:
            if level in self.experiment_data:
                config = EXPERIMENT_CONFIGS[level]
                row = []
                for key in module_keys:
                    row.append(1 if getattr(config, key, False) else 0)
                module_matrix.append(row)
                level_labels.append(level.upper())
        
        if module_matrix:
            im = ax4.imshow(module_matrix, cmap='RdYlGn', aspect='auto')
            ax4.set_xticks(range(len(modules)))
            ax4.set_xticklabels(modules, rotation=45, ha='right')
            ax4.set_yticks(range(len(level_labels)))
            ax4.set_yticklabels(level_labels)
            ax4.set_title('奖励模块启用状态')
            
            # 添加文本标注
            for i in range(len(level_labels)):
                for j in range(len(modules)):
                    text = '✓' if module_matrix[i][j] else '✗'
                    ax4.text(j, i, text, ha="center", va="center", 
                           color="black" if module_matrix[i][j] else "red", fontweight='bold')
        
        plt.tight_layout()
        
        # 保存图片
        output_path = os.path.join(self.results_dir, output_file)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        debug_print(f"📊 性能对比图已生成: {output_path}")
    
    def analyze_reward_module_impact(self):
        """分析每个奖励模块的影响"""
        if len(self.experiment_data) < 2:
            debug_print("❌ 需要至少2个实验结果进行对比分析")
            return
        
        debug_print("\n" + "🔍" * 60)
        debug_print("奖励模块影响分析")
        debug_print("🔍" * 60)
        
        # 按实验顺序分析
        experiment_order = [RewardExperimentConfig.BASIC, RewardExperimentConfig.LOAD_EFFICIENCY,
                          RewardExperimentConfig.ROLE_SPECIALIZATION, RewardExperimentConfig.COLLABORATION,
                          RewardExperimentConfig.BEHAVIOR_SHAPING, RewardExperimentConfig.FULL]
        
        prev_data = None
        prev_level = None
        
        for level in experiment_order:
            if level not in self.experiment_data:
                continue
                
            current_data = self.experiment_data[level]
            
            if prev_data is not None:
                # 计算性能变化
                prev_rewards = np.array(prev_data['episode_rewards'])
                curr_rewards = np.array(current_data['episode_rewards'])
                
                if prev_rewards.ndim > 1:
                    prev_perf = np.mean(np.sum(prev_rewards[-100:], axis=1))
                else:
                    prev_perf = np.mean(prev_rewards[-100:])
                
                if curr_rewards.ndim > 1:
                    curr_perf = np.mean(np.sum(curr_rewards[-100:], axis=1))
                else:
                    curr_perf = np.mean(curr_rewards[-100:])
                
                performance_change = curr_perf - prev_perf
                performance_change_pct = (performance_change / abs(prev_perf)) * 100 if prev_perf != 0 else 0
                
                # 分析新增的模块
                prev_config = EXPERIMENT_CONFIGS[prev_level]
                curr_config = EXPERIMENT_CONFIGS[level]
                
                new_modules = []
                if not prev_config.enable_load_efficiency and curr_config.enable_load_efficiency:
                    new_modules.append("载重效率模块")
                if not prev_config.enable_role_specialization and curr_config.enable_role_specialization:
                    new_modules.append("角色专业化模块")
                if not prev_config.enable_collaboration and curr_config.enable_collaboration:
                    new_modules.append("协作模块")
                if not prev_config.enable_behavior_shaping and curr_config.enable_behavior_shaping:
                    new_modules.append("行为塑造模块")
                if not prev_config.enable_advanced_penalties and curr_config.enable_advanced_penalties:
                    new_modules.append("高级惩罚模块")
                
                debug_print(f"\n📈 {prev_level.upper()} → {level.upper()}")
                debug_print(f"   新增模块: {', '.join(new_modules) if new_modules else '无'}")
                debug_print(f"   性能变化: {performance_change:+.2f} ({performance_change_pct:+.1f}%)")
                
                if performance_change > 0:
                    debug_print(f"   影响评估: ✅ 正面影响，性能提升")
                elif performance_change < -50:  # 显著下降
                    debug_print(f"   影响评估: ❌ 负面影响，性能显著下降")
                else:
                    debug_print(f"   影响评估: ⚠️ 轻微影响或需要更多训练")
                
                # 协作指标变化
                if 'collaboration_analytics' in prev_data and 'collaboration_analytics' in current_data:
                    prev_analytics = prev_data['collaboration_analytics']
                    curr_analytics = current_data['collaboration_analytics']
                    
                    if prev_analytics and curr_analytics:
                        prev_conflict_rate = prev_analytics.get('conflict_rate', 0)
                        curr_conflict_rate = curr_analytics.get('conflict_rate', 0)
                        conflict_change = curr_conflict_rate - prev_conflict_rate
                        
                        debug_print(f"   冲突率变化: {prev_conflict_rate:.3f} → {curr_conflict_rate:.3f} ({conflict_change:+.3f})")
            
            prev_data = current_data
            prev_level = level
        
        debug_print("🔍" * 60)

# 便捷使用函数
def analyze_latest_experiments():
    """分析最新的实验结果"""
    analyzer = ExperimentAnalyzer()
    
    # 加载所有实验结果
    loaded_count = analyzer.load_experiment_results()
    
    if loaded_count == 0:
        debug_print("❌ 未找到任何实验结果文件")
        return
    
    debug_print(f"✅ 已加载 {loaded_count} 个实验结果")
    
    # 生成对比报告
    analyzer.generate_comparison_report()
    
    # 生成可视化图表
    analyzer.plot_performance_comparison()
    
    # 分析模块影响
    analyzer.analyze_reward_module_impact()
    
    debug_print(f"\n🎯 分析完成！结果保存在 {analyzer.results_dir} 目录中")

if __name__ == "__main__":
    # 演示用法
    analyze_latest_experiments()

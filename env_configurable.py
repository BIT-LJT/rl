"""
可配置奖励的环境类 - 支持增量式奖励实验

这个环境类继承自SamplingEnv2_0，但可以根据reward_config中的实验等级
动态调整奖励函数，支持增量式奖励实验。

使用方法：
1. 在config.py中设置REWARD_EXPERIMENT_LEVEL
2. 在main.py中使用SamplingEnvConfigurable替换SamplingEnv2_0
3. 进行增量式实验，观察不同奖励模块的影响
"""

import numpy as np
from env import SamplingEnv2_0
from utils import debug_print
import config
from reward_config import EXPERIMENT_CONFIGS

class SamplingEnvConfigurable(SamplingEnv2_0):
    """可配置奖励的采样环境"""
    
    def __init__(self, num_points=30, num_agents=5):
        super().__init__(num_points, num_agents)
        
        # 获取当前实验配置
        self.reward_config = EXPERIMENT_CONFIGS[config.REWARD_EXPERIMENT_LEVEL]
        self.reward_constants = self.reward_config.get_reward_constants()
        
        # 初始化额外需要的统计属性
        self.agent_task_completion = np.zeros(self.num_agents, dtype=int)  # 每个智能体完成的任务数
        self.agent_high_priority_tasks = np.zeros(self.num_agents, dtype=int)  # 每个智能体完成的高优先级任务数
        
        # 输出实验信息
        debug_print(f"\n🧪 启动增量式奖励实验")
        debug_print(f"   实验等级: {config.REWARD_EXPERIMENT_LEVEL.upper()}")
        debug_print(f"   描述: {self.reward_config.get_experiment_description()}")
        debug_print(f"   预期行为: {', '.join(self.reward_config.get_expected_behaviors())}")
        debug_print(f"   关键指标: {', '.join(self.reward_config.get_key_metrics())}")
        
        # 输出启用的奖励模块
        self._print_enabled_modules()
    
    def reset(self):
        """重置环境，包括额外的统计属性"""
        obs = super().reset()
        
        # 重置额外的统计属性
        self.agent_task_completion = np.zeros(self.num_agents, dtype=int)
        self.agent_high_priority_tasks = np.zeros(self.num_agents, dtype=int)
        
        return obs
    
    def _print_enabled_modules(self):
        """输出当前启用的奖励模块"""
        debug_print(f"\n🎛️ 奖励模块状态:")
        debug_print(f"   基础奖励: ✅ (永远启用)")
        debug_print(f"   载重效率: {'✅' if self.reward_config.enable_load_efficiency else '❌'}")
        debug_print(f"   角色专业化: {'✅' if self.reward_config.enable_role_specialization else '❌'}")
        debug_print(f"   协作模块: {'✅' if self.reward_config.enable_collaboration else '❌'}")
        debug_print(f"   行为塑造: {'✅' if self.reward_config.enable_behavior_shaping else '❌'}")
        debug_print(f"   高级惩罚: {'✅' if self.reward_config.enable_advanced_penalties else '❌'}")
    
    def step(self, actions, current_step, max_steps):
        """
        可配置的step函数，根据实验等级应用不同的奖励模块
        """
        rewards = np.zeros(self.num_agents)
        actions = np.array(actions)
        
        # 统计总决策次数
        self.total_decisions += len([a for a in actions if a < self.num_points])
        
        # 获取奖励常量
        R = self.reward_constants
        
        # === 基础时间步惩罚（永远启用）===
        rewards += R['P_TIME_STEP']
        
        # === 处理充电状态 ===
        for i in range(self.num_agents):
            if self.agent_charging_status[i] > 0:
                self.agent_charging_status[i] -= 1
                if self.agent_charging_status[i] == 0:
                    self.agent_energy[i] = self.agent_energy_max[i]
        
        # === 冲突解决（协作模块）===
        if self.reward_config.enable_collaboration:
            rewards = self._handle_conflicts_with_rewards(actions, rewards)
        else:
            # 简单冲突处理：随机选择一个智能体
            self._handle_conflicts_simple(actions)
        
        # === 处理智能体动作 ===
        for i, target in enumerate(actions):
            if self.agent_charging_status[i] > 0:
                continue
                
            prev_pos = self.agent_positions[i].copy()
            self.agent_paths[i].append(prev_pos)
            
            # === 处理返回基地动作 ===
            if target == self.num_points + 1:
                rewards[i] += self._handle_return_home(i, R)
                
            # === 处理充电动作 ===  
            elif target == self.num_points:
                rewards[i] += self._handle_charging(i, R)
                
            # === 处理任务点动作 ===
            elif target < self.num_points:
                # 记录任务责任分配
                self._assign_task_responsibility(target, i, current_step)
                rewards[i] += self._handle_task_collection(i, target, R)
            
            # === 行为塑造奖励（如果启用）===
            if self.reward_config.enable_behavior_shaping:
                rewards[i] += self._calculate_shaping_rewards(i, target, prev_pos, R)
        
        # === 时间窗口处理 ===
        self._update_time_windows(R, rewards)
        
        # === 检查任务完成状态 ===
        obs = self._get_obs()
        all_tasks_processed = np.all(self.done_points == 1)
        all_agents_at_base = np.all([np.array_equal(pos, self.central_station) for pos in self.agent_positions])
        mission_accomplished = all_tasks_processed and all_agents_at_base
        
        # === 超时检查 ===
        is_timeout = current_step >= max_steps - 1
        tasks_not_finished_in_time = False
        
        if all_tasks_processed:
            if not hasattr(self, '_tasks_completed_step'):
                self._tasks_completed_step = current_step
            if current_step - self._tasks_completed_step > 100:
                tasks_not_finished_in_time = True
        
        done = mission_accomplished or is_timeout or tasks_not_finished_in_time
        
        # === 终局奖励处理 ===
        if done:
            rewards += self._calculate_final_rewards(mission_accomplished, is_timeout, 
                                                   tasks_not_finished_in_time, R)
        
        return obs, rewards, done
    
    def _handle_conflicts_with_rewards(self, actions, rewards):
        """处理冲突并应用协作奖励"""
        task_targets = actions[actions < self.num_points]
        unique_targets, counts = np.unique(task_targets, return_counts=True)
        conflicted_targets = unique_targets[counts > 1]
        
        assigned_tasks = set()
        
        for target_idx in conflicted_targets:
            competing_agent_indices = np.where(actions == target_idx)[0]
            self.conflict_count += 1
            
            # 智能冲突解决
            winner_idx = self._resolve_conflict_intelligently(target_idx, competing_agent_indices)
            assigned_tasks.add(target_idx)
            
            # 为败者分配替代任务
            losers = [idx for idx in competing_agent_indices if idx != winner_idx]
            for agent_idx in losers:
                alternative = self._find_best_alternative_task_intelligent(agent_idx, assigned_tasks)
                if alternative is not None:
                    actions[agent_idx] = alternative
                    assigned_tasks.add(alternative)
                    rewards[agent_idx] += self.reward_constants['P_CONFLICT'] * 0.3  # 减轻冲突惩罚
                else:
                    actions[agent_idx] = self.num_points
                    rewards[agent_idx] += self.reward_constants['P_CONFLICT']
        
        return rewards
    
    def _handle_conflicts_simple(self, actions):
        """简单冲突处理：随机分配"""
        task_targets = actions[actions < self.num_points]
        unique_targets, counts = np.unique(task_targets, return_counts=True)
        conflicted_targets = unique_targets[counts > 1]
        
        for target_idx in conflicted_targets:
            competing_indices = np.where(actions == target_idx)[0]
            winner = np.random.choice(competing_indices)
            
            for idx in competing_indices:
                if idx != winner:
                    actions[idx] = self.num_points  # 让败者回家
    
    def _handle_return_home(self, agent_id, R):
        """处理返回基地的动作"""
        reward = 0.0
        
        # 移动到基地
        dist = self._distance(self.agent_positions[agent_id], self.central_station)
        energy_cost = dist * self.energy_cost_factor[agent_id]
        
        if self.agent_energy[agent_id] < energy_cost:
            return R['P_ENERGY_DEPLETED']
        
        self.agent_positions[agent_id] = self.central_station
        self.agent_energy[agent_id] -= energy_cost
        
        # 载重奖励处理
        if self.agent_loads[agent_id] > 0:
            load_ratio = self.agent_loads[agent_id] / self.agent_capacity[agent_id]
            
            # 基础返回奖励
            reward += R['R_RETURN_HOME_COEFF'] * self.agent_loads[agent_id]
            
            # 载重效率模块奖励
            if self.reward_config.enable_load_efficiency:
                # 满载奖励
                if load_ratio >= 0.6:
                    reward += R['R_FULL_LOAD_BONUS']
                
                # 大载重智能体额外奖励
                if agent_id >= 3 and load_ratio >= 0.8:
                    reward += R['R_ROLE_CAPACITY_BONUS']
                
                # 高容量智能体低载惩罚
                if agent_id >= 3 and load_ratio < 0.6:
                    penalty_intensity = 1.0 - load_ratio
                    low_load_penalty = R['P_LOW_LOAD_HEAVY_AGENT'] * penalty_intensity
                    reward += low_load_penalty
            
            # 记录载重利用率
            self.agent_load_utilization[agent_id].append(load_ratio)
            
            # 清空载重和样本
            self.agent_loads[agent_id] = 0
            self.agent_samples[agent_id] = []
        else:
            # 空载返回处理
            if self.reward_config.enable_load_efficiency:
                # 空载惩罚
                agent_type = "重载" if agent_id >= 3 else "快速"
                final_penalty = R['P_EMPTY_RETURN']
                reward += final_penalty
                
                debug_print(f"   ❌ 空载回家 ({agent_type}智能体, 惩罚: {final_penalty:.1f})")
        
        return reward
    
    def _handle_charging(self, agent_id, R):
        """处理充电动作"""
        if not np.array_equal(self.agent_positions[agent_id], self.central_station):
            return 0.0  # 不在基地无法充电
        
        if self.agent_energy[agent_id] >= self.agent_energy_max[agent_id] * 0.9:
            return 0.0  # 能量充足无需充电
        
        # 开始充电
        charge_time = self.fast_charge_time if agent_id < 3 else self.slow_charge_time
        self.agent_charging_status[agent_id] = charge_time
        self.agent_charge_counts[agent_id] += 1
        
        return 0.0  # 充电本身不给奖励
    
    def _handle_task_collection(self, agent_id, target, R):
        """处理任务采集动作"""
        reward = 0.0
        
        # 获取任务优先级（在开始就获取，用于统计）
        priority = self.priority[target]
        
        # 检查任务是否已完成
        if self.done_points[target] == 1:
            if self.reward_config.enable_advanced_penalties:
                reward += R['P_INVALID_TASK_ATTEMPT']
            else:
                reward += -10.0  # 轻微惩罚
            return reward
        
        # 检查是否会超载
        if self.agent_loads[agent_id] + self.samples[target] > self.agent_capacity[agent_id]:
            if self.reward_config.enable_advanced_penalties:
                reward += R['P_OVERLOAD_ATTEMPT']
            else:
                reward += -10.0  # 轻微惩罚
            return reward
        
        # 移动到任务点
        dist = self._distance(self.agent_positions[agent_id], self.points[target])
        energy_cost = dist * self.energy_cost_factor[agent_id]
        
        if self.agent_energy[agent_id] < energy_cost:
            return R['P_ENERGY_DEPLETED']
        
        # 执行移动和采集
        self.agent_positions[agent_id] = self.points[target]
        self.agent_energy[agent_id] -= energy_cost
        self.agent_loads[agent_id] += self.samples[target]
        self.agent_samples[agent_id].append(target)
        self.done_points[target] = 1
        self.successfully_collected_points[target] = 1
        
        # 基础采集奖励
        reward += R['R_COLLECT_BASE']
        
        # 角色专业化奖励
        if self.reward_config.enable_role_specialization:
            if priority >= 4 and agent_id < 3:  # 快速智能体处理高优先级
                reward += R['R_FAST_AGENT_HIGH_PRIORITY']
                
                # 时间奖励：基于剩余时间窗口给予奖励
                remaining_time = self.time_windows[target]
                # 根据任务优先级获取初始时间窗口
                initial_time = (3*60) if priority == 1 else ((10*60) if priority == 2 else (30*60))
                time_utilization = remaining_time / initial_time
                
                # 如果在时间窗口的前50%完成，给予时间奖励
                if time_utilization > 0.5:
                    time_bonus = R['R_ROLE_SPEED_BONUS'] * time_utilization
                    reward += time_bonus
            
            elif priority >= 4 and agent_id >= 3:  # 重载智能体处理高优先级
                reward += R['P_HEAVY_AGENT_HIGH_PRIORITY']
            
            elif priority == 3 and agent_id >= 3 and self.samples[target] >= 3:
                # 重载智能体处理高载重低优先级任务
                reward += R['R_ROLE_CAPACITY_BONUS'] * 0.5
        
        # 记录任务完成信息
        self.agent_task_completion[agent_id] += 1
        if priority >= 4:
            self.agent_high_priority_tasks[agent_id] += 1
        
        return reward
    
    def _calculate_shaping_rewards(self, agent_id, target, prev_pos, R):
        """计算行为塑造奖励"""
        if not self.reward_config.enable_behavior_shaping:
            return 0.0
        
        reward = 0.0
        current_pos = self.agent_positions[agent_id]
        
        # 接近目标任务点的奖励
        if target < self.num_points and not self.done_points[target]:
            target_pos = self.points[target]
            prev_dist = self._distance(prev_pos, target_pos)
            curr_dist = self._distance(current_pos, target_pos)
            
            if curr_dist < prev_dist:
                approach_reward = (prev_dist - curr_dist) * R['R_COEFF_APPROACH_TASK']
                reward += min(approach_reward, R['REWARD_SHAPING_CLIP'])
        
        # 接近基地的奖励（有载重或低能量时）
        elif target == self.num_points + 1:
            base_dist_prev = self._distance(prev_pos, self.central_station)
            base_dist_curr = self._distance(current_pos, self.central_station)
            
            if base_dist_curr < base_dist_prev:
                # 有载重时的接近奖励
                if self.agent_loads[agent_id] > 0:
                    approach_reward = (base_dist_prev - base_dist_curr) * R['R_COEFF_APPROACH_HOME_LOADED']
                    reward += min(approach_reward, R['REWARD_SHAPING_CLIP'])
                
                # 低能量时的接近奖励
                energy_ratio = self.agent_energy[agent_id] / self.agent_energy_max[agent_id]
                if energy_ratio < 0.3:
                    approach_reward = (base_dist_prev - base_dist_curr) * R['R_COEFF_APPROACH_HOME_LOW_ENERGY']
                    reward += min(approach_reward, R['REWARD_SHAPING_CLIP'])
        
        return reward
    
    def _update_time_windows(self, R, rewards):
        """更新时间窗口并处理超时"""
        for i in range(self.num_points):
            if self.done_points[i] == 0:
                self.time_windows[i] = max(0, self.time_windows[i] - 1)
                
                if self.time_windows[i] <= 0:
                    # 任务超时
                    assigned_agent = self._get_last_assigned_agent(i)
                    if assigned_agent is not None:
                        timeout_penalty = R['P_TIMEOUT_BASE'] * (4 - self.priority[i])
                        rewards[assigned_agent] += timeout_penalty
                        debug_print(f"⏰ 任务点 {i} (优先级:{self.priority[i]}) 超时，智能体 {assigned_agent} 承担责任，惩罚: {timeout_penalty:.1f}")
                        
                        # 添加任务分配历史信息用于调试
                        if hasattr(self, 'task_assignment_history') and i in self.task_assignment_history:
                            history = self.task_assignment_history[i]
                            debug_print(f"   📋 任务分配历史: {history}")
                    else:
                        debug_print(f"⏰ 任务点 {i} (优先级:{self.priority[i]}) 超时，但无法找到负责的智能体")
                    
                    self.done_points[i] = 1  # 标记为已处理（超时）
    
    def _calculate_final_rewards(self, mission_accomplished, is_timeout, tasks_not_finished_in_time, R):
        """计算终局奖励"""
        final_rewards = np.zeros(self.num_agents)
        
        if mission_accomplished:
            # 任务完成奖励
            collected_count = np.sum(self.successfully_collected_points)
            processed_count = np.sum(self.done_points)
            timeout_count = processed_count - collected_count
            
            if timeout_count == 0:
                # 完美完成
                adjusted_reward = R['FINAL_BONUS_ACCOMPLISHED']
            else:
                # 部分超时，按比例调整
                timeout_penalty_ratio = timeout_count / len(self.done_points)
                adjusted_reward = R['FINAL_BONUS_ACCOMPLISHED'] * (1.0 - timeout_penalty_ratio * 0.3)
            
            # 按贡献分配奖励
            agent_contributions = np.array([self.agent_task_completion[i] for i in range(self.num_agents)])
            total_contribution = np.sum(agent_contributions)
            
            if total_contribution > 0:
                contribution_ratios = agent_contributions / total_contribution
                final_rewards = adjusted_reward * contribution_ratios
            else:
                equal_reward = adjusted_reward / self.num_agents
                final_rewards.fill(equal_reward)
        
        elif is_timeout and self.reward_config.enable_advanced_penalties:
            # 超时惩罚
            num_unprocessed = np.sum(self.done_points == 0)
            if num_unprocessed > 0:
                final_penalty = -50.0 * num_unprocessed  # 使用固定惩罚值
                final_rewards.fill(final_penalty / self.num_agents)
            
            # 未返回基地的额外惩罚
            for i in range(self.num_agents):
                if not np.array_equal(self.agent_positions[i], self.central_station):
                    final_rewards[i] += -100.0
        
        return final_rewards
    
    def get_experiment_summary(self):
        """获取当前实验的总结信息"""
        analytics = self.get_collaboration_analytics()
        
        summary = {
            'experiment_level': config.REWARD_EXPERIMENT_LEVEL,
            'description': self.reward_config.get_experiment_description(),
            'enabled_modules': {
                'load_efficiency': self.reward_config.enable_load_efficiency,
                'role_specialization': self.reward_config.enable_role_specialization,
                'collaboration': self.reward_config.enable_collaboration,
                'behavior_shaping': self.reward_config.enable_behavior_shaping,
                'advanced_penalties': self.reward_config.enable_advanced_penalties,
            },
            'key_metrics': self.reward_config.get_key_metrics(),
            'collaboration_analytics': analytics
        }
        
        return summary
    
    def _assign_task_responsibility(self, task_id, agent_id, current_step):
        """
        分配任务责任给智能体（继承自父类的功能）
        """
        # 确保责任追踪系统已初始化
        if not hasattr(self, 'task_assignments'):
            self.task_assignments = {}
            self.task_assignment_history = {}
            self.task_assignment_timestamp = {}
        
        # 记录当前负责该任务的智能体
        self.task_assignments[task_id] = agent_id
        self.task_assignment_timestamp[task_id] = current_step
        
        # 记录分配历史（用于调试和分析）
        if task_id not in self.task_assignment_history:
            self.task_assignment_history[task_id] = []
        self.task_assignment_history[task_id].append((agent_id, current_step))
    
    def _get_last_assigned_agent(self, point_id):
        """
        获取最后被分配给该任务的智能体（继承自父类的功能）
        """
        # 优先使用任务责任分配记录
        if hasattr(self, 'task_assignments') and point_id in self.task_assignments:
            return self.task_assignments[point_id]
        
        # 如果没有分配记录，使用实际访问者记录（向后兼容）
        return self.point_last_visitor.get(point_id, None)

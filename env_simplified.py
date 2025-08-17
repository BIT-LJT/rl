"""
简化版环境 - 仅包含核心奖励机制，用于调试和基线测试

这个版本移除了复杂的奖励塑造，只保留最核心的奖励/惩罚：
1. 采集任务的基础奖励
2. 时间步惩罚  
3. 最终完成奖励
4. 超时惩罚
5. 未返回基地惩罚

使用方法：
在main.py中将 SamplingEnv2_0 替换为 SamplingEnvSimplified
"""

from env import SamplingEnv2_0
import numpy as np
from utils import debug_print

class SamplingEnvSimplified(SamplingEnv2_0):
    """简化版采样环境 - 用于调试和基线测试"""
    
    def step(self, actions, current_step, max_steps):
        """
        简化版step函数，只保留核心奖励机制
        """
        rewards = np.zeros(self.num_agents)
        actions = np.array(actions)
        
        # 核心奖励参数
        R_COLLECT_BASE = 100.0           # 采集基础奖励
        P_TIME_STEP = -0.1               # 时间步惩罚
        FINAL_BONUS_ACCOMPLISHED = 1000.0  # 最终完成奖励
        FINAL_PENALTY_UNVISITED = -50.0    # 未访问任务惩罚
        FINAL_PENALTY_NOT_AT_BASE = -100.0 # 未返回基地惩罚
        
        # 基础时间步惩罚
        rewards += P_TIME_STEP
        
        # 处理充电状态
        for i in range(self.num_agents):
            if self.agent_charging_status[i] > 0:
                self.agent_charging_status[i] -= 1
                if self.agent_charging_status[i] == 0:
                    self.agent_energy[i] = self.agent_energy_max[i]
        
        # 处理智能体动作
        for i, target in enumerate(actions):
            if self.agent_charging_status[i] > 0:
                continue
                
            prev_pos = self.agent_positions[i].copy()
            self.agent_paths[i].append(prev_pos)
            
            if target == self.num_points + 1:  # 回基地
                dist = self._distance(prev_pos, self.central_station)
                energy_cost = dist * self.energy_cost_factor[i]
                if self.agent_energy[i] >= energy_cost:
                    self.agent_positions[i] = self.central_station
                    self.agent_energy[i] -= energy_cost
                    
                    # 简化的卸载逻辑
                    if self.agent_loads[i] > 0:
                        for point_idx in self.agent_samples[i]:
                            remaining_time = self.time_windows[point_idx]
                            self.delivery_info.append((point_idx, remaining_time, i))
                        
                        self.agent_loads[i] = 0
                        self.agent_samples[i] = []
                        self.agent_task_status[i] = 0
                        
            elif target == self.num_points:  # 待机/充电
                is_at_base = np.array_equal(self.agent_positions[i], self.central_station)
                needs_charge = self.agent_energy[i] < self.agent_energy_max[i]
                
                if is_at_base and needs_charge:
                    if i < 3:
                        self.agent_charging_status[i] = self.fast_charge_time
                    else:
                        self.agent_charging_status[i] = self.slow_charge_time
                        
            elif target < self.num_points:  # 执行任务
                if self.done_points[target] == 0:  # 任务未完成
                    dist = self._distance(prev_pos, self.points[target])
                    energy_cost = dist * self.energy_cost_factor[i]
                    
                    if (self.agent_energy[i] >= energy_cost and 
                        self.agent_loads[i] + self.samples[target] <= self.agent_capacity[i]):
                        
                        # 执行任务
                        self.agent_positions[i] = self.points[target]
                        self.agent_energy[i] -= energy_cost
                        self.agent_loads[i] += self.samples[target]
                        self.agent_samples[i].append(target)
                        self.successfully_collected_points[target] = 1
                        self.done_points[target] = 1
                        self.queue_done.put(target)
                        
                        # 核心奖励：基础采集奖励
                        rewards[i] += R_COLLECT_BASE
                        
                        if self.agent_loads[i] >= self.agent_capacity[i]:
                            self.agent_task_status[i] = 1
        
        # 时间推进和超时处理
        self.time += 1
        self.time_windows -= 1
        
        for i in range(self.num_points):
            if self.done_points[i] == 0 and self.time_windows[i] < 0:
                self.done_points[i] = 1  # 标记为已处理（超时）
        
        # Episode结束条件
        all_tasks_processed = np.all(self.done_points == 1)
        all_agents_at_base = all(np.array_equal(pos, self.central_station) for pos in self.agent_positions)
        mission_accomplished = all_tasks_processed and all_agents_at_base
        
        is_timeout = current_step >= max_steps - 1
        
        # 超时保护
        tasks_not_finished_in_time = False
        if all_tasks_processed:
            if not hasattr(self, '_tasks_completed_step'):
                self._tasks_completed_step = current_step
            if current_step - self._tasks_completed_step > 100:
                tasks_not_finished_in_time = True
        
        done = mission_accomplished or is_timeout or tasks_not_finished_in_time
        
        # 终局奖励/惩罚
        if done:
            if mission_accomplished:
                rewards += FINAL_BONUS_ACCOMPLISHED
                debug_print(f"✅ 简化环境：所有任务完成，获得完成奖励: {FINAL_BONUS_ACCOMPLISHED}")
            elif is_timeout:
                num_unprocessed = np.sum(self.done_points == 0)
                if num_unprocessed > 0:
                    final_penalty = FINAL_PENALTY_UNVISITED * num_unprocessed
                    rewards += final_penalty
                    debug_print(f"⚠️ 简化环境：超时，{num_unprocessed}个任务未完成，惩罚: {final_penalty}")
                
                for i in range(self.num_agents):
                    if not np.array_equal(self.agent_positions[i], self.central_station):
                        rewards[i] += FINAL_PENALTY_NOT_AT_BASE
                        debug_print(f"❌ 简化环境：智能体{i}未返回基地，惩罚: {FINAL_PENALTY_NOT_AT_BASE}")
        
        return self._get_obs(), rewards, done

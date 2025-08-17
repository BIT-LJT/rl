import numpy as np
import random
import queue
import config
from utils import debug_print
class SamplingEnv2_0:
    def __init__(self, num_points=30, num_agents=5):
        self.size = 5000
        self.num_points = num_points
        self.num_agents = num_agents
        self.action_dim = self.num_points + 2

        self.agent_capacity = np.array(config.agent_capacity)#智能体容量
        self.agent_speed = np.array(config.agent_speed)#智能体速度
        self.agent_energy_max = np.array(config.agent_energy_max)#智能体能量最大值

        BASE_SPEED = np.max(self.agent_speed)#智能体速度最大值
        self.energy_cost_factor = BASE_SPEED / self.agent_speed#能量消耗因子

        self.fast_charge_time = config.fast_charge_time
        self.slow_charge_time = config.slow_charge_time

        self.queue_done = queue.Queue()
        self.queue_new_point = queue.Queue()

        self.reset()

    def reset(self):
        self.points = np.random.randint(0, self.size, (self.num_points, 2))
        self.samples = np.random.randint(1, 6, self.num_points)
        self.priority = np.random.choice([1, 2, 3], self.num_points, p=[0.3, 0.4, 0.3])
        # 更紧迫的时间窗口设计：让速度优势更明显
        self.time_windows = np.array([(3*60) if p == 1 else ((10*60) if p == 2 else (30*60)) for p in self.priority], dtype=float)
        # 高优先级: 3分钟 (180秒) - 真正紧迫
        # 中优先级: 10分钟 (600秒) - 中等紧迫  
        # 低优先级: 30分钟 (1800秒) - 相对宽松
        
        self.done_points = np.zeros(self.num_points)
        self.successfully_collected_points = np.zeros(self.num_points)
        self.traffic_map = np.ones((self.size // 500, self.size // 500))
        self.agent_positions = np.full((self.num_agents, 2), self.size // 2, dtype=float)
        self.agent_loads = np.zeros(self.num_agents)
        self.agent_energy = self.agent_energy_max.copy().astype(float)
        self.agent_samples = [[] for _ in range(self.num_agents)]
        self.agent_task_status = np.zeros(self.num_agents)
        
        self.agent_charging_status = np.zeros(self.num_agents, dtype=int)
        self.agent_charge_counts = np.zeros(self.num_agents, dtype=int)
        # 纯奖励系统：移除休息状态管理，让智能体自主学习
        self.agent_rest_reward_given = np.zeros(self.num_agents, dtype=bool)  # 追踪任务完成后休息奖励是否已给予
        
        # 新增: 用于存储每个样本的送达信息
        self.delivery_info = []
        
        # 协作分析统计变量
        self.conflict_count = 0  # 冲突次数
        self.total_decisions = 0  # 总决策次数
        self.agent_task_priorities = [[] for _ in range(self.num_agents)]  # 每个智能体处理的任务优先级
        self.agent_load_utilization = [[] for _ in range(self.num_agents)]  # 每个智能体的载重利用率

        # 智能体的家/基地：出发点、卸载点、充电点、休息点
        self.central_station = np.array([self.size // 2, self.size // 2], dtype=float)  # 也可称为 self.home_base

        self.time = 0
        self.spawn_count = 0
        self.new_points_num = 0
        self.point_last_visitor = {}
        self.agent_paths = [[] for _ in range(self.num_agents)]
        self.task_creation_time = np.full(self.num_points, self.time)
        
        # === 任务责任追踪系统 ===
        self.task_assignments = {}  # 记录每个任务点当前的负责智能体
        self.task_assignment_history = {}  # 记录任务分配历史，用于调试
        self.task_assignment_timestamp = {}  # 记录任务分配的时间戳
        
        # 增强鲁棒性：初始化任务完成步骤标记，避免AttributeError
        self._tasks_completed_step = -1
        
        # 新增：跟踪上一步动作，用于决策摇摆惩罚
        self.last_actions = [self.num_points] * self.num_agents  # 初始化为待机动作

        return self._get_obs()

    def _get_action_masks(self):
        """
        生成一个全面的动作掩码，主动屏蔽所有已知的无效动作。
        """
        masks = np.ones((self.num_agents, self.action_dim), dtype=bool)
        
        # 1. 屏蔽所有已完成的任务点 (全局规则)
        done_task_indices = np.where(self.done_points == 1)[0]
        if len(done_task_indices) > 0:
            masks[:, done_task_indices] = False

        for i in range(self.num_agents):
            # 2. 如果智能体正在充电，它只能待在原地
            if self.agent_charging_status[i] > 0:
                masks[i, :] = False  # 屏蔽所有其他动作
                masks[i, self.num_points] = True  # 只允许"待机/充电"这个动作
                continue  # 处理下一个智能体

            # 3. 检查并屏蔽会导致超载的任务点
            current_load = self.agent_loads[i]
            agent_capacity = self.agent_capacity[i]
            for p_idx in range(self.num_points):
                # 如果该任务点本身是可选的 (尚未被全局规则屏蔽)
                if masks[i, p_idx]:
                    task_sample_size = self.samples[p_idx]
                    if current_load + task_sample_size > agent_capacity:
                        masks[i, p_idx] = False # 屏蔽这个会导致超载的任务
                        
        return masks

    def _get_obs(self):
        # 计算全局任务完成信息
        total_tasks = len(self.done_points)
        completed_tasks = np.sum(self.done_points)
        task_completion_ratio = completed_tasks / total_tasks if total_tasks > 0 else 0.0
        all_tasks_completed = float(np.all(self.done_points == 1))
        
        return {
            "agent_positions": self.agent_positions.copy(),
            "points": self.points.copy(),
            "samples": self.samples.copy(),
            "priority": self.priority.copy(),
            "time_windows": self.time_windows.copy(),
            "done_points": self.done_points.copy(),
            "agent_loads": self.agent_loads.copy(),
            "agent_energy": self.agent_energy.copy(),
            "traffic_map": self.traffic_map.copy(),
            "agent_task_status": self.agent_task_status.copy(),
            "agent_charging_status": self.agent_charging_status.copy(),
            "action_masks": self._get_action_masks(),
            # 新增全局信息
            "task_completion_ratio": task_completion_ratio,
            "all_tasks_completed": all_tasks_completed,
            "total_loaded_agents": float(np.sum(self.agent_loads > 0)),
            # --- 新增: 智能体意图信息（所有智能体的上一时刻动作）---
            "agent_last_actions": np.array(self.last_actions.copy()) if config.ENABLE_AGENT_INTENTION_OBS else np.array([self.num_points] * self.num_agents)
        }

    def _distance(self, pos1, pos2):
        dist = np.linalg.norm(pos1 - pos2)
        mid_point = (pos1 + pos2) / 2
        grid_x = int(mid_point[0] // 500)
        grid_y = int(mid_point[1] // 500)
        grid_x = np.clip(grid_x, 0, self.traffic_map.shape[0] - 1)
        grid_y = np.clip(grid_y, 0, self.traffic_map.shape[1] - 1)
        
        traffic_factor = self.traffic_map[grid_x, grid_y]
        return dist / traffic_factor

    def step(self, actions, current_step, max_steps):
        rewards = np.zeros(self.num_agents)
        actions = np.array(actions)
        
        # 统计总决策次数（每个智能体的每次行动都算一次决策）
        self.total_decisions += len([a for a in actions if a < self.num_points])  # 只统计实际的任务点选择
        
        R_COLLECT_BASE = 50.0#采集基础奖励
        R_RETURN_HOME_COEFF = 0.5#回家奖励系数
        R_FULL_LOAD_BONUS = 300.0#满载奖励
        R_COEFF_APPROACH_TASK = 0.1#接近目标点奖励系数
        R_COEFF_APPROACH_HOME_LOADED = 0.05 #接近目标点奖励
        R_COEFF_APPROACH_HOME_LOW_ENERGY = 0.1 #接近基地奖励
        REWARD_SHAPING_CLIP = 10.0 #奖励剪裁
        R_ROLE_SPEED_BONUS = 20.0#角色速度奖励
        R_ROLE_CAPACITY_BONUS = 50.0#角色载重奖励
        R_FAST_AGENT_HIGH_PRIORITY = 50.0  # 快速智能体处理高优先级任务的专业化奖励（从60.0调整为50.0）
        R_STAY_HOME_AFTER_COMPLETION = 10.0  # 任务完成后留在基地(central_station)的奖励
        R_SMART_RETURN_AFTER_COMPLETION = 50.0  # 任务完成后智能选择回家(central_station)卸载的奖励
        FINAL_BONUS_ACCOMPLISHED = 1000.0#最终奖励

        P_TIME_STEP = -0.1 #时间步惩罚
        P_INACTIVITY = -10.0 #不活动惩罚
        P_ENERGY_DEPLETED = -50.0 #能量不足惩罚
        P_TIMEOUT_BASE = -30.0 #超时惩罚
        P_EMPTY_RETURN = -20.0  # 空载回家惩罚
        P_SWING_PENALTY = -5.0  # 新增：决策摇摆惩罚
        P_WASTED_MOVE_PENALTY = -1.0  # 新增：选择已完成任务的惩罚
        P_LOW_LOAD_HEAVY_AGENT = -15.0  # 高容量智能体低载返回惩罚
        P_HEAVY_AGENT_HIGH_PRIORITY = -30.0  # 重载智能体处理高优先级任务的效率惩罚
        P_NOT_RETURN_AFTER_COMPLETION = -20.0  # 任务完成后有载重但不回家(central_station)的惩罚
        P_INVALID_TASK_ATTEMPT = -100.0  # 尝试执行已完成任务的重大惩罚
        P_OVERLOAD_ATTEMPT = -80.0  # 尝试超载的重大惩罚
        P_POINTLESS_ACTION = -20.0  # 无意义行动的惩罚
        P_TASK_COMPLETED_TIME_PENALTY = -5.0  # 任务完成后每时间步的惩罚
        FINAL_PENALTY_UNVISITED = -50.0#未访问惩罚
        FINAL_PENALTY_NOT_AT_BASE = -100.0#未回家惩罚

        rewards += P_TIME_STEP
        
        # 检查是否所有任务已完成，如果是，给予时间惩罚以激励快速返回
        all_tasks_completed = np.all(self.done_points == 1)
        if all_tasks_completed:
            for i in range(self.num_agents):
                # 对所有智能体施加时间惩罚，激励快速结束任务
                rewards[i] += P_TASK_COMPLETED_TIME_PENALTY
                # 对仍有载重的智能体施加额外惩罚
                if self.agent_loads[i] > 0:
                    rewards[i] += P_TASK_COMPLETED_TIME_PENALTY * 2  # 双倍惩罚
        
        for i in range(self.num_agents):
            if self.agent_charging_status[i] > 0:
                self.agent_charging_status[i] -= 1
                if self.agent_charging_status[i] == 0:
                    self.agent_energy[i] = self.agent_energy_max[i]

        # 智能冲突处理机制：基于任务优先级和智能体能力
        task_targets = actions[actions < self.num_points]
        unique_targets, counts = np.unique(task_targets, return_counts=True)
        conflicted_targets = unique_targets[counts > 1]
        
        # 记录所有已分配的任务点，避免重复分配
        assigned_tasks = set()

        for target_idx in conflicted_targets:
            competing_agent_indices = np.where(actions == target_idx)[0]
            # 记录冲突次数（每个冲突目标点计为一次冲突）
            self.conflict_count += 1
            
            # 使用智能分配算法选择最佳智能体
            winner_global_idx = self._resolve_conflict_intelligently(target_idx, competing_agent_indices)
            
            # === 记录获胜者的任务责任 ===
            self._assign_task_responsibility(target_idx, winner_global_idx, current_step)
            assigned_tasks.add(target_idx)
            
            # 改进：为败者智能排序后再分配替代任务
            losers = [agent_idx for agent_idx in competing_agent_indices if agent_idx != winner_global_idx]
            
            # 按照智能体对任务的适应性排序（距离、能量、载重等综合评分）
            loser_priorities = []
            for agent_idx in losers:
                # 计算该智能体的综合紧迫度得分
                agent_pos = self.agent_positions[agent_idx]
                energy_ratio = self.agent_energy[agent_idx] / self.agent_energy_max[agent_idx]
                load_ratio = self.agent_loads[agent_idx] / self.agent_capacity[agent_idx]
                
                # 距离基地的距离（距离越远越需要优先安排）
                distance_to_base = self._distance(agent_pos, self.central_station)
                
                # 综合优先级得分（越高越优先）
                priority_score = (
                    (1.0 - energy_ratio) * 0.4 +     # 能量越少越优先
                    load_ratio * 0.3 +               # 载重越多越优先
                    distance_to_base / (self.size * 1.414) * 0.3  # 距离基地越远越优先
                )
                loser_priorities.append((agent_idx, priority_score))
            
            # 按优先级排序（降序）
            loser_priorities.sort(key=lambda x: x[1], reverse=True)
            
            # 按优先级顺序为败者分配替代任务
            for agent_idx, priority_score in loser_priorities:
                # 使用智能分配算法寻找最佳替代任务
                alternative_task = self._find_best_alternative_task_intelligent(agent_idx, assigned_tasks)
                
                if alternative_task is not None:
                    actions[agent_idx] = alternative_task
                    # === 记录替代任务的责任分配 ===
                    self._assign_task_responsibility(alternative_task, agent_idx, current_step)
                    assigned_tasks.add(alternative_task)  # 记录已分配
                    # 冲突惩罚已删除，智能重分配不再给予惩罚
                    debug_print(f"🧠 智能体 {agent_idx} (优先级:{priority_score:.3f}) 智能重分配到任务点 {alternative_task}")
                else:
                    # 没有替代任务，才回家
                    actions[agent_idx] = self.num_points 
                    # 冲突惩罚已删除，智能体返回基地不再受惩罚
                    debug_print(f"🏠 智能体 {agent_idx} 冲突后无替代任务，返回基地")
        
        for i, target in enumerate(actions):
            if self.agent_charging_status[i] > 0:
                continue

            prev_pos = self.agent_positions[i].copy()
            self.agent_paths[i].append(prev_pos)
            
            # --- 新增: 决策摇摆惩罚 ---
            last_target = self.last_actions[i]
            if last_target < self.num_points and target < self.num_points and last_target != target:
                rewards[i] += P_SWING_PENALTY
                debug_print(f"🔄️ 智能体 {i} 决策摇摆 (从 {last_target} -> {target})，惩罚: {P_SWING_PENALTY}")
            
            # --- 新增: 无效移动惩罚 ---
            if target < self.num_points and self.done_points[target] == 1:
                rewards[i] += P_WASTED_MOVE_PENALTY
                debug_print(f"🚫 智能体 {i} 选择已完成任务 {target}，惩罚: {P_WASTED_MOVE_PENALTY}")
            
            # --- 新增: 连续能量惩罚机制 ---
            energy_ratio = self.agent_energy[i] / self.agent_energy_max[i]
            LOW_ENERGY_THRESHOLD = config.LOW_ENERGY_THRESHOLD  # 从config获取低能量阈值
            if energy_ratio < LOW_ENERGY_THRESHOLD:
                # 能量越低，惩罚越大（与能量剩余量成反比）
                energy_penalty_intensity = (LOW_ENERGY_THRESHOLD - energy_ratio) / LOW_ENERGY_THRESHOLD
                energy_penalty = P_TIME_STEP * config.ENERGY_PENALTY_MULTIPLIER * energy_penalty_intensity  # 从config获取惩罚倍数
                rewards[i] += energy_penalty
                if energy_ratio < 0.1:  # 极低能量时才打印提示，避免输出过多
                    debug_print(f"⚡ 智能体 {i} 能量过低 ({energy_ratio*100:.1f}%)，连续惩罚: {energy_penalty:.2f}")

            if target == self.num_points + 1:#基地
                dist = self._distance(prev_pos, self.central_station)#距离
                energy_cost = dist * self.energy_cost_factor[i]#能量消耗
                if self.agent_energy[i] < energy_cost:#能量不足
                    rewards[i] += P_ENERGY_DEPLETED
                    continue
                self.agent_positions[i] = self.central_station#更新位置
                self.agent_energy[i] -= energy_cost#更新能量
                
                # 检查是否是任务完成后主动选择回家（有载重时）
                all_tasks_completed = np.all(self.done_points == 1)#所有任务完成
                if all_tasks_completed and self.agent_loads[i] > 0:#有载重
                    # 奖励与载重率成正比：载重越多，奖励越大
                    load_ratio = self.agent_loads[i] / self.agent_capacity[i]
                    smart_return_reward = R_SMART_RETURN_AFTER_COMPLETION * load_ratio
                    rewards[i] += smart_return_reward
                    debug_print(f"🎯 智能体 {i} 任务完成后主动回家卸载(载重率{load_ratio*100:.1f}%)，获得智能奖励: {smart_return_reward:.1f}")
                
                # 输出智能体回家时的载重信息
                load_info = f"载重: {self.agent_loads[i]}/{self.agent_capacity[i]} ({self.agent_loads[i]/self.agent_capacity[i]*100:.1f}%)"
                debug_print(f"🏠 智能体 {i} 回到基地！{load_info}")
                
                if self.agent_loads[i] > 0:#有载重
                    # 记录载重利用率
                    load_utilization = self.agent_loads[i] / self.agent_capacity[i]#载重利用率
                    self.agent_load_utilization[i].append(load_utilization)#记录载重利用率
                    
                    debug_print(f"   📦 卸载样本: {len(self.agent_samples[i])} 个任务点的样本")
                    for point_idx in self.agent_samples[i]:
                        remaining_time = self.time_windows[point_idx]#剩余时间窗口
                        self.delivery_info.append((point_idx, remaining_time, i))#记录送回样本点
                        debug_print(f"   ✓ 送回样本点 {point_idx}，剩余时间窗口: {remaining_time:.2f} 秒")

                    rewards[i] += R_RETURN_HOME_COEFF * self.agent_loads[i]#奖励
                    
                    # 载重率分析
                    load_ratio = self.agent_loads[i] / self.agent_capacity[i]#载重率
                    
                    # 高载重奖励(≥760%)
                    if load_ratio >= 0.6:
                        rewards[i] += R_FULL_LOAD_BONUS#高载重奖励
                        debug_print(f"   🎉 高载重奖励！载重率 {load_ratio*100:.1f}%")
                    
                    # 重载智能体容量奖励(≥80%)
                    if i in [3, 4] and load_ratio >= 0.8:
                        rewards[i] += R_ROLE_CAPACITY_BONUS#大载重智能体额外奖励
                        debug_print(f"   🚛 大载重智能体额外奖励！")
                    
                    # --- 新增: 重载智能体非线性载重效率奖励 ---
                    # 使用平方项强化满载行为，激励"批量运输"策略
                    if i in [3, 4]:  # 重载智能体（Agent 3, 4）
                        load_bonus = config.R_LOAD_EFFICIENCY * (load_ratio ** 2) * self.agent_loads[i]
                        rewards[i] += load_bonus
                        debug_print(f"   🚚 重载智能体 {i} 高效运输，载重率 {load_ratio:.2f}，非线性奖励: {load_bonus:.2f}")
                    
                    # 高容量智能体低载惩罚(<60%)
                    if i in [3, 4] and load_ratio < 0.6:
                        # 根据载重率计算惩罚强度：载重率越低惩罚越重
                        penalty_intensity = (0.6 - load_ratio) / 0.6#惩罚强度
                        low_load_penalty = P_LOW_LOAD_HEAVY_AGENT * penalty_intensity#低载重惩罚
                        rewards[i] += low_load_penalty#奖励
                        debug_print(f"   ⚠️ 高容量智能体低载惩罚！载重率 {load_ratio*100:.1f}% (惩罚: {low_load_penalty:.1f})")
                    
                    self.agent_loads[i] = 0#载重清零
                    self.agent_samples[i] = []#样本清零
                    self.agent_task_status[i] = 0#任务状态清零
                    
                    # 卸载完成，智能体可以自主决定下一步行动
                else:
                    # 记录空载回家（载重利用率为0）并给予智能惩罚
                    self.agent_load_utilization[i].append(0.0)#记录空载回家
                    
                    # 计算剩余任务点数量来判断是否应该减轻空载惩罚
                    remaining_tasks = np.sum(self.done_points == 0)#剩余任务点数量
                    total_tasks = len(self.done_points)#总任务点数量      
                    task_scarcity = 1.0 - (remaining_tasks / total_tasks)  # 任务稀缺度 0-1
                    
                    # 计算智能空载惩罚
                    base_penalty = P_EMPTY_RETURN
                    
                    # 根据智能体类型调整惩罚：重载智能体空载惩罚更重
                    if i >= 3:  # 重载智能体
                        capacity_penalty = base_penalty * 1.5
                    else:  # 快速智能体
                        capacity_penalty = base_penalty
                    
                    # 根据能量状态调整惩罚：低能量时惩罚减轻
                    energy_ratio = self.agent_energy[i] / self.agent_energy_max[i]
                    if energy_ratio < 0.2:  # 能量不足20%时，惩罚减半
                        energy_adjustment = 0.5
                        reason = "低能量"
                    elif energy_ratio < 0.4:  # 能量不足40%时，惩罚减少25%
                        energy_adjustment = 0.75
                        reason = "较低能量"
                    else:
                        energy_adjustment = 1.0
                        reason = "充足能量"
                    
                    # 根据任务稀缺度调整惩罚：任务越少，空载惩罚越轻
                    if task_scarcity > 0.8:  # 任务完成度超过80%
                        scarcity_adjustment = 0.2  # 大幅减轻惩罚
                        scarcity_reason = "任务稀缺"
                    elif task_scarcity > 0.6:  # 任务完成度超过60%
                        scarcity_adjustment = 0.5  # 中度减轻惩罚
                        scarcity_reason = "任务较少"
                    else:
                        scarcity_adjustment = 1.0  # 正常惩罚
                        scarcity_reason = "任务充足"
                    
                    final_penalty = capacity_penalty * energy_adjustment * scarcity_adjustment
                    rewards[i] += final_penalty
                    
                    agent_type = "重载" if i >= 3 else "快速"
                    if scarcity_adjustment < 1.0:
                        debug_print(f"   ⚠️ 空载回家 ({agent_type}智能体, {reason}, {scarcity_reason}, 惩罚: {final_penalty:.1f})")
                    else:
                        debug_print(f"   ❌ 空载回家 ({agent_type}智能体, {reason}, 惩罚: {final_penalty:.1f})")
                
                debug_print(f"   ⚡ 剩余能量: {self.agent_energy[i]:.0f}/{self.agent_energy_max[i]:.0f} ({self.agent_energy[i]/self.agent_energy_max[i]*100:.1f}%)")
                debug_print()  # 空行，增加可读性
                
            elif target == self.num_points:
                is_at_base = np.array_equal(self.agent_positions[i], self.central_station)
                needs_charge = self.agent_energy[i] < self.agent_energy_max[i]
                all_tasks_completed = np.all(self.done_points == 1)
                
                if is_at_base and needs_charge:
                    if i < 3:
                        self.agent_charging_status[i] = self.fast_charge_time
                    else:
                        self.agent_charging_status[i] = self.slow_charge_time
                    self.agent_charge_counts[i] += 1
                elif is_at_base and all_tasks_completed and self.agent_loads[i] == 0:
                    # 任务完成后空载留在基地是正确行为，只在第一次给予奖励
                    if not self.agent_rest_reward_given[i]:
                        rewards[i] += R_STAY_HOME_AFTER_COMPLETION
                        self.agent_rest_reward_given[i] = True
                        debug_print(f"   🏠 智能体 {i} 任务完成后首次在基地休息，获得奖励: {R_STAY_HOME_AFTER_COMPLETION}")
                    # 后续停留不再给奖励，避免智能体学会永远待在基地
                # 注释：不存在"在基地有载重"的情况，因为一到基地就会自动卸载
                # elif is_at_base and all_tasks_completed and self.agent_loads[i] > 0:
                #     # 这个条件永远不会触发，因为在基地就已经自动卸载了
                else:
                    rewards[i] += P_INACTIVITY
                continue

            elif target < self.num_points:
                # === 记录任务责任分配 ===
                self._assign_task_responsibility(target, i, current_step)
                
                # 检查是否尝试执行已完成的任务
                if self.done_points[target] == 1:
                    rewards[i] += P_INVALID_TASK_ATTEMPT
                    debug_print(f"   ❌ 智能体 {i} 尝试执行已完成任务点 {target}，惩罚: {P_INVALID_TASK_ATTEMPT}")
                    continue
                
                # 检查是否会导致超载
                if self.agent_loads[i] + self.samples[target] > self.agent_capacity[i]:
                    rewards[i] += P_OVERLOAD_ATTEMPT
                    debug_print(f"   ⚠️ 智能体 {i} 尝试超载执行任务点 {target} (当前:{self.agent_loads[i]:.1f} + 新增:{self.samples[target]:.1f} > 容量:{self.agent_capacity[i]:.1f})，惩罚: {P_OVERLOAD_ATTEMPT}")
                    continue
                
                # 执行任务
                dist = self._distance(prev_pos, self.points[target])
                energy_cost = dist * self.energy_cost_factor[i]
                if self.agent_energy[i] < energy_cost:
                    rewards[i] += P_ENERGY_DEPLETED
                    actions[i] = self.num_points
                    continue
                self.agent_positions[i] = self.points[target]
                self.agent_energy[i] -= energy_cost
                self.agent_loads[i] += self.samples[target]
                self.agent_samples[i].append(target)
                self.successfully_collected_points[target] = 1
                self.done_points[target] = 1
                self.queue_done.put(target)
                self.point_last_visitor[target] = i
                if self.agent_loads[i] >= self.agent_capacity[i]:
                    self.agent_task_status[i] = 1
                
                # 记录任务优先级统计
                priority = self.priority[target]
                self.agent_task_priorities[i].append(priority)
                
                rewards[i] += R_COLLECT_BASE * (4 - priority)
                
                # 专业化奖励和惩罚机制
                if priority == 1:  # 高优先级任务
                    # 协作奖励已删除，智能体之间不再有协作奖励机制
                    
                    if i in [0, 1, 2]:  # 快速智能体处理高优先级任务
                        # 专业化奖励：快速智能体天然适合高优先级任务
                        rewards[i] += R_FAST_AGENT_HIGH_PRIORITY
                        debug_print(f"   ⚡ 快速智能体 {i} 专业化处理高优先级任务，获得奖励: {R_FAST_AGENT_HIGH_PRIORITY}")
                        
                        # 时间奖励：根据处理速度给予额外奖励
                        time_spent = self.time - self.task_creation_time[target]
                        time_window = 2.5 * 60  # 150秒，覆盖高优先级任务大部分时间
                        if time_spent < time_window:
                             time_bonus = R_ROLE_SPEED_BONUS * (1 - time_spent / time_window)
                             rewards[i] += time_bonus
                             debug_print(f"   ⚡ 快速处理时间奖励: {time_bonus:.1f}")
                    
                    else:  # 重载智能体处理高优先级任务
                        # 效率惩罚：重载智能体不适合高优先级任务
                        rewards[i] += P_HEAVY_AGENT_HIGH_PRIORITY
                        debug_print(f"   🐌 重载智能体 {i} 处理高优先级任务效率低，惩罚: {P_HEAVY_AGENT_HIGH_PRIORITY}")
                
                elif priority == 3 and i in [3, 4]:  # 重载智能体处理低优先级任务
                    # 重载智能体适合处理低优先级、高载重任务
                    if self.samples[target] >= 3:  # 高载重任务
                        rewards[i] += R_ROLE_CAPACITY_BONUS * 0.5  # 适度的容量奖励
                        debug_print(f"   🚛 重载智能体 {i} 处理高载重低优先级任务，容量奖励: {R_ROLE_CAPACITY_BONUS * 0.5:.1f}")

            current_pos = self.agent_positions[i]
            if target < self.num_points:
                approach_reward = R_COEFF_APPROACH_TASK * (self._distance(prev_pos, self.points[target]) - self._distance(current_pos, self.points[target]))
                rewards[i] += np.clip(approach_reward, -REWARD_SHAPING_CLIP, REWARD_SHAPING_CLIP)
            
            approach_home_delta = self._distance(prev_pos, self.central_station) - self._distance(current_pos, self.central_station)
            
            # 使用动态载重阈值来鼓励智能体回家
            dynamic_threshold = self._get_dynamic_load_threshold(i, current_step, max_steps)
            current_load_ratio = self.agent_loads[i] / self.agent_capacity[i]
            
            if current_load_ratio >= dynamic_threshold and self.agent_loads[i] > 0:
                # 根据动态阈值调整回家奖励强度
                threshold_bonus = 1.0 + (0.9 - dynamic_threshold)  # 阈值越低，奖励越高
                rewards[i] += np.clip(R_COEFF_APPROACH_HOME_LOADED * self.agent_loads[i] * approach_home_delta * threshold_bonus, -REWARD_SHAPING_CLIP, REWARD_SHAPING_CLIP)
                
                # 添加载重达到动态阈值的额外奖励
                if current_load_ratio >= dynamic_threshold:
                    load_threshold_bonus = 5.0 * (1.0 - dynamic_threshold)  # 阈值越低，达到阈值的奖励越高
                    rewards[i] += load_threshold_bonus
                    if load_threshold_bonus > 0:
                        debug_print(f"📈 智能体 {i} 达到动态载重阈值 {dynamic_threshold:.2f} (当前载重率: {current_load_ratio:.2f})，获得奖励: {load_threshold_bonus:.1f}")
            
            if self.agent_energy[i] < self.agent_energy_max[i] * 0.1:
                rewards[i] += np.clip(R_COEFF_APPROACH_HOME_LOW_ENERGY * approach_home_delta, -REWARD_SHAPING_CLIP, REWARD_SHAPING_CLIP)

        self.time += 1
        self.time_windows -= 1
        
        for i in range(self.num_points):
            if self.done_points[i] == 0 and self.time_windows[i] <= 0:
                assigned_agent = self._get_last_assigned_agent(i)
                if assigned_agent is not None:
                    timeout_penalty = P_TIMEOUT_BASE * (4 - self.priority[i])
                    rewards[assigned_agent] += timeout_penalty
                    debug_print(f"⏰ 任务点 {i} (优先级:{self.priority[i]}) 超时，智能体 {assigned_agent} 承担责任，惩罚: {timeout_penalty:.1f}")
                    
                    # 添加任务分配历史信息用于调试
                    if i in self.task_assignment_history:
                        history = self.task_assignment_history[i]
                        debug_print(f"   📋 任务分配历史: {history}")
                else:
                    debug_print(f"⏰ 任务点 {i} (优先级:{self.priority[i]}) 超时，但无法找到负责的智能体")
                
                self.done_points[i] = 1  # 标记为已处理（超时）

        # 纯奖励系统：不再自动管理休息状态，让智能体自主学习最优行为

        # 修复episode结束条件：所有任务点都已处理完毕（采集或超时）
        all_tasks_processed = np.all(self.done_points == 1)
        # 保留成功采集统计用于其他逻辑
        all_tasks_collected = np.all(self.successfully_collected_points == 1)
        all_agents_at_base = all(np.array_equal(pos, self.central_station) for pos in self.agent_positions)
        
        # 修复核心逻辑漏洞：要求所有智能体都必须返回基地才算完美完成
        # 这确保了FINAL_PENALTY_NOT_AT_BASE等惩罚机制能够正确发挥作用
        mission_accomplished = all_tasks_processed and all_agents_at_base
        
        # 调试信息：检查episode结束条件（仅在问题时显示）
        if all_tasks_processed and not mission_accomplished and current_step % 50 == 0:  # 每50步打印一次
            not_at_base_agents = [i for i, pos in enumerate(self.agent_positions) if not np.array_equal(pos, self.central_station)]
            debug_print(f"⚠️ 任务已处理完但episode未结束 - 智能体 {not_at_base_agents} 未返回基地")
            debug_print(f"   载重状态: {self.agent_loads}")
        elif mission_accomplished:
            processed_count = np.sum(self.done_points)
            collected_count = np.sum(self.successfully_collected_points)
            timeout_count = processed_count - collected_count
            debug_print(f"📍 Episode结束条件满足: 已处理={processed_count}/30 (采集={collected_count}, 超时={timeout_count}), 所有智能体已返回基地={all_agents_at_base}")
            
        is_timeout = current_step >= max_steps - 1
        
        # 额外的超时保护：如果任务处理完成后100步内未结束，强制结束
        tasks_not_finished_in_time = False
        if all_tasks_processed:
            # 记录任务完成时间
            if self._tasks_completed_step == -1:  # 使用初始值判断，更加鲁棒
                self._tasks_completed_step = current_step
            
            # 如果任务完成后超过100步还没结束，强制结束
            if current_step - self._tasks_completed_step > 100:
                tasks_not_finished_in_time = True
                debug_print(f"⏰ 强制结束Episode：任务已完成但运行过久 (任务完成后已过 {current_step - self._tasks_completed_step} 步)")
            
        done = mission_accomplished or is_timeout or tasks_not_finished_in_time

        if done:
            if mission_accomplished:
                # 根据任务完成情况调整奖励
                collected_count = np.sum(self.successfully_collected_points)
                timeout_count = np.sum(self.done_points) - collected_count
                
                # 基础完成奖励
                base_reward = FINAL_BONUS_ACCOMPLISHED
                
                # 如果有超时任务，适当减少奖励
                if timeout_count > 0:
                    timeout_penalty_ratio = timeout_count / len(self.done_points)
                    adjusted_reward = base_reward * (1.0 - timeout_penalty_ratio * 0.3)  # 最多减少30%
                    debug_print(f"任务处理完成！采集={collected_count}, 超时={timeout_count}, 调整后奖励: {adjusted_reward:.1f}")
                else:
                    adjusted_reward = base_reward
                    debug_print(f"完美完成！所有{collected_count}个任务点均被成功采集，获得完整奖励: {adjusted_reward:.1f}")
                
                # 改进的信誉分配机制：根据贡献分配奖励
                agent_contributions = np.zeros(self.num_agents)
                
                # 统计每个智能体的贡献（送回的样本点数量，按优先级加权）
                for point_idx, _, agent_id in self.delivery_info:
                    priority_weight = 4 - self.priority[point_idx]  # 高优先级=3分，中=2分，低=1分
                    agent_contributions[agent_id] += priority_weight
                
                total_contributions = np.sum(agent_contributions)
                if total_contributions > 0:
                    # 按贡献比例分配奖励
                    contribution_ratios = agent_contributions / total_contributions
                    final_rewards = adjusted_reward * contribution_ratios
                    rewards += final_rewards
                    debug_print(f"🎖️ 最终奖励按贡献分配:")
                    for i in range(self.num_agents):
                        if agent_contributions[i] > 0:
                            debug_print(f"   智能体{i}: 贡献={agent_contributions[i]:.1f}, 奖励={final_rewards[i]:.1f}")
                else:
                    # 如果没有成功采集（所有任务都超时），则平分基础奖励
                    equal_reward = adjusted_reward / self.num_agents
                    rewards += equal_reward
                    debug_print(f"⚠️ 无有效贡献记录，平分奖励: 每个智能体获得 {equal_reward:.1f}")
                
                if all_agents_at_base:
                    debug_print(f"   🎯 所有智能体已返回基地")
                else:
                    debug_print(f"   🎯 所有载重智能体已返回卸载")
                    # 给予空载且在基地的智能体额外奖励（如果之前没有给过）
                    for i in range(self.num_agents):
                        if (self.agent_loads[i] == 0 and 
                            np.array_equal(self.agent_positions[i], self.central_station) and
                            not self.agent_rest_reward_given[i]):
                            rewards[i] += R_STAY_HOME_AFTER_COMPLETION
                            self.agent_rest_reward_given[i] = True
                            debug_print(f"   🏠 智能体 {i} 任务完成后在基地待命，获得额外奖励: {R_STAY_HOME_AFTER_COMPLETION}")
            elif tasks_not_finished_in_time:
                # 强制结束时的处理
                debug_print("⏰ 由于任务完成后运行过久，强制结束episode")
                for i in range(self.num_agents):
                    if self.agent_loads[i] > 0:
                        # 有载重但未回家的智能体给予中等惩罚
                        rewards[i] += -100.0
                        debug_print(f"   ⚠️ 智能体 {i} 有载重但未及时回家，惩罚: -100.0")
            elif is_timeout:
                # 区分未处理和未采集的任务点
                num_unprocessed = np.sum(self.done_points == 0)  # 完全未处理的任务点
                num_uncollected = np.sum(self.successfully_collected_points == 0)  # 未成功采集的任务点
                
                if num_unprocessed > 0:
                    final_penalty = FINAL_PENALTY_UNVISITED * num_unprocessed
                    rewards += final_penalty
                    debug_print(f"回合超时，有 {num_unprocessed} 个任务点完全未处理，施加终局惩罚: {final_penalty}")
                
                if num_uncollected > num_unprocessed:
                    timeout_count = num_uncollected - num_unprocessed
                    debug_print(f"另有 {timeout_count} 个任务点因超时而未被采集")
                
                for i in range(self.num_agents):
                    if not np.array_equal(self.agent_positions[i], self.central_station):
                        rewards[i] += FINAL_PENALTY_NOT_AT_BASE
                        debug_print(f"警告：智能体 {i} 在回合超时时未返回基地，施加惩罚: {FINAL_PENALTY_NOT_AT_BASE}")

        # 更新上一时刻动作记录
        self.last_actions = list(actions)  # 使用解决冲突后的动作
        
        return self._get_obs(), rewards, done
    
    def get_collaboration_analytics(self):
        """
        获取协作分析统计数据
        """
        analytics = {}
        
        # 冲突率分析
        if self.total_decisions > 0:
            conflict_rate = self.conflict_count / self.total_decisions
        else:
            conflict_rate = 0.0
        
        analytics['conflict_rate'] = conflict_rate
        analytics['total_conflicts'] = self.conflict_count
        analytics['total_decisions'] = self.total_decisions
        
        # 角色专业化分析
        role_specialization = {}
        
        # 优先级任务分配分析
        for i in range(self.num_agents):
            agent_type = "fast" if i < 3 else "heavy"
            if self.agent_task_priorities[i]:
                high_priority_tasks = sum(1 for p in self.agent_task_priorities[i] if p == 1)
                total_tasks = len(self.agent_task_priorities[i])
                high_priority_ratio = high_priority_tasks / total_tasks if total_tasks > 0 else 0
            else:
                high_priority_ratio = 0.0
                total_tasks = 0
            
            role_specialization[f'agent_{i}'] = {
                'type': agent_type,
                'high_priority_ratio': high_priority_ratio,
                'total_tasks': total_tasks,
                'task_priorities': self.agent_task_priorities[i].copy()
            }
        
        # 载重效率分析
        load_efficiency = {}
        for i in range(self.num_agents):
            if self.agent_load_utilization[i]:
                avg_utilization = np.mean(self.agent_load_utilization[i])
                utilization_std = np.std(self.agent_load_utilization[i])
                empty_returns = sum(1 for u in self.agent_load_utilization[i] if u == 0)
                total_returns = len(self.agent_load_utilization[i])
                empty_return_rate = empty_returns / total_returns if total_returns > 0 else 0
            else:
                avg_utilization = 0.0
                utilization_std = 0.0
                empty_return_rate = 0.0
                total_returns = 0
            
            load_efficiency[f'agent_{i}'] = {
                'avg_utilization': avg_utilization,
                'utilization_std': utilization_std,
                'empty_return_rate': empty_return_rate,
                'total_returns': total_returns,
                'utilization_history': self.agent_load_utilization[i].copy()
            }
        
        analytics['role_specialization'] = role_specialization
        analytics['load_efficiency'] = load_efficiency
        
        return analytics
    
    def debug_print_collaboration_summary(self):
        """
        打印协作分析摘要
        """
        analytics = self.get_collaboration_analytics()
        
        debug_print("\n" + "=" * 60)
        debug_print("🤝 协作分析摘要")
        debug_print("=" * 60)
        
        # 冲突率分析
        debug_print(f"📊 冲突率分析:")
        debug_print(f"   总冲突次数: {analytics['total_conflicts']}")
        debug_print(f"   总决策次数: {analytics['total_decisions']}")
        debug_print(f"   冲突率: {analytics['conflict_rate']:.3f} ({analytics['conflict_rate']*100:.1f}%)")
        
        # 角色专业化分析
        debug_print(f"\n🎭 角色专业化分析:")
        fast_agents_high_priority = []
        heavy_agents_high_priority = []
        
        for agent_id in range(self.num_agents):
            agent_key = f'agent_{agent_id}'
            role_data = analytics['role_specialization'][agent_key]
            agent_type = "快速" if agent_id < 3 else "重载"
            
            if role_data['total_tasks'] > 0:
                debug_print(f"   智能体{agent_id} ({agent_type}): 高优先级任务比例 {role_data['high_priority_ratio']:.2f} ({role_data['total_tasks']}个任务)")
                
                if agent_id < 3:  # 快速智能体
                    fast_agents_high_priority.append(role_data['high_priority_ratio'])
                else:  # 重载智能体
                    heavy_agents_high_priority.append(role_data['high_priority_ratio'])
        
        # 专业化程度总结
        if fast_agents_high_priority and heavy_agents_high_priority:
            fast_avg = np.mean(fast_agents_high_priority)
            heavy_avg = np.mean(heavy_agents_high_priority)
            specialization_gap = fast_avg - heavy_avg
            debug_print(f"   快速智能体平均高优先级比例: {fast_avg:.3f}")
            debug_print(f"   重载智能体平均高优先级比例: {heavy_avg:.3f}")
            debug_print(f"   专业化差距: {specialization_gap:.3f} ({'良好' if specialization_gap > 0.1 else '需改进'})")
        
        # 载重效率分析
        debug_print(f"\n📦 载重效率分析:")
        for agent_id in range(self.num_agents):
            agent_key = f'agent_{agent_id}'
            load_data = analytics['load_efficiency'][agent_key]
            agent_type = "快速" if agent_id < 3 else "重载"
            
            if load_data['total_returns'] > 0:
                debug_print(f"   智能体{agent_id} ({agent_type}): 平均载重率 {load_data['avg_utilization']:.2f}, "
                      f"空载率 {load_data['empty_return_rate']:.2f} ({load_data['total_returns']}次返回)")
        
        # 载重效率总结
        heavy_utilizations = []
        fast_utilizations = []
        for agent_id in range(self.num_agents):
            agent_key = f'agent_{agent_id}'
            load_data = analytics['load_efficiency'][agent_key]
            if load_data['total_returns'] > 0:
                if agent_id < 3:
                    fast_utilizations.append(load_data['avg_utilization'])
                else:
                    heavy_utilizations.append(load_data['avg_utilization'])
        
        if fast_utilizations and heavy_utilizations:
            fast_avg_util = np.mean(fast_utilizations)
            heavy_avg_util = np.mean(heavy_utilizations)
            efficiency_gap = heavy_avg_util - fast_avg_util
            debug_print(f"   快速智能体平均载重率: {fast_avg_util:.3f}")
            debug_print(f"   重载智能体平均载重率: {heavy_avg_util:.3f}")
            debug_print(f"   载重效率差距: {efficiency_gap:.3f} ({'优秀' if efficiency_gap > 0.1 else '一般' if efficiency_gap > 0 else '需改进'})")
        
        debug_print("=" * 60)

    def _assign_task_responsibility(self, task_id, agent_id, current_step):
        """
        分配任务责任给智能体
        
        Args:
            task_id: 任务点ID
            agent_id: 智能体ID  
            current_step: 当前步数
        """
        # 记录当前负责该任务的智能体
        self.task_assignments[task_id] = agent_id
        self.task_assignment_timestamp[task_id] = current_step
        
        # 记录分配历史（用于调试和分析）
        if task_id not in self.task_assignment_history:
            self.task_assignment_history[task_id] = []
        self.task_assignment_history[task_id].append((agent_id, current_step))
    
    def _get_last_assigned_agent(self, point_id):
        """
        获取最后被分配给该任务的智能体
        
        优先返回任务分配记录，如果没有则返回实际访问者
        """
        # 优先使用任务责任分配记录
        if point_id in self.task_assignments:
            return self.task_assignments[point_id]
        
        # 如果没有分配记录，使用实际访问者记录（向后兼容）
        return self.point_last_visitor.get(point_id, None)
    
    def _calculate_task_assignment_score(self, agent_id, task_id):
        """
        计算智能体执行特定任务的综合评分
        
        Args:
            agent_id: 智能体ID
            task_id: 任务点ID
            
        Returns:
            score: 综合评分（越高越适合）
        """
        if task_id >= len(self.points) or self.done_points[task_id] == 1:
            return -float('inf')  # 无效或已完成的任务
            
        # 检查载重容量
        if self.agent_loads[agent_id] + self.samples[task_id] > self.agent_capacity[agent_id]:
            return -float('inf')  # 超载
            
        # 检查能量是否足够
        distance = self._distance(self.agent_positions[agent_id], self.points[task_id])
        energy_cost = distance * self.energy_cost_factor[agent_id]
        if self.agent_energy[agent_id] < energy_cost:
            return -float('inf')  # 能量不足
        
        score = 0.0
        
        # 1. 任务优先级权重 (0-100分)
        task_priority = self.priority[task_id]
        priority_score = (4 - task_priority) * 25  # 高优先级=75分，中=50分，低=25分
        
        # 2. 智能体类型匹配度 (0-50分)
        agent_type_score = 0
        is_fast_agent = agent_id < 3
        
        if task_priority == 1:  # 高优先级任务
            if is_fast_agent:
                agent_type_score = 50  # 快速智能体处理高优先级任务完美匹配
            else:
                agent_type_score = 10  # 重载智能体处理高优先级任务不匹配
        elif task_priority == 3:  # 低优先级任务
            task_load = self.samples[task_id]
            if not is_fast_agent and task_load >= 3:  # 重载智能体 + 大载重任务
                agent_type_score = 40
            elif is_fast_agent and task_load <= 2:  # 快速智能体 + 小载重任务
                agent_type_score = 30
            else:
                agent_type_score = 20
        else:  # 中优先级任务
            agent_type_score = 25  # 中等匹配
        
        # 3. 距离效率 (0-30分)
        # 归一化距离（假设最大距离为地图对角线）
        max_distance = np.sqrt(2) * self.size
        normalized_distance = min(distance / max_distance, 1.0)
        distance_score = (1.0 - normalized_distance) * 30
        
        # 4. 载重利用率 (0-20分)
        current_load_ratio = self.agent_loads[agent_id] / self.agent_capacity[agent_id]
        task_load_impact = self.samples[task_id] / self.agent_capacity[agent_id]
        final_load_ratio = current_load_ratio + task_load_impact
        
        # 奖励合理的载重利用率（60%-90%最优）
        if 0.6 <= final_load_ratio <= 0.9:
            load_score = 20
        elif 0.4 <= final_load_ratio < 0.6:
            load_score = 15
        elif 0.9 < final_load_ratio <= 1.0:
            load_score = 18
        else:
            load_score = 5
            
        # 5. 时间窗口紧迫度 (0-25分)
        time_remaining = max(self.time_windows[task_id], 0)
        if task_priority == 1:  # 高优先级
            max_time = 3 * 60
        elif task_priority == 2:  # 中优先级
            max_time = 10 * 60
        else:  # 低优先级
            max_time = 30 * 60
            
        time_urgency = 1.0 - min(time_remaining / max_time, 1.0)
        urgency_score = time_urgency * 25
        
        # 6. 能量效率加分 (0-10分)
        energy_ratio = self.agent_energy[agent_id] / self.agent_energy_max[agent_id]
        energy_score = energy_ratio * 10
        
        # 总分计算（使用配置权重）
        score = (priority_score * config.ICR_PRIORITY_WEIGHT + 
                agent_type_score * config.ICR_AGENT_TYPE_WEIGHT + 
                distance_score * config.ICR_DISTANCE_WEIGHT + 
                load_score * config.ICR_LOAD_WEIGHT + 
                urgency_score * config.ICR_URGENCY_WEIGHT + 
                energy_score * config.ICR_ENERGY_WEIGHT)
        
        return score
    
    def _resolve_conflict_intelligently(self, task_id, competing_agents):
        """
        智能解决冲突：基于综合评分选择最适合的智能体
        
        Args:
            task_id: 冲突的任务点ID
            competing_agents: 竞争的智能体ID列表
            
        Returns:
            winner_agent_id: 获胜的智能体ID
        """
        best_score = -float('inf')
        best_agent = competing_agents[0]  # 默认第一个
        
        task_priority = self.priority[task_id]
        task_load = self.samples[task_id]
        
        debug_print(f"🔍 任务点 {task_id} 冲突解决 (优先级: {task_priority}, 载重: {task_load}):")
        
        for agent_id in competing_agents:
            score = self._calculate_task_assignment_score(agent_id, task_id)
            agent_type = "快速" if agent_id < 3 else "重载"
            distance = self._distance(self.agent_positions[agent_id], self.points[task_id])
            
            debug_print(f"   智能体 {agent_id} ({agent_type}): 得分 {score:.1f}, 距离 {distance:.1f}")
            
            if score > best_score:
                best_score = score
                best_agent = agent_id
        
        debug_print(f"   🏆 获胜者: 智能体 {best_agent} (得分: {best_score:.1f})")
        return best_agent
    
    def _find_best_alternative_task_intelligent(self, agent_id, excluded_tasks):
        """
        智能寻找替代任务：基于智能体能力和任务匹配度
        
        Args:
            agent_id: 智能体ID
            excluded_tasks: 已分配的任务集合
            
        Returns:
            best_task_id: 最佳替代任务ID，如果没有则返回None
        """
        available_tasks = []
        
        # 获取所有可用任务
        for task_id in range(self.num_points):
            if (task_id not in excluded_tasks and 
                self.done_points[task_id] == 0 and
                self.agent_loads[agent_id] + self.samples[task_id] <= self.agent_capacity[agent_id]):
                
                # 检查能量
                distance = self._distance(self.agent_positions[agent_id], self.points[task_id])
                energy_cost = distance * self.energy_cost_factor[agent_id]
                if self.agent_energy[agent_id] >= energy_cost:
                    available_tasks.append(task_id)
        
        if not available_tasks:
            return None
        
        # 计算每个可用任务的评分
        best_score = -float('inf')
        best_task = None
        
        agent_type = "快速" if agent_id < 3 else "重载"
        debug_print(f"   🔄 为 {agent_type}智能体 {agent_id} 寻找替代任务:")
        
        for task_id in available_tasks:
            score = self._calculate_task_assignment_score(agent_id, task_id)
            priority = self.priority[task_id]
            load = self.samples[task_id]
            
            debug_print(f"      任务 {task_id} (P{priority}, 载重{load}): 得分 {score:.1f}")
            
            if score > best_score:
                best_score = score
                best_task = task_id
        
        if best_task is not None:
            debug_print(f"      ✅ 最佳替代: 任务 {best_task} (得分: {best_score:.1f})")
        
        return best_task
    
    def _get_available_tasks_for_agent(self, agent_id):
        """
        获取智能体可执行的任务点列表
        """
        available_tasks = []
        for p_idx in range(self.num_points):
            if (self.done_points[p_idx] == 0 and 
                self.agent_loads[agent_id] + self.samples[p_idx] <= self.agent_capacity[agent_id]):
                available_tasks.append(p_idx)
        return available_tasks
    
    # 原有的基于距离的替代任务查找已被智能版本替代
    # 保留作为后备方案
    def _find_best_alternative_task_simple(self, agent_id, excluded_tasks=None):
        """
        简单的基于距离的替代任务查找（后备方案）
        """
        if excluded_tasks is None:
            excluded_tasks = set()
            
        available_tasks = self._get_available_tasks_for_agent(agent_id)
        
        # 排除已被其他智能体选择的任务
        candidate_tasks = [t for t in available_tasks if t not in excluded_tasks]
        
        if not candidate_tasks:
            return None
        
        # 计算到各个候选任务的距离，选择最近的
        distances = [self._distance(self.agent_positions[agent_id], self.points[t]) for t in candidate_tasks]
        best_task_idx = np.argmin(distances)
        return candidate_tasks[best_task_idx]
    
    def _get_dynamic_load_threshold(self, agent_id, current_step, max_steps):
        """
        根据任务点稀少程度和游戏进度动态调整载重阈值
        """
        # 计算剩余任务点数量
        remaining_tasks = np.sum(self.done_points == 0)
        total_tasks = len(self.done_points)
        task_scarcity = 1.0 - (remaining_tasks / total_tasks)  # 任务稀缺度 0-1
        
        # 计算游戏进度
        game_progress = current_step / max_steps  # 游戏进度 0-1
        
        # 基础载重阈值
        base_threshold = 0.9 if agent_id < 3 else 0.8  # 快速智能体vs重载智能体
        
        # 根据任务稀缺度调整：任务越少，阈值越低
        scarcity_adjustment = task_scarcity * 0.4  # 最多降低40%
        
        # 根据游戏后期调整：后期降低要求
        if game_progress > 0.7:
            late_game_adjustment = (game_progress - 0.7) * 0.3  # 后期额外降低30%
        else:
            late_game_adjustment = 0
        
        # 最终阈值
        final_threshold = base_threshold - scarcity_adjustment - late_game_adjustment
        
        # 确保阈值在合理范围内
        final_threshold = max(0.3, min(0.9, final_threshold))  # 限制在30%-90%之间
        
        return final_threshold

    def _update_traffic(self):
        for _ in range(random.randint(2, 5)):
            x = random.randint(0, self.traffic_map.shape[0] - 1)
            y = random.randint(0, self.traffic_map.shape[1] - 1)
            self.traffic_map[x, y] = random.choice([0.5, 0.2])

    def _spawn_new_point(self):
        MAX_NUM_NEW_POINTS = 3
        if self.new_points_num >= MAX_NUM_NEW_POINTS:
            return
        
        if not self.queue_done.empty():
            target_idx = self.queue_done.get()
            new_point = np.random.randint(0, self.size, (1, 2))
            new_sample = np.random.randint(1, 6)
            new_priority = 1 if random.random() < 0.3 else 2
            new_time_window = (0.5 if new_priority == 1 else 2) * 60

            self.points[target_idx] = new_point[0]
            self.samples[target_idx] = new_sample
            self.priority[target_idx] = new_priority
            self.time_windows[target_idx] = new_time_window
            self.done_points[target_idx] = 0
            self.task_creation_time[target_idx] = self.time
            self.new_points_num += 1

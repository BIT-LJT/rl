import numpy as np
import random
import queue

class SamplingEnv2_0:
    def __init__(self, num_points=30, num_agents=5):
        self.size = 5000
        self.num_points = num_points
        self.num_agents = num_agents
        self.action_dim = self.num_points + 2

        self.agent_capacity = np.array([10, 10, 10, 20, 20])
        self.agent_speed = np.array([15, 15, 15, 10, 10])
        self.agent_energy_max = np.array([300000, 300000, 300000, 450000, 450000])

        BASE_SPEED = np.max(self.agent_speed)
        self.energy_cost_factor = BASE_SPEED / self.agent_speed

        self.fast_charge_time = 10
        self.slow_charge_time = 30 * 60

        self.queue_done = queue.Queue()
        self.queue_new_point = queue.Queue()

        self.reset()

    def reset(self):
        self.points = np.random.randint(0, self.size, (self.num_points, 2))
        self.samples = np.random.randint(1, 6, self.num_points)
        self.priority = np.random.choice([1, 2, 3], self.num_points, p=[0.3, 0.4, 0.3])
        self.time_windows = np.array([(1*60*60) if p == 1 else ((3*60*60) if p == 2 else (6*60*60)) for p in self.priority], dtype=float)
        
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
        
        # 新增: 用于存储每个样本的送达信息
        self.delivery_info = []

        self.central_station = np.array([self.size // 2, self.size // 2], dtype=float)

        self.time = 0
        self.spawn_count = 0
        self.new_points_num = 0
        self.point_last_visitor = {}
        self.agent_paths = [[] for _ in range(self.num_agents)]
        self.task_creation_time = np.full(self.num_points, self.time)

        return self._get_obs()

    def _get_action_masks(self):
        masks = np.ones((self.num_agents, self.action_dim), dtype=bool)
        for i in range(self.num_agents):
            if self.agent_charging_status[i] > 0:
                masks[i, :] = False
                masks[i, self.num_points] = True
                continue

            for p_idx in range(self.num_points):
                if self.done_points[p_idx] == 1:
                    masks[i, p_idx] = False
                    continue
                if self.agent_loads[i] + self.samples[p_idx] > self.agent_capacity[i]:
                    masks[i, p_idx] = False
                    continue
        return masks

    def _get_obs(self):
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
            "action_masks": self._get_action_masks()
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
        
        R_COLLECT_BASE = 50.0
        R_RETURN_HOME_COEFF = 0.5
        R_FULL_LOAD_BONUS = 300.0
        R_COEFF_APPROACH_TASK = 0.1
        R_COEFF_APPROACH_HOME_LOADED = 0.05
        R_COEFF_APPROACH_HOME_LOW_ENERGY = 0.1
        REWARD_SHAPING_CLIP = 10.0
        R_COLLAB_HIGH_PRIORITY = 20.0
        R_ROLE_SPEED_BONUS = 20.0
        R_ROLE_CAPACITY_BONUS = 50.0
        FINAL_BONUS_ACCOMPLISHED = 1000.0

        P_TIME_STEP = -0.1
        P_INACTIVITY = -10.0
        P_ENERGY_DEPLETED = -50.0
        P_TIMEOUT_BASE = -30.0
        P_CONFLICT = -2.0
        FINAL_PENALTY_UNVISITED = -50.0
        FINAL_PENALTY_NOT_AT_BASE = -100.0

        rewards += P_TIME_STEP
        
        for i in range(self.num_agents):
            if self.agent_charging_status[i] > 0:
                self.agent_charging_status[i] -= 1
                if self.agent_charging_status[i] == 0:
                    self.agent_energy[i] = self.agent_energy_max[i]

        task_targets = actions[actions < self.num_points]
        unique_targets, counts = np.unique(task_targets, return_counts=True)
        conflicted_targets = unique_targets[counts > 1]

        for target_idx in conflicted_targets:
            competing_agent_indices = np.where(actions == target_idx)[0]
            distances = [self._distance(self.agent_positions[i], self.points[target_idx]) for i in competing_agent_indices]
            winner_local_idx = np.argmin(distances)
            winner_global_idx = competing_agent_indices[winner_local_idx]
            for i, agent_idx in enumerate(competing_agent_indices):
                if agent_idx != winner_global_idx:
                    actions[agent_idx] = self.num_points 
                    rewards[agent_idx] += P_CONFLICT
        
        for i, target in enumerate(actions):
            if self.agent_charging_status[i] > 0:
                continue

            prev_pos = self.agent_positions[i].copy()
            self.agent_paths[i].append(prev_pos)

            if target == self.num_points + 1:
                dist = self._distance(prev_pos, self.central_station)
                energy_cost = dist * self.energy_cost_factor[i]
                if self.agent_energy[i] < energy_cost:
                    rewards[i] += P_ENERGY_DEPLETED
                    continue
                self.agent_positions[i] = self.central_station
                self.agent_energy[i] -= energy_cost
                if self.agent_loads[i] > 0:
                    for point_idx in self.agent_samples[i]:
                        remaining_time = self.time_windows[point_idx]
                        self.delivery_info.append((point_idx, remaining_time, i))
                        print(f"信息：智能体 {i} 送回样本点 {point_idx}，剩余时间窗口: {remaining_time:.2f} 秒")

                    rewards[i] += R_RETURN_HOME_COEFF * self.agent_loads[i]
                    if self.agent_loads[i] >= self.agent_capacity[i] * 0.8:
                        rewards[i] += R_FULL_LOAD_BONUS
                    if i in [3, 4] and self.agent_loads[i] > self.agent_capacity[i] * 0.8:
                        rewards[i] += R_ROLE_CAPACITY_BONUS
                    self.agent_loads[i] = 0
                    self.agent_samples[i] = []
                    self.agent_task_status[i] = 0
                
            elif target == self.num_points:
                is_at_base = np.array_equal(self.agent_positions[i], self.central_station)
                needs_charge = self.agent_energy[i] < self.agent_energy_max[i]
                if is_at_base and needs_charge:
                    if i < 3:
                        self.agent_charging_status[i] = self.fast_charge_time
                    else:
                        self.agent_charging_status[i] = self.slow_charge_time
                    self.agent_charge_counts[i] += 1
                else:
                    rewards[i] += P_INACTIVITY
                continue

            elif target < self.num_points:
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
                priority = self.priority[target]
                rewards[i] += R_COLLECT_BASE * (4 - priority)
                if priority == 1:
                    for j in range(self.num_agents):
                        if i != j: rewards[j] += R_COLLAB_HIGH_PRIORITY
                    if i in [0, 1, 2]:
                        time_spent = self.time - self.task_creation_time[target]
                        time_window = 1 * 60
                        if time_spent < time_window:
                             rewards[i] += R_ROLE_SPEED_BONUS * (1 - time_spent / time_window)

            current_pos = self.agent_positions[i]
            if target < self.num_points:
                approach_reward = R_COEFF_APPROACH_TASK * (self._distance(prev_pos, self.points[target]) - self._distance(current_pos, self.points[target]))
                rewards[i] += np.clip(approach_reward, -REWARD_SHAPING_CLIP, REWARD_SHAPING_CLIP)
            
            approach_home_delta = self._distance(prev_pos, self.central_station) - self._distance(current_pos, self.central_station)
            if self.agent_loads[i] >= self.agent_capacity[i] * 0.9 and self.agent_loads[i] > 0:
                rewards[i] += np.clip(R_COEFF_APPROACH_HOME_LOADED * self.agent_loads[i] * approach_home_delta, -REWARD_SHAPING_CLIP, REWARD_SHAPING_CLIP)
            if self.agent_energy[i] < self.agent_energy_max[i] * 0.1:
                rewards[i] += np.clip(R_COEFF_APPROACH_HOME_LOW_ENERGY * approach_home_delta, -REWARD_SHAPING_CLIP, REWARD_SHAPING_CLIP)

        self.time += 1
        self.time_windows -= 1
        
        for i in range(self.num_points):
            if self.done_points[i] == 0 and self.time_windows[i] < 0:
                assigned_agent = self._get_last_assigned_agent(i)
                if assigned_agent is not None:
                    rewards[assigned_agent] += P_TIMEOUT_BASE * (4 - self.priority[i])
                self.done_points[i] = 1

        all_tasks_collected = np.all(self.successfully_collected_points == 1)
        all_agents_at_base = all(np.array_equal(pos, self.central_station) for pos in self.agent_positions)
        mission_accomplished = all_tasks_collected and all_agents_at_base
        is_timeout = current_step >= max_steps - 1
        done = mission_accomplished or is_timeout

        if done:
            if mission_accomplished:
                rewards += FINAL_BONUS_ACCOMPLISHED
                print(f"恭喜！所有任务点均已完成且所有智能体已返回基地，获得通关奖励: {FINAL_BONUS_ACCOMPLISHED}")
            elif is_timeout:
                num_unvisited = np.sum(self.successfully_collected_points == 0)
                if num_unvisited > 0:
                    final_penalty = FINAL_PENALTY_UNVISITED * num_unvisited
                    rewards += final_penalty
                    print(f"回合超时，且有 {num_unvisited} 个点未访问，施加终局惩罚: {final_penalty}")
                
                for i in range(self.num_agents):
                    if not np.array_equal(self.agent_positions[i], self.central_station):
                        rewards[i] += FINAL_PENALTY_NOT_AT_BASE
                        print(f"警告：智能体 {i} 在回合超时时未返回基地，施加惩罚: {FINAL_PENALTY_NOT_AT_BASE}")

        return self._get_obs(), rewards, done

    def _get_last_assigned_agent(self, point_id):
        return self.point_last_visitor.get(point_id, None)

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

# 📊 TensorBoard训练监控指南

## 🚀 快速开始

### 1. 启动训练
```bash
python main.py
```

### 2. 启动TensorBoard
在另一个终端中运行：
```bash
tensorboard --logdir=runs --port=6006
```

### 3. 查看训练情况
在浏览器中打开：`http://localhost:6006`

---

## 📈 主要监控指标

### 🎯 **训练核心指标 (Training)**

| 指标名称 | 说明 | 理想趋势 |
|---------|------|----------|
| `Total_Reward_Sum` | 所有智能体总奖励 | 📈 持续上升 |
| `Average_Reward` | 平均奖励 | 📈 逐步提升 |
| `Noise_Std` | 探索噪声标准差 | 📉 逐渐衰减 |
| `Task_Completion_Rate` | 任务完成率 | 📈 趋向100% |

### 🤖 **智能体表现 (Agent_Rewards & Agent_Performance)**

| 指标类型 | 说明 | 监控要点 |
|---------|------|----------|
| `Agent_{i}_fast` | 快速智能体奖励 | 应专注高优先级任务 |
| `Agent_{i}_heavy` | 重载智能体奖励 | 应通过满载运输获得高奖励 |
| `Charge_Count_Agent_{i}` | 充电次数 | 过多表示能量管理问题 |

### 🎭 **角色专业化 (Role_Specialization & Role_Analysis)**

| 指标名称 | 说明 | 成功标准 |
|---------|------|----------|
| `Fast_Agents_Avg_Reward` | 快速智能体平均奖励 | 稳定且合理 |
| `Heavy_Agents_Avg_Reward` | 重载智能体平均奖励 | 应≥快速智能体 |
| `Reward_Specialization_Gap` | 奖励专业化差距 | >0 表示重载智能体优势发挥 |
| `Specialization_Gap` | 高优先级任务分工差距 | >0.1 表示角色分工良好 |

### 🤝 **协作效率 (Collaboration)**

| 指标名称 | 说明 | 优化目标 |
|---------|------|----------|
| `Conflict_Rate` | 冲突发生率 | 📉 随训练减少 |
| `Total_Conflicts` | 累计冲突次数 | 监控协作改善 |
| `Total_Decisions` | 累计决策次数 | 了解决策频率 |

### 📦 **载重效率 (Load_Efficiency & Load_Analysis)**

| 指标名称 | 说明 | 期望值 |
|---------|------|--------|
| `Agent_{i}_Avg_Utilization` | 平均载重利用率 | 重载智能体 > 快速智能体 |
| `Agent_{i}_Empty_Return_Rate` | 空载返回率 | 📉 越低越好 |
| `Efficiency_Gap` | 载重效率差距 | >0.1 表示重载智能体发挥优势 |

### 🔧 **训练诊断 (Loss & Gradients)**

| 指标名称 | 说明 | 健康状态 |
|---------|------|----------|
| `Critic_Loss` | Q网络损失 | 📉 逐步收敛 |
| `Actor_Loss` | 策略网络损失 | 平稳波动 |
| `Critic_Grad_Norm` | Critic梯度范数 | <1.0 (梯度裁剪后) |
| `Actor_Grad_Norm` | Actor梯度范数 | <1.0 (梯度裁剪后) |

### ⏱️ **延迟策略更新 (Delayed Policy Updates)**

| 指标名称 | 说明 | 监控要点 |
|---------|------|----------|
| `Update_Counter` | 总更新次数 | 跟踪训练进度 |
| `Actor_Update_Ratio` | Actor更新比例 | 应为 1/ACTOR_UPDATE_FREQUENCY |

### 📚 **学习率监控 (Learning_Rate)**

| 指标名称 | 说明 | 预期行为 |
|---------|------|----------|
| `Actor_LR` | Actor学习率 | 按调度策略变化 |
| `Critic_LR` | Critic学习率 | 按调度策略变化 |

---

## 🔍 如何解读训练进展

### ✅ **训练正常的信号**

1. **奖励曲线稳步上升**
   - `Total_Reward_Sum` 呈现上升趋势
   - `Average_Reward` 逐步提升

2. **角色专业化形成**
   - `Specialization_Gap` > 0.1（快速智能体更多处理高优先级任务）
   - `Efficiency_Gap` > 0.1（重载智能体载重效率更高）

3. **协作能力提升**
   - `Conflict_Rate` 逐步下降
   - 智能体学会主动避免冲突

4. **载重优化成功**
   - 重载智能体的 `Avg_Utilization` 显著高于快速智能体
   - `Empty_Return_Rate` 下降

### ⚠️ **需要关注的警告信号**

1. **奖励停滞或下降**
   - 检查学习率是否过高/过低
   - 观察梯度范数是否异常
   - 考虑启用延迟策略更新

2. **角色专业化失败**
   - `Specialization_Gap` ≤ 0：快速智能体没有专注高优任务
   - `Efficiency_Gap` ≤ 0：重载智能体没有发挥载重优势

3. **梯度问题**
   - `Grad_Norm` 接近0：梯度消失
   - `Grad_Norm` 持续为1.0：梯度爆炸（被裁剪）

4. **协作问题**
   - `Conflict_Rate` 居高不下：智能体没有学会协作

5. **延迟策略更新问题** (启用时)
   - `Critic_Loss` 仍然上升：考虑增加ACTOR_UPDATE_FREQUENCY
   - `Actor_Update_Ratio` 不正确：检查配置参数

### 🚀 **延迟策略更新优势 (TD3风格)**

当启用延迟策略更新时，您将获得：

#### ✅ **训练稳定性提升**
- **解决Critic Loss上升**：Critic有更多时间"追赶"Actor的变化
- **更平滑的损失曲线**：减少Actor-Critic学习速度不匹配问题
- **更稳健的策略**：Actor基于更准确的Q值估计进行更新

#### 📊 **监控关键指标**
- **`Training/Actor_Update_Ratio`**: 应稳定在 `1/ACTOR_UPDATE_FREQUENCY`
- **`Loss/Critic_Loss`**: 应比传统方法更快收敛
- **`Loss/Actor_Loss`**: 更新频率较低但更稳定

#### ⚙️ **参数调优建议**
```python
# config.py
ACTOR_UPDATE_FREQUENCY = 2  # 推荐起始值：2-3
# 如果Critic_Loss仍然上升，可以增加到3或4
# 如果训练太慢，可以减少到1（回到传统方法）
```

---

## 🛠️ 配置选项

在 `config.py` 中可以调整TensorBoard行为：

```python
# TensorBoard Logging Settings
ENABLE_TENSORBOARD = True  # 是否启用TensorBoard记录
TENSORBOARD_LOG_INTERVAL = 1  # 每多少个episode记录一次基础指标
TENSORBOARD_DIAGNOSTIC_INTERVAL = 100  # 每多少次更新记录一次诊断信息
TENSORBOARD_COLLABORATION_INTERVAL = 100  # 每多少个episode记录一次协作分析
```

---

## 📊 关键图表解读

### 1. **Role_Specialization/Reward_Specialization_Gap**
- **理想状态**: 正值且逐步增加
- **含义**: 重载智能体通过专业化获得更高奖励

### 2. **Load_Analysis/Efficiency_Gap** 
- **理想状态**: 正值且稳定在0.1以上
- **含义**: 重载智能体载重效率明显优于快速智能体

### 3. **Collaboration/Conflict_Rate**
- **理想状态**: 从高值逐步下降并稳定在低值
- **含义**: 智能体学会了主动协作和冲突规避

### 4. **Training/Task_Completion_Rate**
- **理想状态**: 尽快达到100%并保持稳定
- **含义**: 智能体能够高效完成所有任务

---

## 🎯 实用技巧

### 📌 **快速诊断问题**
1. 先看 `Training/Total_Reward_Sum` - 整体趋势
2. 再看 `Role_Analysis/Specialization_Gap` - 角色分工
3. 最后看 `Loss/` 相关指标 - 训练稳定性

### 📌 **对比不同实验**
- TensorBoard会自动为每次训练创建带时间戳的目录
- 可以同时查看多个实验的曲线进行对比

### 📌 **实时监控**
- TensorBoard会自动刷新数据
- 训练过程中可以实时观察指标变化

---

## 🎊 恭喜！

您现在拥有了一个功能完善的训练监控系统，可以深度洞察MADDPG智能体的学习进展和协作演化过程！

"""
增量式奖励实验配置系统

这个模块允许您逐步开启不同的奖励组件，系统地分析每个奖励模块对协作策略的影响。

使用方法：
1. 从 BASIC 开始训练，确保基础收敛
2. 逐步升级到更复杂的配置 
3. 观察每个新增模块对协作指标的影响

实验流程建议：
BASIC → LOAD_EFFICIENCY → ROLE_SPECIALIZATION → COLLABORATION → BEHAVIOR_SHAPING → FULL
"""

class RewardExperimentConfig:
    """增量式奖励实验配置"""
    
    # 实验配置等级定义
    BASIC = "basic"                    # 基础：只有采集奖励
    LOAD_EFFICIENCY = "load_efficiency"       # 载重效率：+满载奖励
    ROLE_SPECIALIZATION = "role_specialization"   # 角色专业化：+快速/重载智能体专业化
    COLLABORATION = "collaboration"           # 协作：+冲突解决和协作奖励
    BEHAVIOR_SHAPING = "behavior_shaping"     # 行为塑造：+接近奖励和智能行为
    FULL = "full"                     # 完整版：所有奖励模块
    
    def __init__(self, experiment_level=BASIC):
        """
        初始化奖励配置
        
        Args:
            experiment_level: 实验等级，从BASIC到FULL
        """
        self.experiment_level = experiment_level
        self._setup_reward_modules()
    
    def _setup_reward_modules(self):
        """根据实验等级设置启用的奖励模块"""
        
        # 所有奖励模块的开关
        self.enable_basic_rewards = True  # 基础奖励：永远启用
        self.enable_load_efficiency = False
        self.enable_role_specialization = False  
        self.enable_collaboration = False
        self.enable_behavior_shaping = False
        self.enable_advanced_penalties = False
        
        # 根据实验等级逐步启用模块
        if self.experiment_level in [self.LOAD_EFFICIENCY, self.ROLE_SPECIALIZATION, 
                                    self.COLLABORATION, self.BEHAVIOR_SHAPING, self.FULL]:
            self.enable_load_efficiency = True
            
        if self.experiment_level in [self.ROLE_SPECIALIZATION, self.COLLABORATION, 
                                    self.BEHAVIOR_SHAPING, self.FULL]:
            self.enable_role_specialization = True
            
        if self.experiment_level in [self.COLLABORATION, self.BEHAVIOR_SHAPING, self.FULL]:
            self.enable_collaboration = True
            
        if self.experiment_level in [self.BEHAVIOR_SHAPING, self.FULL]:
            self.enable_behavior_shaping = True
            
        if self.experiment_level == self.FULL:
            self.enable_advanced_penalties = True
    
    def get_reward_constants(self):
        """获取当前实验等级对应的奖励常量"""
        
        # 基础奖励常量（永远启用）
        constants = {
            # === 基础奖励模块 ===
            'R_COLLECT_BASE': 50.0,           # 采集基础奖励
            'R_RETURN_HOME_COEFF': 0.5,       # 回家奖励系数
            'P_TIME_STEP': -0.1,              # 时间步惩罚
            'P_ENERGY_DEPLETED': -50.0,       # 能量不足惩罚
            'FINAL_BONUS_ACCOMPLISHED': 1000.0, # 最终完成奖励
            
            # === 载重效率模块 ===
            'R_FULL_LOAD_BONUS': 300.0 if self.enable_load_efficiency else 0.0,
            'R_ROLE_CAPACITY_BONUS': 50.0 if self.enable_load_efficiency else 0.0,
            'P_EMPTY_RETURN': -20.0 if self.enable_load_efficiency else 0.0,
            'P_LOW_LOAD_HEAVY_AGENT': -15.0 if self.enable_load_efficiency else 0.0,
            
            # === 角色专业化模块 ===
            'R_FAST_AGENT_HIGH_PRIORITY': 30.0 if self.enable_role_specialization else 0.0,
            'R_ROLE_SPEED_BONUS': 20.0 if self.enable_role_specialization else 0.0,
            'P_HEAVY_AGENT_HIGH_PRIORITY': -15.0 if self.enable_role_specialization else 0.0,
            
            # === 协作模块 ===
            'R_COLLAB_HIGH_PRIORITY': 20.0 if self.enable_collaboration else 0.0,
            'P_CONFLICT': -2.0 if self.enable_collaboration else 0.0,
            
            # === 行为塑造模块 ===
            'R_COEFF_APPROACH_TASK': 0.1 if self.enable_behavior_shaping else 0.0,
            'R_COEFF_APPROACH_HOME_LOADED': 0.05 if self.enable_behavior_shaping else 0.0,
            'R_COEFF_APPROACH_HOME_LOW_ENERGY': 0.1 if self.enable_behavior_shaping else 0.0,
            'R_STAY_HOME_AFTER_COMPLETION': 10.0 if self.enable_behavior_shaping else 0.0,
            'R_SMART_RETURN_AFTER_COMPLETION': 50.0 if self.enable_behavior_shaping else 0.0,
            'REWARD_SHAPING_CLIP': 10.0 if self.enable_behavior_shaping else 0.0,
            
            # === 高级惩罚模块 ===
            'P_INVALID_TASK_ATTEMPT': -100.0 if self.enable_advanced_penalties else -10.0,
            'P_OVERLOAD_ATTEMPT': -80.0 if self.enable_advanced_penalties else -10.0,
            'P_TIMEOUT_BASE': -30.0 if self.enable_advanced_penalties else -5.0,
            'P_INACTIVITY': -10.0 if self.enable_advanced_penalties else 0.0,
            'P_NOT_RETURN_AFTER_COMPLETION': -20.0 if self.enable_advanced_penalties else 0.0,
            'P_POINTLESS_ACTION': -20.0 if self.enable_advanced_penalties else 0.0,
        }
        
        return constants
    
    def get_experiment_description(self):
        """获取当前实验等级的描述"""
        descriptions = {
            self.BASIC: "基础实验：只有采集奖励和基础惩罚，验证智能体能否学会基本任务采集",
            self.LOAD_EFFICIENCY: "载重效率实验：+满载奖励和空载惩罚，学习高效载重策略",
            self.ROLE_SPECIALIZATION: "角色专业化实验：+快速/重载智能体专业化奖励，学习分工协作",
            self.COLLABORATION: "协作实验：+冲突解决和协作奖励，学习避免冲突和协作策略",
            self.BEHAVIOR_SHAPING: "行为塑造实验：+接近奖励和智能行为奖励，学习精细化行为模式",
            self.FULL: "完整实验：所有奖励模块，学习最复杂的协作策略"
        }
        return descriptions.get(self.experiment_level, "未知实验等级")
    
    def get_expected_behaviors(self):
        """获取当前等级预期学会的行为"""
        behaviors = {
            self.BASIC: [
                "基础任务采集",
                "能量管理（避免能量耗尽）",
                "基础返回基地行为"
            ],
            self.LOAD_EFFICIENCY: [
                "优化载重利用率",
                "避免空载返回",
                "满载时主动返回基地"
            ],
            self.ROLE_SPECIALIZATION: [
                "快速智能体优先处理高优先级任务",
                "重载智能体专注处理大载重任务",
                "根据智能体类型选择合适任务"
            ],
            self.COLLABORATION: [
                "避免任务冲突",
                "智能冲突解决",
                "协作处理高优先级任务"
            ],
            self.BEHAVIOR_SHAPING: [
                "路径优化（接近奖励引导）",
                "任务完成后的智能行为选择",
                "精细化的状态感知行为"
            ],
            self.FULL: [
                "复杂的多智能体协作策略",
                "高级的任务分配优化",
                "全面的效率和协作平衡"
            ]
        }
        return behaviors.get(self.experiment_level, [])
    
    def get_key_metrics(self):
        """获取当前等级的关键评估指标"""
        metrics = {
            self.BASIC: [
                "任务完成率",
                "平均episode长度", 
                "能量利用效率"
            ],
            self.LOAD_EFFICIENCY: [
                "载重利用率",
                "空载返回次数",
                "满载奖励获得次数"
            ],
            self.ROLE_SPECIALIZATION: [
                "快速智能体高优先级任务比例",
                "重载智能体载重效率",
                "角色专业化差距"
            ],
            self.COLLABORATION: [
                "冲突率",
                "冲突解决成功率",
                "协作任务完成效率"
            ],
            self.BEHAVIOR_SHAPING: [
                "路径效率",
                "智能行为奖励获得情况",
                "精细化行为模式评分"
            ],
            self.FULL: [
                "综合协作效率",
                "多维度平衡评分",
                "复杂策略学习成功率"
            ]
        }
        return metrics.get(self.experiment_level, [])

# 预定义的实验配置实例
EXPERIMENT_CONFIGS = {
    RewardExperimentConfig.BASIC: RewardExperimentConfig(RewardExperimentConfig.BASIC),
    RewardExperimentConfig.LOAD_EFFICIENCY: RewardExperimentConfig(RewardExperimentConfig.LOAD_EFFICIENCY),
    RewardExperimentConfig.ROLE_SPECIALIZATION: RewardExperimentConfig(RewardExperimentConfig.ROLE_SPECIALIZATION),
    RewardExperimentConfig.COLLABORATION: RewardExperimentConfig(RewardExperimentConfig.COLLABORATION),
    RewardExperimentConfig.BEHAVIOR_SHAPING: RewardExperimentConfig(RewardExperimentConfig.BEHAVIOR_SHAPING),
    RewardExperimentConfig.FULL: RewardExperimentConfig(RewardExperimentConfig.FULL),
}

def print_experiment_guide():
    """打印实验指导"""
    from utils import debug_print
    
    debug_print("\n" + "🧪" * 60)
    debug_print("增量式奖励实验指导")
    debug_print("🧪" * 60)
    
    for level in [RewardExperimentConfig.BASIC, RewardExperimentConfig.LOAD_EFFICIENCY,
                  RewardExperimentConfig.ROLE_SPECIALIZATION, RewardExperimentConfig.COLLABORATION,
                  RewardExperimentConfig.BEHAVIOR_SHAPING, RewardExperimentConfig.FULL]:
        
        config = EXPERIMENT_CONFIGS[level]
        debug_print(f"\n📊 实验等级: {level.upper()}")
        debug_print(f"   描述: {config.get_experiment_description()}")
        debug_print(f"   预期行为: {', '.join(config.get_expected_behaviors())}")
        debug_print(f"   关键指标: {', '.join(config.get_key_metrics())}")
    
    debug_print("\n💡 实验建议:")
    debug_print("   1. 按顺序进行实验，确保每个等级都能收敛")
    debug_print("   2. 记录每个等级的关键指标变化")
    debug_print("   3. 对比不同等级的协作策略差异")
    debug_print("   4. 分析奖励模块对最终性能的贡献")
    debug_print("🧪" * 60)

if __name__ == "__main__":
    # 演示用法
    print_experiment_guide()

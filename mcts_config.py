#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCTS配置文件
集中管理所有MCTS相关的参数配置
"""

from typing import Dict, Any


class MCTSConfig:
    """MCTS配置类"""
    
    # 默认配置
    DEFAULT_CONFIG = {
        # 核心MCTS参数
        'c_puct': 0.8,              # UCB探索系数，调优最优值
        'max_simulations': 80,       # 最大模拟次数，调优最优值
        'max_depth': 12,             # 最大搜索深度，调优最优值
        'simulation_timeout': 20,    # 模拟超时时间（秒），调优最优值
        
        # 奖励函数权重(調參)
        'reward_weights': {
            'completeness': 0.4,     # 信息完整性权重
            'path_quality': 0.25,    # 推理路径质量权重
            'answer_relevance': 0.2, # 答案相关性权重
            'efficiency': 0.1,       # 效率权重
            'diversity': 0.05        # 多样性权重
        },
        
        # 动作评分权重
        'action_scores': {
            'search_base': 0.6,           # 搜索动作基础分数
            'neighbors_base': 0.5,        # 邻居动作基础分数
            'shared_neighbors_base': 0.4, # 共同邻居动作基础分数
            'answer_base': 0.0           # 回答动作基础分数
        },
        
        # 模拟策略参数
        'simulation': {
            'temperature_base': 1.0,     # 基础温度
            'min_temperature': 0.1,      # 最小温度
            'max_temperature': 2.0,      # 最大温度
            'early_answer_threshold': 0.7 # 早期回答的置信度阈值
        },
        
        # 性能优化参数
        'optimization': {
            'max_cells_retrieved': 10,   # 最大检索单元格数
            'max_neighbors_explored': 8, # 最大邻居探索数
            'enable_caching': True,      # 启用缓存
            'parallel_actions': False    # 是否并行执行动作（实验性）
        },
        
        # 调试和统计
        'debug': {
            'verbose_search': False,     # 详细搜索日志
            'log_rewards': False,        # 记录奖励计算
            'save_search_tree': False,   # 保存搜索树
            'track_node_visits': True    # 跟踪节点访问
        }
    }
    
    # 针对不同数据集的优化配置
    DATASET_CONFIGS = {
        'hitab': {
            'c_puct': 1.6,
            'max_simulations': 60,
            'max_depth': 10,
            'reward_weights': {
                'completeness': 0.35,
                'path_quality': 0.3,
                'answer_relevance': 0.25,
                'efficiency': 0.1,
                'diversity': 0.0
            }
        },
        
        'ait-qa': {
            'c_puct': 0.8,              # 调优最优值
            'max_simulations': 80,       # 调优最优值
            'max_depth': 12,            # 调优最优值
            'simulation_timeout': 20,    # 调优最优值
            'reward_weights': {
                'completeness': 0.45,
                'path_quality': 0.2,
                'answer_relevance': 0.2,
                'efficiency': 0.1,
                'diversity': 0.05
            }
        }
    }
    
    # 性能级别配置
    PERFORMANCE_LEVELS = {
        'fast': {
            'max_simulations': 20,
            'max_depth': 5,
            'simulation_timeout': 15,
            'optimization': {
                'max_cells_retrieved': 6,
                'max_neighbors_explored': 4,
                'enable_caching': True
            }
        },
        
        'balanced': {
            'max_simulations': 50,
            'max_depth': 8,
            'simulation_timeout': 30,
            'optimization': {
                'max_cells_retrieved': 10,
                'max_neighbors_explored': 8,
                'enable_caching': True
            }
        },
        
        'thorough': {
            'max_simulations': 100,
            'max_depth': 12,
            'simulation_timeout': 60,
            'optimization': {
                'max_cells_retrieved': 15,
                'max_neighbors_explored': 12,
                'enable_caching': True
            }
        }
    }
    
    def __init__(self, dataset: str = None, performance_level: str = 'balanced', custom_config: Dict = None):
        """
        初始化配置
        
        Args:
            dataset: 数据集名称（hitab, ait-qa）
            performance_level: 性能级别（fast, balanced, thorough）
            custom_config: 自定义配置字典
        """
        self.config = self._merge_configs(dataset, performance_level, custom_config)
    
    def _merge_configs(self, dataset: str, performance_level: str, custom_config: Dict) -> Dict:
        """合并各种配置"""
        # 从默认配置开始
        config = self.DEFAULT_CONFIG.copy()
        
        # 应用数据集特定配置
        if dataset and dataset in self.DATASET_CONFIGS:
            config = self._deep_merge(config, self.DATASET_CONFIGS[dataset])
        
        # 应用性能级别配置
        if performance_level in self.PERFORMANCE_LEVELS:
            config = self._deep_merge(config, self.PERFORMANCE_LEVELS[performance_level])
        
        # 应用自定义配置
        if custom_config:
            config = self._deep_merge(config, custom_config)
        
        return config
    
    def _deep_merge(self, base_dict: Dict, update_dict: Dict) -> Dict:
        """深度合并字典"""
        import copy
        result = copy.deepcopy(base_dict)
        
        for key, value in update_dict.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get(self, key: str, default=None):
        """获取配置值，支持嵌套键（用.分隔）"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """设置配置值，支持嵌套键（用.分隔）"""
        keys = key.split('.')
        target = self.config
        
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
        
        target[keys[-1]] = value
    
    def update_from_args(self, args):
        """从命令行参数更新配置"""
        # 映射命令行参数到配置键
        arg_mappings = {
            'mcts_c_puct': 'c_puct',
            'mcts_max_simulations': 'max_simulations',
            'mcts_max_depth': 'max_depth',
            'mcts_timeout': 'simulation_timeout'
        }
        
        for arg_name, config_key in arg_mappings.items():
            if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                self.set(config_key, getattr(args, arg_name))
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return self.config.copy()
    
    def save_to_file(self, filename: str):
        """保存配置到文件"""
        import json
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load_from_file(cls, filename: str):
        """从文件加载配置"""
        import json
        with open(filename, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        instance = cls()
        instance.config = config
        return instance
    
    def print_config(self):
        """打印当前配置"""
        print("\n🎛️ 当前MCTS配置:")
        print("="*50)
        self._print_dict(self.config, indent=0)
        print("="*50)
    
    def _print_dict(self, d: Dict, indent: int = 0):
        """递归打印字典"""
        for key, value in d.items():
            if isinstance(value, dict):
                print("  " * indent + f"{key}:")
                self._print_dict(value, indent + 1)
            else:
                print("  " * indent + f"{key}: {value}")


# 预定义的一些常用配置
QUICK_TEST_CONFIG = MCTSConfig(
    performance_level='fast',
    custom_config={
        'max_simulations': 10,
        'max_depth': 3,
        'simulation_timeout': 10
    }
)

PRODUCTION_CONFIG = MCTSConfig(
    performance_level='balanced',
    custom_config={
        'debug': {
            'verbose_search': False,
            'log_rewards': False,
            'save_search_tree': False
        }
    }
)

DEBUG_CONFIG = MCTSConfig(
    performance_level='balanced',
    custom_config={
        'debug': {
            'verbose_search': True,
            'log_rewards': True,
            'save_search_tree': True,
            'track_node_visits': True
        }
    }
)


def create_config_for_dataset(dataset: str, performance_level: str = 'balanced') -> MCTSConfig:
    """为特定数据集创建优化配置"""
    return MCTSConfig(dataset=dataset, performance_level=performance_level)


def optimize_config_for_hardware() -> Dict:
    """根据硬件性能优化配置"""
    import psutil
    import multiprocessing
    
    # 获取系统信息
    cpu_count = multiprocessing.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # 根据硬件调整配置
    config_adjustments = {}
    
    if cpu_count >= 8 and memory_gb >= 16:
        # 高性能机器
        config_adjustments = {
            'max_simulations': 80,
            'max_depth': 10,
            'optimization': {
                'parallel_actions': True,
                'max_cells_retrieved': 15
            }
        }
    elif cpu_count >= 4 and memory_gb >= 8:
        # 中等性能机器
        config_adjustments = {
            'max_simulations': 50,
            'max_depth': 8,
            'optimization': {
                'parallel_actions': False,
                'max_cells_retrieved': 10
            }
        }
    else:
        # 低性能机器
        config_adjustments = {
            'max_simulations': 20,
            'max_depth': 6,
            'optimization': {
                'parallel_actions': False,
                'max_cells_retrieved': 6
            }
        }
    
    return config_adjustments


if __name__ == '__main__':
    # 演示配置使用
    print("🎛️ MCTS配置系统演示")
    
    # 创建默认配置
    config = MCTSConfig()
    config.print_config()
    
    # 创建数据集特定配置
    hitab_config = create_config_for_dataset('hitab', 'thorough')
    print(f"\nHiTab数据集UCB系数: {hitab_config.get('c_puct')}")
    
    # 硬件优化配置
    hw_optimizations = optimize_config_for_hardware()
    print(f"\n硬件优化建议: {hw_optimizations}")
    
    # 保存和加载配置
    config.save_to_file('mcts_config_example.json')
    print("\n配置已保存到 mcts_config_example.json") 
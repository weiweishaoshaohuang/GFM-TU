#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCTS超参数调优工具
用于自动寻找最佳的MCTS参数组合
"""

import os
import sys
import json
import time
import itertools
import argparse
from typing import Dict, List, Tuple, Any

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from test_mcts import MCTSTestSuite, create_test_args


class MCTSHyperparameterTuner:
    """MCTS超参数调优器"""
    
    def __init__(self, base_args):
        """初始化调优器"""
        self.base_args = base_args
        self.best_params = None
        self.best_score = 0.0
        self.tuning_results = []
        
        # 定义参数搜索空间 網格
        self.param_space = {
            # 'mcts_c_puct': [0.8, 1.0, 1.4, 2.0, 2.5],           # UCB探索系数
            # 'mcts_max_simulations': [20, 30, 50, 80, 100],       # 最大模拟次数
            # 'mcts_max_depth': [6, 8, 10, 12],                    # 最大搜索深度
            # 'mcts_timeout': [15, 20, 30, 45]                     # 模拟超时
            'mcts_c_puct': [0.8, 1.0, 1.2],      
            'mcts_max_simulations': [80, 100, 120],  
            'mcts_max_depth': [8, 10, 12],          
            'mcts_timeout': [15, 20]   
        }
        
        # 定义评估指标权重
        self.metric_weights = {
            'em_score': 0.4,        # EM分数权重
            'llm_score': 0.4,       # LLM分数权重
            'success_rate': 0.15,   # 成功率权重
            'time_efficiency': 0.05 # 时间效率权重
        }
    
    def grid_search(self, max_combinations: int = 50) -> Dict:
        """网格搜索最佳参数"""
        print("🔍 开始网格搜索超参数...")
        
        # 生成参数组合
        param_combinations = self._generate_param_combinations(max_combinations)
        print(f"共生成{len(param_combinations)}个参数组合")
        
        # 加载测试数据（使用更少的样本进行调优）
        test_suite = MCTSTestSuite(self.base_args)
        test_cases = test_suite.load_test_data(num_samples=2)  # 调优时使用少量样本
        
        if not test_cases:
            print("❌ 无法加载测试数据")
            return {}
        
        print(f"使用{len(test_cases)}个测试样本进行调优")
        
        # 测试每个参数组合
        for i, params in enumerate(param_combinations):
            print(f"\n--- 测试参数组合 {i+1}/{len(param_combinations)} ---")
            print(f"参数: {params}")
            
            # 更新参数
            args = self._update_args_with_params(self.base_args, params)
            
            # 运行测试
            start_time = time.time()
            test_suite_for_params = MCTSTestSuite(args)
            
            try:
                # 只测试MCTS方法
                mcts_results = []
                for test_case in test_cases:
                    result = test_suite_for_params.run_single_test(test_case, use_mcts=True)
                    mcts_results.append(result)
                
                end_time = time.time()
                
                # 计算综合分数
                score = self._calculate_combined_score(mcts_results, end_time - start_time)
                
                # 记录结果
                tuning_result = {
                    'params': params,
                    'score': score,
                    'results': mcts_results,
                    'total_time': end_time - start_time
                }
                self.tuning_results.append(tuning_result)
                
                print(f"综合分数: {score:.4f}")
                
                # 更新最佳参数
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = params
                    print(f"🎯 发现更佳参数组合！分数: {score:.4f}")
                
            except Exception as e:
                print(f"❌ 参数测试失败: {e}")
                continue
        
        # 返回调优结果
        tuning_summary = {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'all_results': self.tuning_results,
            'param_analysis': self._analyze_parameter_importance()
        }
        
        return tuning_summary
    
    def _generate_param_combinations(self, max_combinations: int) -> List[Dict]:
        """生成参数组合"""
        all_combinations = []
        
        # 生成所有可能的组合
        param_names = list(self.param_space.keys())
        param_values = [self.param_space[name] for name in param_names]
        
        for combination in itertools.product(*param_values):
            params = dict(zip(param_names, combination))
            all_combinations.append(params)
        
        # 如果组合数量太多，随机选择部分组合
        if len(all_combinations) > max_combinations:
            import random
            random.seed(42)  # 确保可重现
            all_combinations = random.sample(all_combinations, max_combinations)
        
        return all_combinations
    
    def _update_args_with_params(self, base_args, params: Dict):
        """用新参数更新args"""
        import copy
        new_args = copy.deepcopy(base_args)
        
        for param_name, param_value in params.items():
            setattr(new_args, param_name, param_value)
        
        return new_args
    
    def _calculate_combined_score(self, results: List[Dict], total_time: float) -> float:
        """计算综合分数"""
        successful_results = [r for r in results if r['success']]
        
        if not successful_results:
            return 0.0
        
        # 计算各项指标
        avg_em = sum(r['em_score'] for r in successful_results) / len(successful_results)
        avg_llm = sum(r['llm_score'] for r in successful_results) / len(successful_results)
        success_rate = len(successful_results) / len(results)
        
        # 时间效率（越短越好，标准化到0-1）
        max_reasonable_time = 60  # 最大合理时间（秒）
        time_efficiency = max(0, 1.0 - total_time / max_reasonable_time)
        
        # 计算加权综合分数
        combined_score = (
            self.metric_weights['em_score'] * avg_em +
            self.metric_weights['llm_score'] * avg_llm +
            self.metric_weights['success_rate'] * success_rate +
            self.metric_weights['time_efficiency'] * time_efficiency
        )
        
        return combined_score
    
    def _analyze_parameter_importance(self) -> Dict:
        """分析参数重要性"""
        if not self.tuning_results:
            return {}
        
        param_analysis = {}
        
        for param_name in self.param_space.keys():
            # 按参数值分组计算平均分数
            param_scores = {}
            for result in self.tuning_results:
                param_value = result['params'][param_name]
                if param_value not in param_scores:
                    param_scores[param_value] = []
                param_scores[param_value].append(result['score'])
            
            # 计算每个参数值的平均分数 比如'avg_scores_by_value': {0.8: 0.72, 1.0: 0.81, 1.4: 0.83, 2.0: 0.75, 2.5: 0.70} 看出1.4對應的平均0,83最高
            param_avg_scores = {}
            for value, scores in param_scores.items():
                param_avg_scores[value] = sum(scores) / len(scores)
            
            param_analysis[param_name] = {
                'avg_scores_by_value': param_avg_scores,
                'best_value': max(param_avg_scores.keys(), key=lambda x: param_avg_scores[x]),
                'score_range': max(param_avg_scores.values()) - min(param_avg_scores.values()) # 越大代表調參效果越好
            }
        
        return param_analysis
    
    def bayesian_optimization(self, n_iterations: int = 20) -> Dict:
        """贝叶斯优化（简化版）"""
        print("🧠 开始贝叶斯优化...")
        
        # 加载测试数据
        test_suite = MCTSTestSuite(self.base_args)
        test_cases = test_suite.load_test_data(num_samples=2)
        
        if not test_cases:
            print("❌ 无法加载测试数据")
            return {}
        
        # 初始随机采样
        initial_samples = 5
        random_params = self._generate_param_combinations(initial_samples)
        
        for i, params in enumerate(random_params):
            print(f"\n--- 初始采样 {i+1}/{initial_samples} ---")
            score = self._evaluate_params(params, test_cases)
            
            tuning_result = {
                'params': params,
                'score': score,
                'iteration': i,
                'method': 'random_init'
            }
            self.tuning_results.append(tuning_result)
            
            if score > self.best_score:
                self.best_score = score
                self.best_params = params
        
        # 贝叶斯优化迭代（简化为局部搜索）
        for iteration in range(n_iterations - initial_samples):
            print(f"\n--- 贝叶斯优化迭代 {iteration+1}/{n_iterations-initial_samples} ---")
            
            # 基于当前最佳参数进行局部搜索
            candidate_params = self._generate_local_candidates(self.best_params)
            
            best_candidate_score = 0
            best_candidate_params = None
            
            for params in candidate_params:
                score = self._evaluate_params(params, test_cases)
                
                tuning_result = {
                    'params': params,
                    'score': score,
                    'iteration': initial_samples + iteration,
                    'method': 'bayesian'
                }
                self.tuning_results.append(tuning_result)
                
                if score > best_candidate_score:
                    best_candidate_score = score
                    best_candidate_params = params
            
            # 更新最佳参数
            if best_candidate_score > self.best_score:
                self.best_score = best_candidate_score
                self.best_params = best_candidate_params
                print(f"🎯 贝叶斯优化发现更佳参数！分数: {best_candidate_score:.4f}")
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'optimization_history': self.tuning_results
        }
    
    def _evaluate_params(self, params: Dict, test_cases: List[Dict]) -> float:
        """评估参数组合"""
        args = self._update_args_with_params(self.base_args, params)
        test_suite = MCTSTestSuite(args)
        
        try:
            start_time = time.time()
            results = []
            
            for test_case in test_cases:
                result = test_suite.run_single_test(test_case, use_mcts=True)
                results.append(result)
            
            end_time = time.time()
            score = self._calculate_combined_score(results, end_time - start_time)
            
            print(f"参数 {params} -> 分数: {score:.4f}")
            return score
            
        except Exception as e:
            print(f"参数评估失败: {e}")
            return 0.0
    
    def _generate_local_candidates(self, base_params: Dict, n_candidates: int = 5) -> List[Dict]:
        """生成局部候选参数 生成N_candidate個數據其中數據照base_params隨機找一個參數將他往右或往左調"""
        candidates = []
        
        for _ in range(n_candidates):
            candidate = base_params.copy()
            
            # 随机选择一个参数进行调整
            import random
            param_to_adjust = random.choice(list(self.param_space.keys()))
            current_value = candidate[param_to_adjust]
            possible_values = self.param_space[param_to_adjust]
            
            # 选择相邻的值
            current_index = possible_values.index(current_value)
            
            # 向左或向右移动
            if random.random() < 0.5 and current_index > 0:
                candidate[param_to_adjust] = possible_values[current_index - 1]
            elif current_index < len(possible_values) - 1:
                candidate[param_to_adjust] = possible_values[current_index + 1]
            
            candidates.append(candidate)
        
        return candidates
    
    def print_tuning_report(self, tuning_results: Dict):
        """打印调优报告"""
        print("\n" + "="*80)
        print("🎯 MCTS超参数调优报告")
        print("="*80)
        
        if not tuning_results.get('best_params'):
            print("❌ 没有找到有效的参数组合")
            return
        
        print(f"\n🏆 最佳参数组合:")
        best_params = tuning_results['best_params']
        best_score = tuning_results['best_score']
        
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        print(f"  综合分数: {best_score:.4f}")
        
        # 参数重要性分析
        if 'param_analysis' in tuning_results:
            print(f"\n📊 参数重要性分析:")
            param_analysis = tuning_results['param_analysis']
            
            for param_name, analysis in param_analysis.items():
                best_value = analysis['best_value']
                score_range = analysis['score_range']
                print(f"  {param_name}:")
                print(f"    最佳值: {best_value}")
                print(f"    影响程度: {score_range:.4f}")
        
        # 推荐配置
        print(f"\n💡 推荐配置:")
        print("建议在main.py或配置文件中使用以下参数:")
        for param, value in best_params.items():
            print(f"  --{param} {value}")


def create_tuning_args():
    """创建调优参数"""
    # 基于测试参数创建
    args = create_test_args()
    
    # 修改为使用AIT-QA数据集
    args.dataset = "ait-qa"
    args.qa_path = "dataset/AIT-QA/aitqa_clean_questions.json"
    args.table_folder = ""  # AIT-QA不需要table_folder
    args.embed_cache_dir = "dataset/AIT-QA/"
    
    # 调优特定参数
    args.test_samples = 2  # 调优时使用较少样本
    args.debug = False     # 调优时关闭调试输出
    
    return args


def main():
    """主调优函数"""
    print("🎛️ 启动MCTS超参数调优...")
    
    # 创建基础参数
    base_args = create_tuning_args()
    
    # 检查API key
    if not base_args.key:
        print("❌ 错误: 需要有效的API key")
        print("解决方案:")
        print("1. 使用start.bat中的配置（应该已自动加载）")
        print("2. 或者使用命令行参数: --key your-api-key")
        print(f"3. 当前使用模型: {base_args.model}")
        print(f"4. 当前base_url: {base_args.base_url}")
        return
    
    # 检查数据集
    if not os.path.exists(base_args.qa_path):
        print(f"❌ 找不到数据集: {base_args.qa_path}")
        return
    
    # 创建调优器
    tuner = MCTSHyperparameterTuner(base_args)
    
    # 选择调优方法
    tuning_method = input("\n选择调优方法 (1: 网格搜索, 2: 贝叶斯优化): ").strip()
    
    if tuning_method == "1":
        print("🔍 使用网格搜索...")
        max_combinations = int(input("最大参数组合数 (推荐20-50): ").strip() or "30")
        tuning_results = tuner.grid_search(max_combinations)
    elif tuning_method == "2":
        print("🧠 使用贝叶斯优化...")
        n_iterations = int(input("优化迭代次数 (推荐15-30): ").strip() or "20")
        tuning_results = tuner.bayesian_optimization(n_iterations)
    else:
        print("❌ 无效选择，使用默认网格搜索")
        tuning_results = tuner.grid_search(30)
    
    # 打印调优报告
    tuner.print_tuning_report(tuning_results)
    
    # 保存结果
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_file = f"mcts_tuning_results_{timestamp}.json"
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(tuning_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 调优结果已保存到: {result_file}")
    print("\n✅ 超参数调优完成！")


if __name__ == '__main__':
    main() 
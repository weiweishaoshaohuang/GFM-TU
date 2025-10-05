#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCTS系统测试脚本
用于验证MCTS实现的正确性和性能对比
"""

import os
import sys
import argparse
import json
import time
from typing import Dict, List, Any

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs import GEMINI_KEYS, hyperparameter
from GraphRetriever.graph_retriver import GraphRetriever
from Generator.openai_api import ChatGPTTool
from GraphRetriever.dense_retriever import load_dense_retriever
from Generator.Gemini_model import GeminiTool
from iterative_reasoning import GraphReasoner
from mcts_adapter import MCTSGraphReasoner
from compute_score import eval_ex_match, LLM_eval

def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.strip().lower()
    if v in ("1","true","t","yes","y"):
        return True
    if v in ("0","false","f","no","n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected (true/false, 1/0)")

class MCTSTestSuite:
    """MCTS测试套件"""
    
    def __init__(self, args):
        self.args = args
        self.results = {
            'traditional': [],
            'mcts': [],
            'comparison': {}
        }
        
        # 加载模型
        self.model, self.dense_retriever = self._load_model()
        
    def _load_model(self):
        """加载模型"""
        if 'gemini' in self.args.model:
            gemini_key_index = 0
            gemini_key = GEMINI_KEYS[gemini_key_index]
            model = GeminiTool(gemini_key, self.args)
        else:
            model = ChatGPTTool(self.args)
        
        dense_retriever = load_dense_retriever(self.args)
        return model, dense_retriever
    
    def load_test_data(self, num_samples: int = 5) -> List[Dict]:
        """加载测试数据"""
        test_cases = []
        
        if self.args.dataset in ('hitab', 'Hitab'):
            import jsonlines
            with open(self.args.qa_path, "r+", encoding='utf-8') as f:
                qas = [item for item in jsonlines.Reader(f)]
            
            # 取前几个样本进行测试
            selected_qas = qas[:num_samples]
            
            for qa in selected_qas:
                table_path = self.args.table_folder + qa['table_id'] + '.json'
                with open(table_path, "r+", encoding='utf-8') as f:
                    table = json.load(f)
                
                test_case = {
                    'query': qa['question'],
                    'answer': '|'.join([str(i) for i in qa['answer']]),
                    'table_caption': table['title'],
                    'table': table['texts'],
                    'table_path': table_path
                }
                test_cases.append(test_case)
        
        elif self.args.dataset in ('AIT-QA', 'ait-qa'):
            with open(self.args.qa_path, 'r', encoding='utf-8') as f:
                qas = json.load(f)
            
            selected_qas = qas[:num_samples]
            
            for qa in selected_qas:
                test_case = {
                    'query': qa['question'],
                    'answer': '|'.join([str(i) for i in qa['answers']]),
                    'table_caption': '',
                    'table': qa['table'],
                    'table_path': qa
                }
                test_cases.append(test_case)
        
        print(f"加载了{len(test_cases)}个测试用例")
        return test_cases
    
    def run_single_test(self, test_case: Dict, use_mcts: bool = True) -> Dict:
        """运行单个测试用例"""
        query = test_case['query']
        answer = test_case['answer']
        caption = test_case['table_caption']
        table = test_case['table']
        table_path = test_case['table_path']
        
        print(f"\n{'='*60}")
        print(f"问题: {query}")
        print(f"正确答案: {answer}")
        print(f"推理方法: {'MCTS' if use_mcts else '传统'}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # 创建图检索器
            graph_retriever = GraphRetriever(table_path, self.model, self.dense_retriever, 
                                           self.args.dataset, table_cation=caption)
            
            # 创建推理器
            if use_mcts:
                graph_reasoner = MCTSGraphReasoner(self.args, self.model, query, table, 
                                                 caption, graph_retriever, self.args.dataset)
            else:
                graph_reasoner = GraphReasoner(self.args, self.model, query, table, 
                                             caption, graph_retriever, self.args.dataset)
            
            # 执行推理
            output = graph_reasoner.iterative_reasoning()
            
            end_time = time.time()
            
            # 计算评估指标
            em_score = eval_ex_match(output, answer)
            llm_score = LLM_eval(self.model, query, output, answer)
            
            result = {
                'query': query,
                'predicted_answer': output,
                'ground_truth': answer,
                'em_score': em_score,
                'llm_score': llm_score,
                'time_cost': end_time - start_time,
                'method': 'MCTS' if use_mcts else 'Traditional',
                'success': True
            }
            
            # 如果是MCTS，添加统计信息
            if use_mcts and hasattr(graph_reasoner, 'get_stats'):
                result['mcts_stats'] = graph_reasoner.get_stats()
            
            print(f"模型回答: {output}")
            print(f"EM分数: {em_score}")
            print(f"LLM分数: {llm_score}")
            print(f"耗时: {end_time - start_time:.2f}秒")
            
            return result
            
        except Exception as e:
            print(f"测试失败: {e}")
            return {
                'query': query,
                'error': str(e),
                'success': False,
                'method': 'MCTS' if use_mcts else 'Traditional',
                'time_cost': time.time() - start_time
            }
    
    def run_comparison_test(self, test_cases: List[Dict]) -> Dict:
        """运行对比测试"""
        print("\n🔄 开始对比测试...")
        
        traditional_results = []
        mcts_results = []
        
        for i, test_case in enumerate(test_cases):
            print(f"\n--- 测试用例 {i+1}/{len(test_cases)} ---")
            
            # 传统方法测试
            print("🔹 测试传统方法...")
            traditional_result = self.run_single_test(test_case, use_mcts=False)
            traditional_results.append(traditional_result)
            
            # MCTS方法测试
            print("\n🔹 测试MCTS方法...")
            mcts_result = self.run_single_test(test_case, use_mcts=True)
            mcts_results.append(mcts_result)
        
        # 计算统计信息
        comparison = self._calculate_comparison_stats(traditional_results, mcts_results)
        
        return {
            'traditional_results': traditional_results,
            'mcts_results': mcts_results,
            'comparison': comparison
        }
    
    def _calculate_comparison_stats(self, traditional_results: List[Dict], mcts_results: List[Dict]) -> Dict:
        """计算对比统计信息"""
        stats = {
            'traditional': self._calculate_method_stats(traditional_results),
            'mcts': self._calculate_method_stats(mcts_results),
            'improvement': {}
        }
        
        # 计算改进程度
        trad_stats = stats['traditional']
        mcts_stats = stats['mcts']
        
        if trad_stats['avg_em'] > 0:
            stats['improvement']['em_improvement'] = (mcts_stats['avg_em'] - trad_stats['avg_em']) / trad_stats['avg_em'] * 100
        else:
            stats['improvement']['em_improvement'] = 0
        
        if trad_stats['avg_llm'] > 0:
            stats['improvement']['llm_improvement'] = (mcts_stats['avg_llm'] - trad_stats['avg_llm']) / trad_stats['avg_llm'] * 100
        else:
            stats['improvement']['llm_improvement'] = 0
        
        stats['improvement']['time_change'] = mcts_stats['avg_time'] - trad_stats['avg_time']
        stats['improvement']['success_rate_change'] = mcts_stats['success_rate'] - trad_stats['success_rate']
        
        return stats
    
    def _calculate_method_stats(self, results: List[Dict]) -> Dict:
        """计算单个方法的统计信息"""
        successful_results = [r for r in results if r['success']]
        
        if not successful_results:
            return {
                'total_cases': len(results),
                'successful_cases': 0,
                'success_rate': 0.0,
                'avg_em': 0.0,
                'avg_llm': 0.0,
                'avg_time': 0.0
            }
        
        total_em = sum(r['em_score'] for r in successful_results)
        total_llm = sum(r['llm_score'] for r in successful_results)
        total_time = sum(r['time_cost'] for r in successful_results)
        
        return {
            'total_cases': len(results),
            'successful_cases': len(successful_results),
            'success_rate': len(successful_results) / len(results),
            'avg_em': total_em / len(successful_results),
            'avg_llm': total_llm / len(successful_results),
            'avg_time': total_time / len(successful_results)
        }
    
    def print_final_report(self, comparison_results: Dict):
        """打印最终报告"""
        print("\n" + "="*80)
        print("🎯 MCTS vs 传统方法对比报告")
        print("="*80)
        
        trad_stats = comparison_results['comparison']['traditional']
        mcts_stats = comparison_results['comparison']['mcts']
        improvement = comparison_results['comparison']['improvement']
        
        print(f"\n📊 性能统计:")
        print(f"{'指标':<20} {'传统方法':<15} {'MCTS方法':<15} {'改进(%)':<10}")
        print("-" * 70)
        print(f"{'成功率':<20} {trad_stats['success_rate']:<15.3f} {mcts_stats['success_rate']:<15.3f} {improvement['success_rate_change']*100:<10.2f}")
        print(f"{'平均EM分数':<20} {trad_stats['avg_em']:<15.3f} {mcts_stats['avg_em']:<15.3f} {improvement['em_improvement']:<10.2f}")
        print(f"{'平均LLM分数':<20} {trad_stats['avg_llm']:<15.3f} {mcts_stats['avg_llm']:<15.3f} {improvement['llm_improvement']:<10.2f}")
        print(f"{'平均耗时(秒)':<20} {trad_stats['avg_time']:<15.2f} {mcts_stats['avg_time']:<15.2f} {improvement['time_change']:<10.2f}")
        
        print(f"\n🏆 结论:")
        if improvement['em_improvement'] > 0:
            print(f"✅ MCTS在EM分数上提升了 {improvement['em_improvement']:.2f}%")
        else:
            print(f"❌ MCTS在EM分数上下降了 {abs(improvement['em_improvement']):.2f}%")
        
        if improvement['llm_improvement'] > 0:
            print(f"✅ MCTS在LLM分数上提升了 {improvement['llm_improvement']:.2f}%")
        else:
            print(f"❌ MCTS在LLM分数上下降了 {abs(improvement['llm_improvement']):.2f}%")
        
        if improvement['time_change'] < 0:
            print(f"⚡ MCTS在时间上节省了 {abs(improvement['time_change']):.2f}秒")
        else:
            print(f"⏰ MCTS在时间上增加了 {improvement['time_change']:.2f}秒")
        
        # MCTS详细统计
        mcts_results = comparison_results['mcts_results']
        mcts_detailed = [r for r in mcts_results if r['success'] and 'mcts_stats' in r]
        
        if mcts_detailed:
            print(f"\n🌲 MCTS详细统计:")
            avg_simulations = sum(r['mcts_stats']['iterations'] for r in mcts_detailed) / len(mcts_detailed)
            print(f"平均模拟次数: {avg_simulations:.1f}")
            
            methods_used = [r['mcts_stats']['method_used'] for r in mcts_detailed]
            mcts_usage = sum(1 for m in methods_used if 'MCTS' in m) / len(methods_used) * 100
            print(f"MCTS使用率: {mcts_usage:.1f}%")


def create_test_args():
    """创建测试参数"""
    parser = argparse.ArgumentParser()
    
    # 基本参数 - 复用start.bat中的配置
    parser.add_argument("--model", type=str, default="qwen2-7b-instruct")
    parser.add_argument("--base_url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--key", type=str, default="", help="API key (复用start.bat配置)")
    parser.add_argument("--dataset", type=str, default="ait-qa")
    parser.add_argument("--qa_path", type=str, default="dataset/AIT-QA/aitqa_clean_questions.json")
    parser.add_argument("--table_folder", type=str, default="")
    
    parser.add_argument("--max_iteration_depth", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    
    parser.add_argument("--embed_cache_dir", type=str, default="dataset/AIT-QA/")
    parser.add_argument("--embedder_path", type=str, default="thenlper/gte-base")  # 复用start.bat配置
    
    parser.add_argument("--debug", type=str2bool, default=True)
    
    # MCTS参数
    parser.add_argument("--use_mcts", type=str2bool, default=True)
    parser.add_argument("--mcts_c_puct", type=float, default=0.8)  # 调优最优值
    parser.add_argument("--mcts_max_simulations", type=int, default=80)  # 调优最优值
    parser.add_argument("--mcts_max_depth", type=int, default=12)  # 调优最优值
    parser.add_argument("--mcts_timeout", type=int, default=20)  # 调优最优值
    
    # 测试参数
    parser.add_argument("--test_samples", type=int, default=3, help="测试样本数量")
    parser.add_argument("--save_results", type=str2bool, default=True, help="是否保存结果")
    
    return parser.parse_args()


def main():
    """主测试函数"""
    print("🚀 启动MCTS系统测试...")
    
    # 创建测试参数
    args = create_test_args()
    
    # 检查API key
    if not args.key:
        print("❌ 错误: 需要有效的API key")
        print("解决方案:")
        print("1. 使用start.bat中的配置（已自动加载）")
        print("2. 或者使用命令行参数: --key your-api-key")
        print(f"3. 当前使用模型: {args.model}")
        print(f"4. 当前base_url: {args.base_url}")
        return
    
    # 检查必要文件
    if not os.path.exists(args.qa_path):
        print(f"❌ 找不到QA文件: {args.qa_path}")
        print("请确保已下载并配置了数据集文件")
        return
    
    # 创建测试套件
    test_suite = MCTSTestSuite(args)
    
    # 加载测试数据
    test_cases = test_suite.load_test_data(args.test_samples)
    
    if not test_cases:
        print("❌ 没有找到测试用例")
        return
    
    # 运行对比测试
    comparison_results = test_suite.run_comparison_test(test_cases)
    
    # 打印最终报告
    test_suite.print_final_report(comparison_results)
    
    # 保存结果
    if args.save_results:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        result_file = f"mcts_test_results_{timestamp}.json"
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 结果已保存到: {result_file}")
    
    print("\n✅ 测试完成!")


if __name__ == '__main__':
    main()
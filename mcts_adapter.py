"""
MCTS适配器 - 保持与原有GraphReasoner接口完全兼容
同时提供MCTS推理能力
"""

import copy
from iterative_reasoning import GraphReasoner
from mcts_reasoning import MCTSReasoner
from mcts_event_logger import JsonlLogger


class MCTSGraphReasoner(GraphReasoner):
    """
    MCTS图推理器 - 继承原有GraphReasoner
    保持所有原有接口不变，内部使用MCTS进行推理
    """
    
    def __init__(self, args, model, query, table, caption, graph_retriever, dataset='hitab'):
        """初始化，完全兼容原有构造函数"""
        # 调用父类构造函数
        super().__init__(args, model, query, table, caption, graph_retriever, dataset)
        
        # 添加MCTS相关配置
        self.use_mcts = getattr(args, 'use_mcts', True)  # 是否使用MCTS，默认启用
        self.mcts_config = {
            'c_puct': getattr(args, 'mcts_c_puct', 0.8),  # UCB探索系数，调优最优值
            'max_simulations': getattr(args, 'mcts_max_simulations', 80),  # 最大模拟次数，调优最优值
            'max_depth': getattr(args, 'mcts_max_depth', 12),  # 最大搜索深度，调优最优值
            'simulation_timeout': getattr(args, 'mcts_timeout', 20),  # 模拟超时，调优最优值
        }
        
        # 初始化MCTS推理器
        self.mcts_reasoner = None
        
        # 统计信息
        self.reasoning_stats = {
            'method_used': 'unknown',
            'total_time': 0,
            'iterations': 0,
            'success': False
        }
    
    def iterative_reasoning(self):
        """
        主推理方法 - 保持原有接口不变
        内部根据配置选择使用MCTS或原有方法
        """
        import time
        start_time = time.time()
        
        try:
            if self.use_mcts:
                print("🌲 使用MCTS推理...")
                result = self._mcts_reasoning()
                self.reasoning_stats['method_used'] = 'MCTS'
            else:
                print("📏 使用传统线性推理...")
                result = self._traditional_reasoning()
                self.reasoning_stats['method_used'] = 'Traditional'
            
            self.reasoning_stats['success'] = True
            return result
            
        except Exception as e:
            print(f"推理过程出错: {e}")
            
            # 检查是否是API key相关错误
            if "401" in str(e) or "invalid_api_key" in str(e) or "Incorrect API key" in str(e):
                print("❌ API key错误！请检查您的API key配置")
                print("解决方案:")
                print("1. 检查start.bat中的key配置是否正确")
                print("2. 或者在命令行中指定: --key your-api-key")
                print("3. 确认base_url配置正确（当前通义千问服务）")
                self.reasoning_stats['success'] = False
                return "API key配置错误"
            
            # 出错时回退到传统方法
            if self.use_mcts:
                print("MCTS推理失败，回退到传统方法...")
                try:
                    result = self._traditional_reasoning()
                    self.reasoning_stats['method_used'] = 'Traditional (fallback)'
                    self.reasoning_stats['success'] = True
                    return result
                except Exception as fallback_error:
                    print(f"传统方法也失败: {fallback_error}")
                    self.reasoning_stats['success'] = False
                    return "推理失败"
            else:
                self.reasoning_stats['success'] = False
                return "推理失败"
        
        finally:
            self.reasoning_stats['total_time'] = time.time() - start_time
            if self.args.debug:
                self._print_stats()
    
    def _mcts_reasoning(self):
        """使用MCTS进行推理"""
        # 确保已初始化prompt
        if not hasattr(self, 'start_cells') or not self.start_cells:
            self.initialize_prompt()
        
        # 创建MCTS推理器
        if self.mcts_reasoner is None:
            event_logger = JsonlLogger("mcts_events.jsonl") if getattr(self.args, "debug", False) else None
            self.mcts_reasoner = MCTSReasoner(self, event_logger=event_logger)
            
            # 应用配置参数
            self.mcts_reasoner.c_puct = self.mcts_config['c_puct']
            self.mcts_reasoner.max_simulations = self.mcts_config['max_simulations']
            self.mcts_reasoner.max_depth = self.mcts_config['max_depth']
            self.mcts_reasoner.simulation_timeout = self.mcts_config['simulation_timeout']
        else:
            if getattr(self.args, "debug", False) and self.mcts_reasoner.event_logger is None:
                self.mcts_reasoner.event_logger = JsonlLogger("mcts_events.jsonl")

        # 执行MCTS搜索
        result = self.mcts_reasoner.search()
        
        # 更新统计信息
        self.reasoning_stats['iterations'] = self.mcts_reasoner.stats['total_simulations']
        
        return result
    
    def _traditional_reasoning(self):
        """使用传统方法进行推理"""
        # 调用父类的原始方法
        return super().iterative_reasoning()
    
    def _print_stats(self):
        """打印推理统计信息"""
        print("\n" + "="*50)
        print("推理统计信息:")
        print(f"方法: {self.reasoning_stats['method_used']}")
        print(f"总时间: {self.reasoning_stats['total_time']:.2f}秒")
        print(f"迭代次数: {self.reasoning_stats['iterations']}")
        print(f"成功: {self.reasoning_stats['success']}")
        
        if hasattr(self, 'mcts_reasoner') and self.mcts_reasoner:
            stats = self.mcts_reasoner.stats
            print(f"模拟次数: {stats['total_simulations']}")
            print(f"成功模拟: {stats['successful_simulations']}")
            print(f"最佳奖励: {stats['best_reward']:.3f}")
        
        print("="*50 + "\n")
    
    # 保持所有原有方法的兼容性
    def initialize_prompt(self):
        """保持原有的初始化方法"""
        return super().initialize_prompt()
    
    def LLM_generate(self, prompt, isrepeated=0.0, response_mime_type=None):
        """保持原有的LLM生成方法"""
        return super().LLM_generate(prompt, isrepeated, response_mime_type)
    
    def Thought(self):
        """保持原有的思考方法"""
        return super().Thought()
    
    def Action(self, thinking_text):
        """保持原有的动作方法"""
        return super().Action(thinking_text)
    
    def Answer(self, Thinking_text, answer_explan='', last_interact_step=''):
        """保持原有的回答方法"""
        return super().Answer(Thinking_text, answer_explan, last_interact_step)
    
    # 添加一些配置方法
    def set_mcts_config(self, **kwargs):
        """动态设置MCTS配置"""
        for key, value in kwargs.items():
            if key in self.mcts_config:
                self.mcts_config[key] = value
                print(f"更新MCTS配置: {key} = {value}")
        
        # 如果已经有推理器，更新其配置
        if self.mcts_reasoner:
            self.mcts_reasoner.c_puct = self.mcts_config['c_puct']
            self.mcts_reasoner.max_simulations = self.mcts_config['max_simulations']
            self.mcts_reasoner.max_depth = self.mcts_config['max_depth']
            self.mcts_reasoner.simulation_timeout = self.mcts_config['simulation_timeout']
    
    def enable_mcts(self, enable=True):
        """启用或禁用MCTS"""
        self.use_mcts = enable
        print(f"MCTS推理: {'启用' if enable else '禁用'}")
    
    def get_stats(self):
        """获取推理统计信息"""
        return copy.deepcopy(self.reasoning_stats)


# 兼容性函数 - 用于替换原有的GraphReasoner
def create_reasoner(args, model, query, table, caption, graph_retriever, dataset='hitab'):
    """
    创建推理器的工厂函数
    可以根据配置选择使用MCTS或传统方法
    """
    # 检查是否启用MCTS
    use_mcts = getattr(args, 'use_mcts', True)
    
    if use_mcts:
        print("创建MCTS图推理器...")
        return MCTSGraphReasoner(args, model, query, table, caption, graph_retriever, dataset)
    else:
        print("创建传统图推理器...")
        return GraphReasoner(args, model, query, table, caption, graph_retriever, dataset)


# 为了保持完全兼容，可以直接替换原有的类名
# 这样原有代码不需要任何修改
class GraphReasonerMCTS(MCTSGraphReasoner):
    """别名，用于保持完全兼容"""
    pass 
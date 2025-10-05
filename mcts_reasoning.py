import copy
import math
import random
import time
from collections import OrderedDict
from typing import List, Tuple, Any, Optional

from configs import PROMPT_TEMPLATE, hyperparameter
from Generator.parse_output import get_action_list, parse_action_json
import json

class MCTSNode:
    """MCTS搜索树节点"""
    
    def __init__(self, state: 'ReasoningState', parent: Optional['MCTSNode'] = None, action: Optional[Tuple] = None):
        self.state = state # 當前節點對應的推理狀態(推理到哪的完整快照:已找到的單元格與ID，已探索的鄰居關係，目前步數、推理路徑、置信度/完整度，原始上下文)
        self.parent = parent # 根節點為None
        self.action = action  # 到达此节点的动作 (action_type, parameters) 如('SearchNode', ['percent'])
        self.children: List['MCTSNode'] = []
        self.visits = 0
        self.total_reward = 0.0
        self.ucb_score = float('inf')
        self.is_expanded = False
        self.untried_actions = None  # 尚未展開的 action 清單
        
    def get_average_reward(self) -> float:
        """获取平均奖励"""
        if self.visits == 0:
            return 0.0
        return self.total_reward / self.visits
    
    def calculate_ucb(self, c_puct: float = 1.0) -> float:
        """计算UCB分数"""
        if self.visits == 0: # 沒被訪問過，優先訪問
            return float('inf')
        
        if self.parent is None or self.parent.visits == 0: # 沒辦法計算探索，回退平均(通常不會走到這)
            return self.get_average_reward()
        
        exploitation = self.get_average_reward() # 平均回報
        exploration = c_puct * math.sqrt(math.log(self.parent.visits) / self.visits) # 探索
        return exploitation + exploration
    
    def is_leaf(self) -> bool:
        """是否为叶子节点"""
        return len(self.children) == 0
    
    def is_terminal(self) -> bool:
        """是否为终端节点"""
        return self.state.is_terminal()


class ReasoningState:
    """推理状态表示"""
    
    def __init__(self, original_reasoner=None):
        # 从原始推理器复制基本信息
        if original_reasoner:
            self.query = original_reasoner.query
            self.table_caption = original_reasoner.table_caption
            self.markdown_table = original_reasoner.markdown_table
            self.dataset = original_reasoner.dataset
            self.graph_retriever = original_reasoner.graph_retriever
            self.model = original_reasoner.model
            self.args = original_reasoner.args
        
        # MCTS特有状态信息
        self.step = 1 # 根到這裡走到第幾步
        self.retrieved_cells = [] # 已檢索單元格
        self.retrieved_cell_ids = []
        self.explored_neighbors = {} # 哪些cell做過鄰居/共同鄰居探索，防重複 可能存 0: {"all_neighbors": True}紀錄id:0取過全鄰居或"0-1": True取過共同鄰居
        self.reasoning_path = []
        self.intermediate_results = OrderedDict()
        self.search_cell_history_dict = OrderedDict()
        self.hit_cell_neighbors_history_dict = OrderedDict()
        
        # 子图和连接信息 NOTE
        self.start_cells = ''
        self.connect_graph_prompt = ''
        self.conn_graph_result = []
        self.reasoning_path_dict = OrderedDict()
        self.reasoning_path_prompt = ''
        
        # 状态评估相关
        self.confidence_score = 0.0
        self.completeness_score = 0.0
        self.has_answer_flag = False
        self.final_answer = None
        
    def copy(self) -> 'ReasoningState': # rollout的時候不要汙染現在環境
        """深度复制状态"""
        new_state = ReasoningState()
        
        # 复制基本信息
        new_state.query = self.query
        new_state.table_caption = self.table_caption
        new_state.markdown_table = self.markdown_table
        new_state.dataset = self.dataset
        new_state.graph_retriever = self.graph_retriever
        new_state.model = self.model
        new_state.args = self.args
        
        # 深度复制状态信息
        new_state.step = self.step
        new_state.retrieved_cells = copy.deepcopy(self.retrieved_cells)
        new_state.retrieved_cell_ids = copy.deepcopy(self.retrieved_cell_ids)
        new_state.explored_neighbors = copy.deepcopy(self.explored_neighbors)
        new_state.reasoning_path = copy.deepcopy(self.reasoning_path)
        new_state.intermediate_results = copy.deepcopy(self.intermediate_results)
        new_state.search_cell_history_dict = copy.deepcopy(self.search_cell_history_dict)
        new_state.hit_cell_neighbors_history_dict = copy.deepcopy(self.hit_cell_neighbors_history_dict)
        
        new_state.start_cells = self.start_cells
        new_state.connect_graph_prompt = self.connect_graph_prompt
        new_state.conn_graph_result = copy.deepcopy(self.conn_graph_result)
        new_state.reasoning_path_dict = copy.deepcopy(self.reasoning_path_dict)
        new_state.reasoning_path_prompt = self.reasoning_path_prompt
        
        new_state.confidence_score = self.confidence_score
        new_state.completeness_score = self.completeness_score
        new_state.has_answer_flag = self.has_answer_flag
        new_state.final_answer = self.final_answer
        
        return new_state
    
    def is_terminal(self) -> bool:
        """判断是否为终端状态"""
        # 1. 已经有答案
        if self.has_answer_flag:
            return True
        
        # 2. 超过最大步数
        max_steps = getattr(self.args, 'max_iteration_depth', 5)
        if self.step > max_steps:
            return True
        
        # 3. 置信度足够高且有足够信息
        if self.confidence_score > 0.9 and self.completeness_score > 0.8:
            return True
        
        return False
    
    def get_possible_actions(self) -> List[Tuple[str, Any]]:
        """获取当前状态下可能的动作
        回傳像 [('SearchNode', ['percent']), ('GetAllNeighbours', [(r,c,content)]), ...] 的 (動作名稱, 參數) 兩元組。"""

        actions = []
        
        # 1. SearchNode动作 - 根据问题语义搜索相关单元格
        if len(self.retrieved_cells) < 8:  # 限制搜索的单元格数量 NOTE:應該有參數放在其他文件可以改
            # 生成搜索查询
            search_queries = self._generate_search_queries()
            for query in search_queries[:2]:  # 限制同时考虑的搜索数量 最多只取2個關鍵詞
                actions.append(('SearchNode', [query]))
        
        # 2. GetAllNeighbours动作 - 探索已知单元格的邻居
        for i, cell in enumerate(self.retrieved_cells):
            if i not in self.explored_neighbors or not self.explored_neighbors[i].get('all_neighbors', False):
                if len(self.retrieved_cell_ids) > i:
                    cell_tuple = self._id_to_tuple(self.retrieved_cell_ids[i])
                    actions.append(('GetAllNeighbours', [cell_tuple]))
        
        # 3. GetSharedNeighbours动作 - 查找共同邻居
        if len(self.retrieved_cells) >= 2:
            # 选择最相关的单元格对
            for i in range(min(3, len(self.retrieved_cells))):
                for j in range(i+1, min(3, len(self.retrieved_cells))):
                    if len(self.retrieved_cell_ids) > j:
                        cell1_tuple = self._id_to_tuple(self.retrieved_cell_ids[i])
                        cell2_tuple = self._id_to_tuple(self.retrieved_cell_ids[j])
                        pair_key = f"{i}-{j}"
                        if pair_key not in self.explored_neighbors:
                            actions.append(('GetSharedNeighbours', [cell1_tuple, cell2_tuple]))
        
        # 4. Answer动作 - 如果信息足够，给出答案
        if self._should_try_answer():
            actions.append(('Answer', []))
        
        return actions
    
    def _generate_search_queries(self) -> List[str]:
        """生成搜索查询
        最後回傳關鍵字列表 例如["average","mean","salary","cook"]"""

        queries = []
        
        # 基于问题关键词生成查询
        query_lower = self.query.lower()
        
        # 提取问题中的关键概念
        if 'percent' in query_lower or '%' in query_lower or 'percentage' in query_lower:
            queries.append('percent')
            queries.append('percentage')
        
        if 'total' in query_lower or 'sum' in query_lower:
            queries.append('total')
            queries.append('sum')
        
        if 'average' in query_lower or 'mean' in query_lower:
            queries.append('average')
            queries.append('mean')
        
        # 提取实体词
        words = query_lower.split()
        for word in words:
            if len(word) > 3 and word not in ['what', 'how', 'when', 'where', 'which', 'the', 'and', 'or']:
                queries.append(word)
        
        return queries[:5]  # 限制查询数量
    
    def _id_to_tuple(self, cell_id: int) -> Tuple[int, int, str]:
        """将单元格ID转换为元组格式
        傳入(格id)，回傳(行,列,值)"""

        if hasattr(self.graph_retriever, 'dealed_rows'):
            row_width = len(self.graph_retriever.dealed_rows[0]) # 第一行的元素個數
            row_id = cell_id // row_width
            col_id = cell_id % row_width
            content = self.graph_retriever.dealed_rows[row_id][col_id]
            return (row_id, col_id, str(content))
        return (0, 0, "")
    
    def _should_try_answer(self) -> bool:
        """判断是否应该尝试给出答案"""
        # 基于多个因素决定
        factors = [
            self.confidence_score > 0.7,
            self.completeness_score > 0.6,
            len(self.retrieved_cells) >= 3,
            self.step >= 3
        ]
        return sum(factors) >= 2 # 至少滿足兩個條件以上
    
    def has_answer(self) -> bool:
        """是否已有答案"""
        return self.has_answer_flag
    
    def calculate_completeness(self) -> float:
        """计算信息完整性分数
        NOTE: 這裡參數by經驗，或許可調"""

        score = 0.0
        
        # 基于检索到的单元格数量
        cell_score = min(1.0, len(self.retrieved_cells) / 6)
        score += cell_score * 0.4
        
        # 基于探索的邻居关系
        neighbor_score = min(1.0, len(self.explored_neighbors) / 4)
        score += neighbor_score * 0.3
        
        # 基于推理步数
        step_score = min(1.0, self.step / 5)
        score += step_score * 0.3
        
        self.completeness_score = score
        return score


class MCTSReasoner:
    """基于MCTS的图推理器"""
    
    def __init__(self, original_reasoner, event_logger=None):
        """从原始推理器初始化"""
        self.original_reasoner = original_reasoner
        
        # MCTS超参数
        self.c_puct = 1.4  # UCB探索系数
        self.max_simulations = 50  # 最大模拟次数
        self.max_depth = 8  # 最大搜索深度
        self.simulation_timeout = 30  # 模拟超时(秒)
        
        # 统计信息
        self.stats = {
            'total_simulations': 0,
            'successful_simulations': 0,
            'average_depth': 0,
            'best_reward': 0.0
        }
        # 记录完整动作事件
        self.event_logger = event_logger

    def search(self) -> str:
        """MCTS主搜索循环"""
        print("开始MCTS搜索...")
        
        # 初始化根节点
        initial_state = self._initialize_state()
        root = MCTSNode(initial_state)
        self._log("root", node_id=id(root), actions=root.state.get_possible_actions()) # 记录建立根节点

        start_time = time.time()
        
        for simulation in range(self.max_simulations):
            if time.time() - start_time > self.simulation_timeout:
                print(f"搜索超时，完成{simulation}次模拟")
                break
            
            print(f"第{simulation+1}次模拟...")
            
            # MCTS四个阶段
            leaf_node = self._select(root)
            self._log("select", node_id=id(leaf_node))  # 记录选到节点

            expanded_node = self._expand(leaf_node)
            target_node = expanded_node if expanded_node else leaf_node

            reward, depth = self._simulate(target_node)
            self._backpropagate(target_node, reward)
            
            self.stats['total_simulations'] += 1
            if reward > 0.5:
                self.stats['successful_simulations'] += 1
            
            # 早停检查
            if self._should_early_stop(root, simulation):
                print(f"早停，完成{simulation+1}次模拟")
                break
        
        # 选择最佳路径
        best_path = self._get_best_path(root)
        path_ids = [id(root)] + [id(node) for node in best_path]
        self._log("best_path", path=path_ids)
        return self._execute_path(best_path)
    
    def _initialize_state(self) -> ReasoningState:
        """初始化推理状态"""
        # 调用原始推理器的初始化方法
        self.original_reasoner.initialize_prompt()
        
        # 创建MCTS状态
        state = ReasoningState(self.original_reasoner)
        
        # 复制初始化信息
        state.start_cells = self.original_reasoner.start_cells
        state.connect_graph_prompt = self.original_reasoner.connect_graph_prompt
        state.conn_graph_result = self.original_reasoner.conn_graph_result
        state.retrieved_cells = copy.deepcopy(self.original_reasoner.retrieved_cell)
        state.retrieved_cell_ids = copy.deepcopy(self.original_reasoner.retrieved_cell_id)
        state.hit_cell_neighbors_history_dict = copy.deepcopy(self.original_reasoner.hit_cell_neighbors_history_dict)
        
        # 计算初始分数
        state.calculate_completeness()
        state.confidence_score = 0.1  # 初始置信度较低
        
        return state
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """选择阶段：使用UCB选择最有希望的路径"""
        current = node
        
        while not current.is_leaf() and not current.is_terminal():
            # 第一次走到這個節點時，記下它所有可執行的動作
            if current.untried_actions is None:
                current.untried_actions = current.state.get_possible_actions()
            # 只要還有未展開的 action，就停在這個節點給 _expand 處理
            if current.untried_actions:
                return current

            # 否則照舊：已經全部展開過，就往訪問值最高的子節點走
            if not current.children:
                return current        

            # 使用自适应探索系数
            adaptive_c_puct = self._adaptive_exploration(current.state)
            # 计算所有子节点的UCB分数
            for child in current.children:
                child.ucb_score = child.calculate_ucb(adaptive_c_puct)
            
            
            # 选择UCB分数最高的子节点
            current = max(current.children, key=lambda x: x.ucb_score)
        
        return current

    def _expand(self, node: MCTSNode) -> Optional[MCTSNode]:
        if node.is_terminal():
            return None
        if node.untried_actions is None:
            node.untried_actions = node.state.get_possible_actions()
        if not node.untried_actions:
            node.is_expanded = True
            return None

        action = node.untried_actions.pop(0)
        new_state = self._apply_action(node.state, action)
        new_node = MCTSNode(new_state, parent=node, action=action)
        node.children.append(new_node)
        self._log("expand", parent_id=id(node), node_id=id(new_node), action=action)

        if not node.untried_actions:
            node.is_expanded = True
        return new_node

    # def _expand(self, node: MCTSNode) -> Optional[MCTSNode]:
    #     """扩展阶段：为节点添加新的子节点"""
    #     if node.is_terminal() or node.is_expanded:
    #         return None
        
    #     possible_actions = node.state.get_possible_actions()
        
    #     if not possible_actions:
    #         node.is_expanded = True
    #         return None
        
    #     # 选择第一个未尝试的动作（后续可以优化为更智能的选择）
    #     for action in possible_actions:
    #         # 检查这个动作是否已经被尝试过
    #         action_exists = any(child.action == action for child in node.children)
    #         if not action_exists:
    #             # 应用动作创建新状态
    #             new_state = self._apply_action(node.state, action)
    #             new_node = MCTSNode(new_state, parent=node, action=action)
    #             node.children.append(new_node)
    #             return new_node
        
    #     node.is_expanded = True
    #     return None
    
    def _simulate(self, node: MCTSNode) -> float:
        """模拟阶段：从当前节点随机模拟到终端状态"""
        current_state = node.state.copy()
        depth = 0
        
        while not current_state.is_terminal() and depth < self.max_depth:
            possible_actions = current_state.get_possible_actions()
            if not possible_actions:
                break
            
            # 使用简单的启发式策略选择动作
            action = self._simulation_policy(current_state, possible_actions)
            current_state = self._apply_action(current_state, action)
            depth += 1
        
        # 计算最终奖励
        reward = self._calculate_reward(current_state)
        self._log("simulate", node_id=id(node), reward=reward, depth=depth)
        return reward, depth
    
    def _backpropagate(self, node: MCTSNode, reward: float):
        """反向传播阶段：将奖励传播到根节点"""
        current = node
        while current is not None:
            current.visits += 1
            current.total_reward += reward
            self._log("backprop", node_id=id(current), visits=current.visits, avg=current.get_average_reward()) # 记录反向传播
            current = current.parent
    
    def _apply_action(self, state: ReasoningState, action: Tuple) -> ReasoningState:
        """应用动作到状态，返回新状态"""
        new_state = state.copy()
        action_type, parameters = action
        
        new_state.step += 1
        
        try:
            if action_type == 'SearchNode':
                self._apply_search_action(new_state, parameters)
            elif action_type == 'GetAllNeighbours':
                self._apply_neighbors_action(new_state, parameters)
            elif action_type == 'GetSharedNeighbours':
                self._apply_shared_neighbors_action(new_state, parameters)
            elif action_type == 'Answer':
                self._apply_answer_action(new_state)
        except Exception as e:
            print(f"动作应用失败: {action_type}, 错误: {e}")
            # 即使失败也要更新状态
            pass
        
        # 更新评估分数
        new_state.calculate_completeness()
        new_state.confidence_score = min(1.0, new_state.confidence_score + 0.1)
        
        return new_state
    
    def _apply_search_action(self, state: ReasoningState, parameters: List):
        """应用搜索动作"""
        if not parameters:
            return
        
        query = parameters[0]
        try:
            search_cells, search_cell_ids, search_cell_tuples = state.graph_retriever.search_cell(
                query=query, topk=1
            )
            
            if search_cell_ids and len(search_cell_ids) > 0:
                cell_id = search_cell_ids[0]
                cell_content = search_cells[0]
                
                state.search_cell_history_dict[cell_id] = {query: cell_content}
                
                if cell_content not in state.retrieved_cells:
                    state.retrieved_cells.append(cell_content)
                    state.retrieved_cell_ids.append(cell_id)
                    
                    # 更新推理路径
                    self._update_reasoning_path(state, 'SearchNode', [query], f"找到单元格: {cell_content}")
                    
                    # 更新图结构
                    self._update_graph_structure(state)
                    
        except Exception as e:
            print(f"搜索动作执行失败: {e}")
    
    def _update_reasoning_path(self, state: ReasoningState, action_type: str, parameters: List, result: str):
        """更新推理路径"""
        # 更新reasoning_path_dict
        if state.step not in state.reasoning_path_dict:
            state.reasoning_path_dict[state.step] = []
        
        action_info = [action_type, '|'.join([str(p) for p in parameters])]
        state.reasoning_path_dict[state.step].append(action_info)
        
        # 更新reasoning_path
        action_desc = f"{action_type}({', '.join([str(p) for p in parameters])})"
        state.reasoning_path.append(action_desc)
        
        # 更新中间结果
        state.intermediate_results[state.step] = {
            'think': f"执行{action_type}动作",
            'action': f"Action Step {state.step}: {action_desc}",
            'interaction_prompt': f"Observation Step {state.step}: {result}"
        }
    
    def _update_graph_structure(self, state: ReasoningState):
        """更新图结构信息"""
        try:
            # 重新生成连接图
            hit_cell_same_row_col = state.graph_retriever.getSameRow_ColCells(state.retrieved_cell_ids)
            _, _, state.conn_graph_result = state.graph_retriever.hit_cell_connect_graph(hit_cell_same_row_col)
            
            # 更新连接图prompt
            from configs import PROMPT_TEMPLATE
            state.connect_graph_prompt = PROMPT_TEMPLATE[state.dataset]['connect_graph'].replace(
                '{sub_graph}', state.conn_graph_result
            )
            
        except Exception as e:
            print(f"图结构更新失败: {e}")
    
    def _apply_neighbors_action(self, state: ReasoningState, parameters: List):
        """应用获取邻居动作"""
        if not parameters:
            return
        
        try:
            cell_tuple = parameters[0]
            # 转换为cell_id
            row_id, col_id = cell_tuple[0], cell_tuple[1]
            cell_id = len(state.graph_retriever.dealed_rows[0]) * row_id + col_id
            
            cell_topk, nei_cells, hit_cell_neighbors_content_id = state.graph_retriever.get_neighbors(
                add_id_list=[cell_id],
                get_same_row=False,
                get_all_nei=True
            )
            
            # 更新邻居历史
            state.hit_cell_neighbors_history_dict[cell_id] = hit_cell_neighbors_content_id
            
            # 标记已探索
            for i, cid in enumerate(state.retrieved_cell_ids):
                if cid == cell_id:
                    if i not in state.explored_neighbors:
                        state.explored_neighbors[i] = {}
                    state.explored_neighbors[i]['all_neighbors'] = True
                    break
            
            # 更新推理路径
            result_desc = f"找到{len(hit_cell_neighbors_content_id)}个邻居单元格"
            self._update_reasoning_path(state, 'GetAllNeighbours', [cell_tuple], result_desc)
            
            # 将新发现的邻居添加到检索列表中
            for neighbor_id, neighbor_content in hit_cell_neighbors_content_id.items():
                if neighbor_content not in state.retrieved_cells and len(state.retrieved_cells) < 10:
                    state.retrieved_cells.append(neighbor_content)
                    state.retrieved_cell_ids.append(neighbor_id)
            
            # 更新图结构
            self._update_graph_structure(state)
                    
        except Exception as e:
            print(f"获取邻居动作执行失败: {e}")
    
    def _apply_shared_neighbors_action(self, state: ReasoningState, parameters: List):
        """应用获取共同邻居动作"""
        if len(parameters) < 2:
            return
        
        try:
            cell1_tuple, cell2_tuple = parameters[0], parameters[1]
            
            # 转换为cell_id
            row1, col1 = cell1_tuple[0], cell1_tuple[1]
            row2, col2 = cell2_tuple[0], cell2_tuple[1]
            
            cell1_id = len(state.graph_retriever.dealed_rows[0]) * row1 + col1
            cell2_id = len(state.graph_retriever.dealed_rows[0]) * row2 + col2
            
            # 获取同行同列信息
            hit_cell_same_row_col = state.graph_retriever.getSameRow_ColCells([cell1_id, cell2_id])
            _, connect_id_cell_dict, cell_shared_neighbors = state.graph_retriever.hit_cell_connect_graph(
                hit_cell_same_row_col, get_shared_nei=True
            )
            
            # 更新邻居历史
            state.hit_cell_neighbors_history_dict[cell1_id] = connect_id_cell_dict
            
            # 找到对应的retrieved_cell索引并标记已探索
            cell1_index = None
            cell2_index = None
            for i, cid in enumerate(state.retrieved_cell_ids):
                if cid == cell1_id:
                    cell1_index = i
                elif cid == cell2_id:
                    cell2_index = i
            
            if cell1_index is not None and cell2_index is not None:
                pair_key = f"{min(cell1_index, cell2_index)}-{max(cell1_index, cell2_index)}"
                state.explored_neighbors[pair_key] = True
            
            # 更新推理路径
            shared_count = len(connect_id_cell_dict)
            result_desc = f"找到{shared_count}个共同邻居"
            self._update_reasoning_path(state, 'GetSharedNeighbours', [cell1_tuple, cell2_tuple], result_desc)
            
            # 将新发现的共同邻居添加到检索列表中
            for neighbor_id, neighbor_content in connect_id_cell_dict.items():
                if neighbor_content not in state.retrieved_cells and len(state.retrieved_cells) < 10:
                    state.retrieved_cells.append(neighbor_content)
                    state.retrieved_cell_ids.append(neighbor_id)
            
            # 更新图结构
            self._update_graph_structure(state)
                
        except Exception as e:
            print(f"获取共同邻居动作执行失败: {e}")
    
    def _apply_answer_action(self, state: ReasoningState):
        """应用回答动作"""
        try:
            # 构建思考文本
            thinking_text = self._build_thinking_text(state)
            
            # 调用原始推理器的Answer方法
            answer = self.original_reasoner.Answer(
                thinking_text=thinking_text,
                answer_explan="基于MCTS搜索的推理结果",
                last_interact_step=""
            )
            
            # 处理答案格式
            if isinstance(answer, list):
                state.final_answer = ', '.join([str(i) for i in answer])
            else:
                state.final_answer = str(answer)
            
            state.has_answer_flag = True
            
        except Exception as e:
            print(f"答案生成失败: {e}")
            # 使用备用答案生成方法
            state.final_answer = self._generate_fallback_answer(state)
            state.has_answer_flag = True
    
    def _build_thinking_text(self, state: ReasoningState) -> str:
        """构建思考文本"""
        thinking_parts = []
        
        # 基本信息
        thinking_parts.append(f"基于MCTS搜索，经过{state.step}步推理")
        
        # 检索到的单元格信息
        if state.retrieved_cells:
            thinking_parts.append(f"检索到{len(state.retrieved_cells)}个相关单元格：")
            for i, cell in enumerate(state.retrieved_cells[:5]):  # 限制显示数量
                thinking_parts.append(f"- {cell}")
        
        # 探索的邻居关系
        if state.explored_neighbors:
            thinking_parts.append(f"探索了{len(state.explored_neighbors)}个邻居关系")
        
        # 推理路径
        if state.reasoning_path:
            thinking_parts.append("推理路径：")
            for i, step in enumerate(state.reasoning_path[:3]):  # 限制显示数量
                thinking_parts.append(f"步骤{i+1}: {step}")
        
        return "\n".join(thinking_parts)
    
    def _generate_fallback_answer(self, state: ReasoningState) -> str:
        """生成备用答案"""
        if not state.retrieved_cells:
            return "无法找到相关信息"
        
        # 简单的答案生成逻辑
        query_lower = state.query.lower()
        
        # 数值型问题
        if self._is_numerical_question(state.query):
            # 查找数值
            for cell in state.retrieved_cells:
                cell_str = str(cell)
                if self._contains_number(cell_str):
                    # 提取数字
                    import re
                    numbers = re.findall(r'\d+\.?\d*', cell_str)
                    if numbers:
                        return numbers[0]
        
        # 返回最相关的单元格内容
        for cell in state.retrieved_cells:
            cell_lower = str(cell).lower()
            for keyword in self._extract_keywords(state.query):
                if keyword.lower() in cell_lower:
                    return str(cell)
        
        # 默认返回第一个单元格
        return str(state.retrieved_cells[0])
    
    def _simulation_policy(self, state: ReasoningState, actions: List) -> Tuple:
        """智能模拟策略：在模拟阶段选择动作"""
        scored_actions = []
        
        for action in actions:
            score = self._calculate_action_score(state, action)
            scored_actions.append((action, score))
        
        # 根据分数选择（带随机性）
        scored_actions.sort(key=lambda x: x[1], reverse=True)
        
        # 使用温度调节的选择策略
        temperature = self._get_simulation_temperature(state)
        selected_action = self._temperature_based_selection(scored_actions, temperature)
        
        return selected_action
    
    def _calculate_action_score(self, state: ReasoningState, action: Tuple) -> float:
        """计算动作的启发式分数"""
        action_type, parameters = action
        base_score = 0.0
        
        if action_type == 'SearchNode':
            base_score = self._score_search_action(state, parameters)
        elif action_type == 'GetAllNeighbours':
            base_score = self._score_neighbors_action(state, parameters)
        elif action_type == 'GetSharedNeighbours':
            base_score = self._score_shared_neighbors_action(state, parameters)
        elif action_type == 'Answer':
            base_score = self._score_answer_action(state)
        
        return base_score
    
    def _score_search_action(self, state: ReasoningState, parameters: List) -> float:
        """评估搜索动作的分数"""
        if not parameters:
            return 0.1
        
        query = parameters[0]
        score = 0.6  # 基础分数
        
        # 如果检索的单元格太少，搜索动作优先级更高
        if len(state.retrieved_cells) < 3:
            score += 0.3
        
        # 如果查询与问题关键词匹配，优先级更高
        query_keywords = self._extract_keywords(state.query)
        if any(keyword.lower() in query.lower() for keyword in query_keywords):
            score += 0.2
        
        # 避免重复搜索相同的内容
        for cell in state.retrieved_cells:
            if query.lower() in str(cell).lower():
                score -= 0.2
                break
        
        return max(0.1, min(1.0, score))
    
    def _score_neighbors_action(self, state: ReasoningState, parameters: List) -> float:
        """评估邻居探索动作的分数"""
        if not parameters:
            return 0.1
        
        score = 0.5  # 基础分数
        
        # 如果已有足够的单元格，邻居探索优先级更高
        if len(state.retrieved_cells) >= 2:
            score += 0.2
        
        # 如果这个单元格还没有被探索过邻居
        try:
            cell_tuple = parameters[0]
            row_id, col_id = cell_tuple[0], cell_tuple[1]
            cell_id = len(state.graph_retriever.dealed_rows[0]) * row_id + col_id
            
            # 检查是否已探索
            for i, cid in enumerate(state.retrieved_cell_ids):
                if cid == cell_id:
                    if i not in state.explored_neighbors or not state.explored_neighbors[i].get('all_neighbors', False):
                        score += 0.3
                    break
        except:
            pass
        
        return max(0.1, min(1.0, score))
    
    def _score_shared_neighbors_action(self, state: ReasoningState, parameters: List) -> float:
        """评估共同邻居动作的分数"""
        if len(parameters) < 2:
            return 0.1
        
        score = 0.4  # 基础分数
        
        # 如果有多个单元格，共同邻居可能很有用
        if len(state.retrieved_cells) >= 3:
            score += 0.2
        
        # 检查这对单元格是否已被探索
        try:
            cell1_tuple, cell2_tuple = parameters[0], parameters[1]
            row1, col1 = cell1_tuple[0], cell1_tuple[1]
            row2, col2 = cell2_tuple[0], cell2_tuple[1]
            
            cell1_id = len(state.graph_retriever.dealed_rows[0]) * row1 + col1
            cell2_id = len(state.graph_retriever.dealed_rows[0]) * row2 + col2
            
            # 找到对应索引
            cell1_index = None
            cell2_index = None
            for i, cid in enumerate(state.retrieved_cell_ids):
                if cid == cell1_id:
                    cell1_index = i
                elif cid == cell2_id:
                    cell2_index = i
            
            if cell1_index is not None and cell2_index is not None:
                pair_key = f"{min(cell1_index, cell2_index)}-{max(cell1_index, cell2_index)}"
                if pair_key not in state.explored_neighbors:
                    score += 0.3
        except:
            pass
        
        return max(0.1, min(1.0, score))
    
    def _score_answer_action(self, state: ReasoningState) -> float:
        """评估回答动作的分数
        NOTE:可以優化"""

        # 基于多个因素综合评估
        score = 0.0
        
        # 信息完整性
        if len(state.retrieved_cells) >= 3:
            score += 0.3
        
        # 置信度
        score += state.confidence_score * 0.4
        
        # 推理步数（不要太早也不要太晚）
        ideal_steps = 4
        step_factor = 1.0 - abs(state.step - ideal_steps) / ideal_steps
        score += step_factor * 0.2
        
        # 探索程度
        if state.explored_neighbors:
            score += 0.1
        
        return max(0.1, min(1.0, score))
    
    def _get_simulation_temperature(self, state: ReasoningState) -> float:
        """获取模拟温度，控制探索vs利用的平衡"""
        # 早期模拟更多探索，后期更多利用
        base_temp = 1.0
        
        # 根据步数调整温度
        step_factor = max(0.3, 1.0 - state.step / self.max_depth)
        
        # 根据信息完整性调整温度
        completeness_factor = 1.0 - state.calculate_completeness()
        
        temperature = base_temp * step_factor * completeness_factor
        return max(0.1, min(2.0, temperature))
    
    def _temperature_based_selection(self, scored_actions: List, temperature: float) -> Tuple:
        """基于温度的动作选择"""
        if not scored_actions:
            return ('Answer', [])
        
        if temperature < 0.1:
            # 低温度：选择最佳动作
            return scored_actions[0][0]
        
        # 计算softmax概率
        import math
        scores = [score for _, score in scored_actions]
        max_score = max(scores)
        
        # 数值稳定性处理 溫度越大越平均 越隨機
        exp_scores = [math.exp((score - max_score) / temperature) for score in scores]
        total_exp = sum(exp_scores)
        
        if total_exp == 0:
            return random.choice(scored_actions)[0]
        
        probabilities = [exp_score / total_exp for exp_score in exp_scores]
        
        # 根据概率选择 (輪盤法)
        rand_val = random.random()
        cumulative_prob = 0.0
        
        for i, prob in enumerate(probabilities):
            cumulative_prob += prob
            if rand_val <= cumulative_prob:
                return scored_actions[i][0]
        
        # 备用选择
        return scored_actions[0][0]
    
    def _adaptive_exploration(self, state: ReasoningState) -> float:
        """自适应探索系数"""
        # 根据当前状态动态调整探索参数
        base_exploration = self.c_puct
        
        # 如果信息不足，增加探索
        if len(state.retrieved_cells) < 3:
            base_exploration *= 1.5
        
        # 如果接近最大深度，减少探索
        if state.step > self.max_depth * 0.8:
            base_exploration *= 0.7
        
        return base_exploration
    
    def _calculate_reward(self, state: ReasoningState) -> float:
        """计算状态的奖励值
        藉由調參得出"""

        reward = 0.0
        
        # 1. 信息完整性奖励 (40%)
        completeness = self._evaluate_completeness(state)
        reward += completeness * 0.4
        
        # 2. 推理路径质量奖励 (25%)
        path_quality = self._evaluate_path_quality(state)
        reward += path_quality * 0.25
        
        # 3. 答案相关性奖励 (20%)
        answer_relevance = self._evaluate_answer_relevance(state)
        reward += answer_relevance * 0.2
        
        # 4. 效率奖励 (10%)
        efficiency = self._evaluate_efficiency(state)
        reward += efficiency * 0.1
        
        # 5. 多样性奖励 (5%)
        diversity = self._evaluate_diversity(state)
        reward += diversity * 0.05
        
        return min(1.0, max(0.0, reward))
    
    def _evaluate_completeness(self, state: ReasoningState) -> float:
        """评估信息完整性"""
        score = 0.0
        
        # 基于检索到的单元格数量
        target_cells = min(8, len(state.query.split()) + 2)  # 根据问题复杂度调整目标
        cell_coverage = min(1.0, len(state.retrieved_cells) / target_cells)
        score += cell_coverage * 0.4
        
        # 基于邻居探索的完整性
        explored_ratio = 0.0
        if state.retrieved_cells:
            explored_count = len([i for i in state.explored_neighbors.values() if i])
            explored_ratio = min(1.0, explored_count / len(state.retrieved_cells))
        score += explored_ratio * 0.3
        
        # 基于图连接性
        connectivity_score = self._evaluate_graph_connectivity(state)
        score += connectivity_score * 0.3
        
        return score
    
    def _evaluate_path_quality(self, state: ReasoningState) -> float:
        """评估推理路径质量
        NOTE: ideal_length可做自適應"""

        if not state.reasoning_path:
            return 0.0
        
        score = 0.0
        
        # 路径长度适中性
        ideal_length = 4  # 理想推理步数
        length_penalty = abs(len(state.reasoning_path) - ideal_length) / ideal_length
        length_score = max(0, 1.0 - length_penalty)
        score += length_score * 0.4
        
        # 动作多样性
        action_types = set()
        for step in state.reasoning_path_dict.values():
            for action_info in step:
                if len(action_info) >= 2:
                    action_types.add(action_info[0])
        
        diversity_score = min(1.0, len(action_types) / 3)  # 期望至少3种不同动作
        score += diversity_score * 0.3
        
        # 逻辑连贯性（简化评估） 通常是拿滿的
        coherence_score = self._evaluate_coherence(state)
        score += coherence_score * 0.3
        
        return score
    
    def _evaluate_answer_relevance(self, state: ReasoningState) -> float:
        """评估答案相关性"""
        score = 0.0
        
        # 检查是否包含问题关键词相关的单元格
        query_keywords = self._extract_keywords(state.query)
        relevant_cells = 0
        
        for cell in state.retrieved_cells:
            cell_lower = str(cell).lower()
            for keyword in query_keywords:
                if keyword.lower() in cell_lower:
                    relevant_cells += 1
                    break
        
        if state.retrieved_cells:
            relevance_ratio = relevant_cells / len(state.retrieved_cells)
            score += relevance_ratio * 0.5
        
        # 检查是否包含数值型答案（对于计算类问题）
        if self._is_numerical_question(state.query):
            numerical_cells = sum(1 for cell in state.retrieved_cells if self._contains_number(str(cell)))
            if state.retrieved_cells:
                numerical_ratio = numerical_cells / len(state.retrieved_cells)
                score += numerical_ratio * 0.3
        
        # 如果已有答案，评估答案质量
        if state.has_answer():
            answer_quality = self._evaluate_answer_quality(state)
            score += answer_quality * 0.2
        
        return min(1.0, score)
    
    def _evaluate_efficiency(self, state: ReasoningState) -> float:
        """评估推理效率
        NOTE: 可以優化"""

        # 步数效率
        step_efficiency = max(0, 1.0 - (state.step - 1) / self.max_depth)
        
        # 信息获取效率（每步获取的有用信息）
        info_efficiency = 0.0
        if state.step > 1:
            info_per_step = len(state.retrieved_cells) / state.step
            info_efficiency = min(1.0, info_per_step / 2)  # 期望每步获取2个有用信息
        
        return (step_efficiency + info_efficiency) / 2
    
    def _evaluate_diversity(self, state: ReasoningState) -> float:
        """评估探索多样性
        通常1"""

        if not state.retrieved_cells:
            return 0.0
        
        # 单元格位置多样性
        positions = set()
        for cell_id in state.retrieved_cell_ids:
            if hasattr(state.graph_retriever, 'dealed_rows'):
                row_width = len(state.graph_retriever.dealed_rows[0])
                row_id = cell_id // row_width
                col_id = cell_id % row_width
                positions.add((row_id, col_id))
        
        # 计算位置分散程度
        position_diversity = len(positions) / len(state.retrieved_cell_ids) if state.retrieved_cell_ids else 0
        
        return position_diversity
    
    def _evaluate_graph_connectivity(self, state: ReasoningState) -> float:
        """评估图连接性"""
        if len(state.retrieved_cell_ids) < 2:
            return 0.0
        
        # 简化的连接性评估：检查是否有邻居关系被探索
        connected_pairs = len(state.explored_neighbors)
        max_possible_pairs = len(state.retrieved_cell_ids) * (len(state.retrieved_cell_ids) - 1) // 2  # 簡單判斷C(n,2)
        
        if max_possible_pairs > 0:
            return min(1.0, connected_pairs / max_possible_pairs)
        return 0.0
    
    def _evaluate_coherence(self, state: ReasoningState) -> float:
        """评估推理连贯性"""
        # 简化评估：检查动作序列是否合理
        if not state.reasoning_path_dict:
            return 1.0
        
        # 检查是否先搜索再探索邻居
        has_search = False
        has_neighbor_after_search = False
        
        for step_actions in state.reasoning_path_dict.values():
            for action_info in step_actions:
                if len(action_info) >= 2:
                    action_type = action_info[0]
                    if action_type == 'SearchNode':
                        has_search = True
                    elif action_type in ['GetAllNeighbours', 'GetSharedNeighbours'] and has_search:
                        has_neighbor_after_search = True
        
        coherence_score = 0.5  # 基础分数
        if has_search:
            coherence_score += 0.25
        if has_neighbor_after_search:
            coherence_score += 0.25
        
        return coherence_score
    
    def _evaluate_answer_quality(self, state: ReasoningState) -> float:
        """评估答案质量
        NOTE: 粗淺評估有答案就好"""
        if not state.final_answer:
            return 0.0
        
        # 简化评估：检查答案是否非空且不是错误消息
        answer_str = str(state.final_answer).lower()
        
        if any(error_word in answer_str for error_word in ['错误', 'error', '失败', 'fail', '无法', 'cannot']):
            return 0.1
        
        if len(answer_str.strip()) > 0:
            return 0.8
        
        return 0.0
    
    def _extract_keywords(self, query: str) -> List[str]:
        """提取问题关键词"""
        # 简化的关键词提取
        stop_words = {'what', 'how', 'when', 'where', 'which', 'who', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being'}
        words = query.lower().split()
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        return keywords[:5]  # 返回前5个关键词
    
    def _is_numerical_question(self, query: str) -> bool:
        """判断是否为数值型问题"""
        numerical_indicators = ['percent', 'percentage', '%', 'number', 'count', 'total', 'sum', 'average', 'mean', 'many', 'much']
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in numerical_indicators)
    
    def _contains_number(self, text: str) -> bool:
        """检查文本是否包含数字"""
        import re
        return bool(re.search(r'\d', text))
    
    def _should_early_stop(self, root: MCTSNode, simulation: int) -> bool:
        """判断是否应该早停"""
        # 如果找到高质量解
        if root.get_average_reward() > 0.8 and root.visits > 10:
            return True
        
        # 如果所有路径都被充分探索
        if simulation > 20 and root.get_average_reward() < 0.3:
            return True
        
        return False
    
    def _get_best_path(self, root: MCTSNode) -> List[MCTSNode]:
        """获取最佳路径"""
        path = []
        current = root
        
        while current.children:
            # 选择访问次数最多且奖励最高的子节点
            best_child = max(current.children, key=lambda x: (x.visits, x.get_average_reward()))
            path.append(best_child)
            current = best_child
        
        return path
    
    def _execute_path(self, path: List[MCTSNode]) -> str:
        """执行最佳路径并返回最终答案"""
        if not path:
            return "无法找到有效的推理路径"
        
        final_node = path[-1]
        final_state = final_node.state
        
        if final_state.has_answer():
            return str(final_state.final_answer)
        
        # 如果没有答案，尝试基于当前状态生成答案
        try:
            # 使用原始推理器的答案生成逻辑
            thinking_text = f"基于MCTS搜索，经过{len(path)}步推理"
            answer = self.original_reasoner.Answer(thinking_text)
            return ', '.join([str(i) for i in answer]) if isinstance(answer, list) else str(answer)
        except Exception as e:
            print(f"答案生成失败: {e}")
            return "推理完成，但无法生成最终答案" 
        
    def _log(self, event_type, **payload):
        if self.event_logger:
            self.event_logger.write(event_type, **payload)

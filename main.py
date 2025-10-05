
from configs import GEMINI_KEYS,hyperparameter
from GraphRetriever.graph_retriver import GraphRetriever

from Generator.openai_api import ChatGPTTool
from GraphRetriever.dense_retriever import load_dense_retriever
from Generator.Gemini_model import GeminiTool
# from vertexai.preview import tokenization
import argparse

import tiktoken
from dashscope import get_tokenizer
from iterative_reasoning import GraphReasoner
from mcts_adapter import MCTSGraphReasoner  # 导入MCTS适配器
from compute_score import eval_ex_match,LLM_eval
import json
import jsonlines

global tokenizer
logger = None

def augments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default='qwen2-7b-instruct')
    parser.add_argument("--base_url", type=str, default='https://dashscope.aliyuncs.com/compatible-mode/v1')
    parser.add_argument("--key", type=str, default='')
    parser.add_argument("--dataset", type=str, default='ait-qa')
    parser.add_argument("--qa_path", type=str, default='dataset/AIT-QA/aitqa_clean_questions.json')
    parser.add_argument("--table_folder", type=str, default='')

    parser.add_argument("--max_iteration_depth", type=int, default=5)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=100)  # -1表示运行到数据集末尾

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--embed_cache_dir", type=str, default='dataset/AIT-QA/')
    parser.add_argument("--embedder_path", type=str, default='thenlper/gte-base')

    parser.add_argument("--debug", type=bool, default=False)
    
    # MCTS相关参数
    parser.add_argument("--use_mcts", type=bool, default=True, help="是否使用MCTS推理")
    parser.add_argument("--mcts_c_puct", type=float, default=0.8, help="MCTS UCB探索系数(调优最优值)")
    parser.add_argument("--mcts_max_simulations", type=int, default=80, help="MCTS最大模拟次数(调优最优值)")
    parser.add_argument("--mcts_max_depth", type=int, default=12, help="MCTS最大搜索深度(调优最优值)")
    parser.add_argument("--mcts_timeout", type=int, default=20, help="MCTS模拟超时(秒)(调优最优值)")
    
    args = parser.parse_args()

    return args

def load_model(args):
    if 'gemini' in args.model:
        gemini_key_index = 0
        gemini_key = GEMINI_KEYS[gemini_key_index]
        model = GeminiTool(gemini_key, args)
    else:
        model = ChatGPTTool(args)

    dense_retriever = load_dense_retriever(args)

    return  model,dense_retriever

def load_data(args):

    querys,answers,table_captions,tables,table_paths = [],[],[],[],[]

    if args.dataset in ('hitab','Hitab'):
        qas = []
        with open(args.qa_path, "r+", encoding='utf-8') as f:
            for item in jsonlines.Reader(f):
                qas.append(item)
        
        # 自动检测数据集大小
        if args.end == -1:
            args.end = len(qas)
            print(f"检测到HiTab数据集共有 {len(qas)} 个样本，运行范围：{args.start}-{args.end}")
        
        qas = qas[args.start:args.end]

        for qa in qas:
            table_path = args.table_folder + qa['table_id'] + '.json'
            with open(table_path, "r+", encoding='utf-8') as f:
                table = json.load(f)
            table_captions.append(table['title'])
            answers.append('|'.join([str(i) for i in qa['answer']]))
            querys.append( qa['question'])
            table_paths.append(table_path)
            tables.append(table['texts'])
    elif args.dataset in ('AIT-QA','ait-qa'):
        with open(args.qa_path, 'r', encoding='utf-8') as f:
            qas = json.load(f)
        
        # 自动检测数据集大小
        if args.end == -1:
            args.end = len(qas)
            print(f"检测到AIT-QA数据集共有 {len(qas)} 个样本，运行范围：{args.start}-{args.end}")
        
        qas = qas[args.start:args.end]

        for qa in qas:
            tables.append(qa['table'])
            answers.append('|'.join([str(i) for i in qa['answers']]))
            querys.append( qa['question'])
            table_captions.append('')
            table_paths.append(qa)
    return querys,answers,table_captions,tables,table_paths


def main():
    args = augments()

    model, dense_retriever = load_model(args)
    querys, answers, table_captions, tables,table_paths = load_data(args)

    total_num,EM,LLM_EVAL = 0,0,0
    for query,answer,caption,table,table_path in zip(querys, answers, table_captions, tables,table_paths):
        unsafe = False
        error = 3
        graph_retriever = GraphRetriever(table_path, model, dense_retriever, args.dataset, table_cation=caption)
        
        # 根据配置选择推理器
        if args.use_mcts:
            graph_reasoner = MCTSGraphReasoner(args, model, query, table, caption, graph_retriever, args.dataset)
        else:
            graph_reasoner = GraphReasoner(args, model, query, table, caption, graph_retriever, args.dataset)

        output = graph_reasoner.iterative_reasoning()
        # while error >0:
        #     try:
        #         output = graph_reasoner.iterative_reasoning()
        #         break
        #     except UserWarning as v:
        #         print(query+ '\t' + '\t'+args.dataset+ '\t' +'不满足安全协议' + v.__str__()) # 没有通过gemini 的安全协议
        #         unsafe = True
        #         break
        #     except Exception as e:
        #         print(query+ '\t' +'报错了'+ '\t' +e.__str__())
        #         error -= 1
        #         continue
        # if unsafe or error <= 0:
        #     continue

        print('模型回答为',output,'答案为',answer)
        total_num += 1
        EM += eval_ex_match(output,answer)
        LLM_EVAL += LLM_eval(model,query,output,answer)
    print('EM:',EM/total_num)
    print('LLM EVAL:',LLM_EVAL/total_num)




if __name__ == '__main__':
    main()
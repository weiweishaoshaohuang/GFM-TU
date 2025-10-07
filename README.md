# GFM-TU: 图结构增强的大型语言模型表格理解

> 表格 → 图结构 → 轻量对齐与混合检索 → CoT / MCTS 推理 → Value Model（过程评分与剪枝）

**GFM-TU** 是一个面向复杂表格问答（Table QA）的研究型原型系统。我们将表格显式转换为图结构（单元格为节点、上下左右为主干边，必要时补充表头/行列弱类型），用紧凑的节点描述子与统一模板把图信息注入到 LLM 的推理循环中；在推理层保留 CoT 并引入 MCTS 做多路径探索；在评分层从启发式奖励演进到 **Value Model**（VM），对中间步骤和候选路径进行过程级打分与剪枝。工程侧提供模块化组件与终端可追踪轨迹，便于复核与扩展。

> 状态：研究原型（research prototype）。脚本与配置供参考，可能需要根据你的环境做少量调整。

---

## ✨ 主要特性

- **显式图结构建模**：单元格→节点，四邻接为主干；可选 Header→Data / 同行同列等语义边。  
- **图—文本轻量对齐**：节点描述子（坐标、表头链路摘要、文本/数值/单位/时间）注入统一 Prompt 模板。  
- **混合检索构造候选子图**：BM25 + 稠密向量 + 图启发扩展。  
- **两类推理范式**：CoT（基线）与 MCTS（选择/扩展/仿真/回传）。  
- **Value Model（VM）**：替代启发式奖励，用于 rollout 打分与剪枝，提高一致性与鲁棒性。  
- **参数高效微调（PEFT）**：在开源基座上采用 LoRA 进行轻量适配。  
- **可追踪输出**：终端打印完整推理轨迹与统计，用于误差分析与复核。

---

## 📦 代码结构

```
GFM-TU/
├─ table_reader.py            # 表格读取与单元格解析（表→图、节点描述子）
├─ BM25_Retriever.py          # BM25 检索器
├─ dense_retriever.py         # 稠密向量检索（FAISS 等）
├─ graph_retriver.py          # 图启发扩展（同行/同列/k-hop）
├─ rank_bm25.py               # BM25 实现（第三方封装/适配）
├─ iterative_reasoning.py     # 交互式推理骨架（状态—候选—动作—结果）
├─ mcts_vm.py                 # MCTS + VM 主循环
├─ mcts_adapter.py            # 推理器与评分器的适配层
├─ mcts_config.py             # MCTS / 检索等关键超参配置
├─ mcts_tuning.py             # 搜索/超参调优脚本（可选）
├─ value_models.py            # VM 模型结构与推理接口
├─ generate_vm_dataset.py     # 从日志/轨迹自举生成 VM 训练样本
├─ train_VM.py                # 训练 Value Model
├─ openai_api.py              # 开源/闭源后端统一接口（OpenAI 风格）
├─ Gemini_model.py            # Gemini 1.5 封装
├─ parse_output.py            # 健壮解析（JSON/列表/异常回退）
├─ main.py                    # 统一入口（CoT / MCTS / MCTS+VM）
├─ aitqa_clean_tables.json    # AIT-QA 清洗子集（表结构）
├─ aitqa_clean_questions.json # AIT-QA 问题集合
├─ train_samples.jsonl        # 训练样例（示例/占位）
└─ test_samples.jsonl         # 测试样例（示例/占位）
```

---

## 🧩 环境与依赖（参考）

- Python 3.10+
- PyTorch / Transformers / `peft` / `accelerate`  
- FAISS-cpu、rank_bm25、jieba、pandas、numpy、scikit-learn、tqdm、rich、pydantic
- 如需在线模型：`OPENAI_API_KEY` 或 `GEMINI_API_KEY`

创建虚拟环境并安装依赖（示例）：
```bash
conda create -n gfm-tu python=3.10 -y
conda activate gfm-tu
pip install torch transformers sentencepiece peft accelerate
pip install faiss-cpu rank_bm25 jieba numpy pandas scikit-learn tqdm rich pydantic
# 如需在线推理：
pip install openai google-generativeai
```

环境变量（按需）：
```bash
export OPENAI_API_KEY=xxx      # 使用 OpenAI 风格接口时
export GEMINI_API_KEY=xxx      # 使用 Gemini 1.5 时
```

---

## 🚀 快速开始（参考流程）

### 1）数据准备
将目标数据（HiTab / AIT-QA）路径指向仓库示例或你本地的数据。`table_reader.py` 会将表格转换为图并生成节点描述子。

### 2）仅推理（CoT / MCTS / MCTS+VM）
```bash
# CoT 基线
python main.py \
  --backend qwen2 \
  --mode cot \
  --dataset aitqa \
  --samples test_samples.jsonl

# MCTS（启发式）
python main.py \
  --backend qwen2 \
  --mode mcts \
  --config mcts_config.py \
  --dataset hitab

# MCTS + VM（过程评分与剪枝）
python main.py \
  --backend qwen2 \
  --mode mcts_vm \
  --config mcts_config.py \
  --vm_ckpt checkpoints/vm.pt \
  --dataset aitqa
```

运行后终端将打印：候选命中、节点选择、rollout 得分、回传统计、最终答案，以及 EM / LLM Eval 的统计值。可选将日志保存到 `outputs/` 用于复核。

### 3）训练 Value Model（可选）
```bash
# 3.1 从历史轨迹/日志自举生成训练样本
python generate_vm_dataset.py \
  --traces_dir outputs/traces \
  --out vm_train.jsonl

# 3.2 训练 VM（pairwise/回归 任选）
python train_VM.py \
  --data vm_train.jsonl \
  --base qwen2 \
  --epochs 3 \
  --save_dir checkpoints/

# 3.3 推理加载 VM
python main.py --mode mcts_vm --vm_ckpt checkpoints/vm.pt ...
```

### 4）参数高效微调（LoRA，开源基座）
```bash
# 仅示意：可将 LoRA 层与学习率等写入 config
accelerate launch train_lora.py \
  --model qwen2 \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.1 \
  --train_file train_samples.jsonl \
  --save_dir checkpoints/lora
```

> 注：以上命令为示意/参考。实际可运行选项与默认值以代码为准；必要时修改 `mcts_config.py` 与脚本参数。

---

## ⚙️ 关键配置（示例）

在 `mcts_config.py` 中可以调整核心超参（示例值）：
```python
# 搜索
c_puct = 0.8           # UCB 探索系数（HiTab 可调至 1.6）
max_simulations = 80   # 最大模拟次数
max_depth = 12         # 最大搜索深度
rollout_steps = 3      # 单次仿真步长

# VM
vm_threshold = 0.9     # 剪枝阈值（低于阈值提前截断）

# 检索
bm25_topk = 3
max_cells_retrieved = 10
max_neighbors_explored = 8
```

---

## 📊 基准结果（摘要）

在 HiTab 与 AIT-QA 上，**MCTS+VM** 相对强基线与消融设定（CoT、MCTS-启发式）呈现稳定的小幅提升（多数设置约 **0.6–2.0 个百分点**），并表现出更低的结果方差与更一致的推理路径。完整数据与设置示例见报告“表 3-3、表 3-4”。

---

## 🧪 常见问题

- **无法运行时怎么办？**  
  作为研究原型，环境差异可能导致依赖或接口不完全一致。建议从最小路径（`--mode cot`）跑起，再逐步启用检索、MCTS 与 VM；根据报错调整依赖版本或配置。

- **一定要在线 API 吗？**  
  若使用在线后端需要相应 API Key。也可切换到本地开源模型（并调整 `openai_api.py` 封装）。

- **LLM Eval 怎么配？**  
  固定判分模板、温度与投票规则，避免随机性带来的方差；模板样例可参考报告附录。

---

## 📚 引用与致谢

本项目受图结构学习、表格理解、显式推理与搜索增强等方向的启发。感谢团队成员在数据清洗、实验跑批、日志与文档上的协作，也感谢开源社区提供的大模型、检索组件与训练工具。

---



## 🗺️ 路线图
  ![0f2436d096116db0ff8ddfadb387ddc](https://github.com/user-attachments/assets/f47dbbaa-e181-41a8-a2f7-936d487ec812)
  ![3d4c2a022c41b26d62127b68f170090](https://github.com/user-attachments/assets/a1bf61a0-7b93-46c0-b763-3d737cba0328)



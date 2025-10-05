CUDA_VISIBLE_DEVICES=0 python main.py --model qwen2-72b-instruct \
--key 替换为您的apikay \
--dataset hitab \
--qa_path dataset/hitab/test_samples.jsonl \
--table_folder dataset/hitab/raw/ \
--embedder_path gte-base \
--embed_cache_dir dataset/hitab/ \
--temperature 0.0 \
--max_iteration_depth 4 \
--seed 42



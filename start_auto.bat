@echo off
echo auto run...
python main.py --model qwen2-7b-instruct ^
--base_url https://dashscope.aliyuncs.com/compatible-mode/v1 ^
--key 替换为您的apikay ^
--dataset ait-qa ^
--qa_path dataset/AIT-QA/aitqa_clean_questions.json ^
--embedder_path thenlper/gte-base ^
--embed_cache_dir dataset/AIT-QA/ ^
--temperature 0.0 ^
--max_iteration_depth 4 ^
--seed 42
echo Done!
pause 
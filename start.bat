@echo off
python main.py --model /root/autodl-tmp/models/qwen2-7b-instruct ^
--base_url http://127.0.0.1:6006/v1 ^
--key no-need ^
--dataset ait-qa ^
--qa_path dataset/AIT-QA/aitqa_clean_questions.json ^
--embedder_path thenlper/gte-base ^
--embed_cache_dir dataset/AIT-QA/ ^
--temperature 0.0 ^
--max_iteration_depth 4 ^
--start 0 ^
--end 2 ^
--seed 42 ^
--debug True
pause 
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:6006/v1", api_key="not-needed")
resp = client.chat.completions.create(
    model="/root/autodl-tmp/models/qwen2-7b-instruct",
    messages=[{"role":"user","content":"你是誰?"}],
)
print(resp.choices[0].message.content)
# import os
# from openai import OpenAI


# client = OpenAI(
#     api_key=os.getenv("DASHSCOPE_API_KEY"),
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
# )

completion = client.chat.completions.create(
    model="qwen-plus",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你是谁？"},
    ],
    extra_body={"enable_thinking": True},
)
print(completion.choices[0].message.reasoning_content + "\n\n" + "-"*80 + "\n\n" + completion.choices[0].message.content)

from langchain_core.prompts import PromptTemplate
# 初始化RAG提示词模板
template = """
你是一位专业的糖尿病营养师。请根据以下患者信息和提供的相关知识，为该患者制定一份个性化的饮食建议。你的建议可以是相关营养素（碳水化合物、脂肪、纤维素等），可以是某种食物的原材料（如大米、玉米、豆腐、青菜、猪肉等），也可以是具体的菜名（如清蒸黄鱼、白灼虾等），根据给定的信息决定。你需要给出具体的食谱以及其中每种食物/营养素的摄入量（克）。
患者信息：
{information}
相关知识：
{context}
你的建议必须严格遵循“食物/营养素：摄入量（单位）    注释”的格式，每个建议换行一次。
"""
prompt = PromptTemplate.from_template(template)
print(f"prompt:{prompt}\n")

example_messages = prompt.invoke(
    {"information":"patient's information", "context":"retrieved context"}
).to_messages()
assert len(example_messages) == 1
print(f"example_messages[0]:{example_messages[0].content}\n")
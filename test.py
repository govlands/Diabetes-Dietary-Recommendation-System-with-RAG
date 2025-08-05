from langchain_community.embeddings import DashScopeEmbeddings
embeddings = DashScopeEmbeddings(
    model="text-embedding-v4", dashscope_api_key="sk-9898f5c24a334457a3791842b1e05142"
)

test_texts = [
    "糖尿病的饮食原则有哪些？",
    "2型糖尿病患者如何选择主食？"
]

try:
    result = embeddings.embed_documents(test_texts)
    print("Embedding 正常，返回向量 shape:", len(result), "每条长度:", len(result[0]) if result else 0)
except Exception as e:
    print("Embedding 报错：", e)

# print(completion.model_dump_json())
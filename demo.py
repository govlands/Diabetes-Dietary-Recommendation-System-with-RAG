# 初始化llm、嵌入模型和向量库
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    model="qwen-plus",
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 通义千问 OpenAI 兼容地址
    openai_api_key="sk-9898f5c24a334457a3791842b1e05142"
)

from langchain_community.embeddings import DashScopeEmbeddings
embeddings = DashScopeEmbeddings(
    model="text-embedding-v4", dashscope_api_key="sk-9898f5c24a334457a3791842b1e05142"
)

from langchain_core.vectorstores import InMemoryVectorStore
vector_store = InMemoryVectorStore(embeddings)

# 加载文档
import glob
from langchain_community.document_loaders import PyPDFLoader

# pdf_files = glob.glob("KB/*.pdf")
# docs = []
# for file_path in pdf_files:
#     loader = PyPDFLoader(file_path)
#     docs.extend(loader.load())

file_path = "KB\成人糖尿病食养指南.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()

# 分割文档
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
    separators=[
        "\n\n",
        "。",
        "，",
        " ",
        ",",
        "."
    ]
)
all_splits = text_splitter.split_documents(docs)
print(f"Split blog post into {len(all_splits)} sub-documents.\n")

# print([type(doc.page_content) for doc in all_splits[:5]])

# test_texts = [doc.page_content for doc in all_splits[:5]]
# for text in test_texts:
#     print(f"{text}\n\n")
# print(embeddings.embed_documents(test_texts))

# 将分割后的片段转化为向量存入向量库
document_ids = vector_store.add_documents(documents=all_splits)
print(f"document_ids[:3]:{document_ids[:3]}\n")

# 初始化RAG提示词模板
from langchain_core.prompts import PromptTemplate
template = """你是一位专业的糖尿病营养师。请根据以下患者信息和提供的相关知识，为该患者制定一份个性化的饮食建议。你的建议可以是相关营养素（碳水化合物、脂肪、纤维素等），可以是某种食物的原材料（如大米、玉米、豆腐、青菜、猪肉等），也可以是具体的菜名（如清蒸黄鱼、白灼虾等）。你需要给出具体的食谱以及其中每种食物/营养素的摄入量（克）。

患者信息：
{information}

相关知识：
{context}

你的建议："""
prompt = PromptTemplate.from_template(template)
# print(f"prompt:{prompt}\n")

example_messages = prompt.invoke(
    {"information":"patient's information", "context":"retrieved context"}
).to_messages()
assert len(example_messages) == 1
# print(f"example_messages[0]:{example_messages[0]}\n")

# 检索与生成
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
class State(TypedDict):
    """
    将 根据问题检索信息-提示词构建-llm生成 视为一个工作流，State类的对象包含了工作流中每个节点（环节）所需要的全部信息
    """
    information: str
    query: str
    context: List[Document]
    answer: str
    
def analyze_query(state: State):
    """
    根据state中的患者信息（information）生成query
    """
    information = state["information"]
    # 直接用全部信息作为query，最大化召回
    query = information+" 饮食 营养素 食谱 食用 摄入"
    return {"query": query}


def retrieve(state: State):
    """
    检索节点，使用state中的query，得到相关知识（context）
    """
    retrieved_docs = vector_store.similarity_search(state["query"])
    return {"context": retrieved_docs}

def generate(state: State):
    """
    生成节点，根据state中的information、context和提示词模板生成整个提示词，然后输入llm进行推理
    """
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"information": state["information"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

# 将节点连接并生成一个工作流程图
from langgraph.graph import START, StateGraph
graph_builder = StateGraph(State).add_sequence([analyze_query, retrieve, generate]) # 初始化StateGraph对象并加入两个节点
graph_builder.add_edge(START, "analyze_query") # 添加从START到检索的边
graph = graph_builder.compile() #生成图像

# 最终生成
# result = graph.invoke(input={"question": "What is Task Decomposition?"})
# print(f"Context: {result['context']}\n\n")
# print(f"Answer: {result['answer']}")

# 流式输出，可以看到各个节点执行后的state
information = """姓名：张三
性别：男
年龄：55岁
身高：170cm
体重：75kg
糖尿病类型：2型糖尿病
空腹血糖：8.2 mmol/L
糖化血红蛋白：7.8%
血脂：总胆固醇5.2 mmol/L，甘油三酯2.1 mmol/L
并发症：高血压
饮食偏好：喜欢米饭、面食，偶尔吃肉"""

for step in graph.stream(
    {"information": information}, stream_mode="updates"
):
    print(f"{step}\n\n----------------\n")
# import torch
# print(torch.cuda.is_available())  # True 表示可用
# print(torch.cuda.device_count())
# for i in range(torch.cuda.device_count()):
#     print(i, torch.cuda.get_device_name(i))

# 初始化llm、嵌入模型和向量库
# from transformers import pipeline
# from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
# pipe = pipeline("text-generation", model="Qwen/Qwen3-0.6B")
# hf_pipe = HuggingFacePipeline(pipeline=pipe)
# llm = ChatHuggingFace(llm=hf_pipe)

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    model="qwen-plus",
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 通义千问 OpenAI 兼容地址
    openai_api_key="sk-9898f5c24a334457a3791842b1e05142"
)

from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

from langchain_core.vectorstores import InMemoryVectorStore
vector_store = InMemoryVectorStore(embeddings)

# 加载文档
import bs4
from langchain_community.document_loaders import WebBaseLoader
# Only keep post title, headers, and content from the full HTML.
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()
assert len(docs) == 1
print(f"Total characters: {len(docs[0].page_content)}\n")

# 分割文档
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)
print(f"Split blog post into {len(all_splits)} sub-documents.\n")
# print(f"first sub-doc: {all_splits[0].page_content}")

# 修改片段metadata
total_documents = len(all_splits)
third = total_documents // 3
for i, document in enumerate(all_splits):
    if i < third:
        document.metadata["section"] = "beginning"
    elif i < 2 * third:
        document.metadata["section"] = "middle"
    else:
        document.metadata["section"] = "end"

# 将分割后的片段转化为向量存入向量库
document_ids = vector_store.add_documents(documents=all_splits)
print(f"document_ids[:3]:{document_ids[:3]}\n")

# 初始化RAG提示词模板
from langchain import hub
# N.B. for non-US LangSmith endpoints, you may need to specify
# api_url="https://api.smith.langchain.com" in hub.pull.
prompt = hub.pull("rlm/rag-prompt", api_url="https://api.smith.langchain.com")
# print(f"prompt:{prompt}\n")

example_messages = prompt.invoke(
    {"context": "(context goes here)", "question": "(question goes here)"}
).to_messages()
assert len(example_messages) == 1
# print(f"example_messages[0]:{example_messages[0]}\n")

# 检索与生成
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from typing import Literal
from typing_extensions import Annotated

class Search(TypedDict):
    """Search query."""

    query: Annotated[str, ..., "Search query to run."]
    section: Annotated[
        Literal["beginning", "middle", "end"],
        ...,
        "Section to query.",
    ]

class State(TypedDict):
    """
    将 根据问题检索信息-提示词构建-llm生成 视为一个工作流，State类的对象包含了工作流中每个节点（环节）所需要的全部信息
    """
    question: str
    query: Search
    context: List[Document]
    answer: str

# def analyze_query(state: State):
#     structured_llm = llm.with_structured_output(Search)
#     query = structured_llm.invoke(state["question"])
#     print("analyze_query output:", query)
#     if query is None:
#         raise ValueError("LLM 未能生成结构化的查询，请检查模型和提示词。")
#     return {"query": query}

def analyze_query(state: State):
    """
    根据state中的question生成query
    """
    question = state["question"]
    prompt_text = (
        "你是一个结构化信息抽取助手，请用如下 JSON 格式输出（请使用纯文本格式，无markdown）："
        '{"query": "...", "section": "beginning|middle|end"}。\n'
        f"问题：{question}"
    )
    response = llm.invoke(prompt_text)
    print("analyze_query raw output:", response)
    import json
    try:
        content = response.content if hasattr(response, "content") else str(response)
        query = json.loads(content)
        assert "query" in query and "section" in query
    except Exception as e:
        raise ValueError(f"无法解析LLM输出: {content}\n错误: {e}")
    return {"query": query}


def retrieve(state: State):
    """根据生成的query进行检索"""
    query = state["query"]
    retrieved_docs = vector_store.similarity_search(
        query["query"],
        filter=lambda doc: doc.metadata.get("section") == query["section"],
    )
    return {"context": retrieved_docs}

def generate(state: State):
    """
    生成节点，根据state中的context、question和提示词模板生成整个提示词，然后输入llm进行推理
    """
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

# 将节点连接并生成一个工作流程图
from langgraph.graph import START, StateGraph
graph_builder = StateGraph(State).add_sequence([analyze_query, retrieve, generate])
graph_builder.add_edge(START, "analyze_query")
graph = graph_builder.compile()

# 最终生成
for step in graph.stream(
    {"question": "What does the end of the post say about Task Decomposition?"},
    stream_mode="updates",
):
    print(f"{step}\n\n----------------\n")
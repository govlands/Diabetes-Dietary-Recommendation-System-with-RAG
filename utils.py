from langchain_core.vectorstores import InMemoryVectorStore
import glob
from langchain_community.document_loaders import PyPDFLoader
import pickle
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
import os
import re
from langchain_core.prompts import PromptTemplate
from openai import OpenAI
from langchain_qdrant import QdrantVectorStore
import pymupdf4llm
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams
from time import sleep
import requests
from typing import Optional

def construct_vector_store(embeddings):
    # 定义集合名称和向量维度
    collection_name = "documents"
    vector_size = 1024  # 根据你的嵌入模型输出维度调整
    cache_path = "vector_store.pkl"

    # 优先从本地缓存加载向量库，减少对 Qdrant 的依赖和重复构建
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                vs = pickle.load(f)
            print(f"已从本地缓存加载向量库: {cache_path}")
            return vs
        except Exception as e_load:
            print(f"加载本地向量库缓存失败({cache_path})，将尝试使用 Qdrant 重建：{e_load}")

    # Qdrant 连接与重试配置
    qdrant_host = os.environ.get("QDRANT_HOST", "localhost")
    qdrant_port = int(os.environ.get("QDRANT_PORT", "6333"))
    qdrant_url = f"http://{qdrant_host}:{qdrant_port}"
    max_retries = 5
    backoff_base = 1.0

    # 创建 Qdrant 客户端时关闭兼容性检查以避免启动警告
    client: Optional[QdrantClient] = None
    last_exception: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            client = QdrantClient(host=qdrant_host, port=qdrant_port, check_compatibility=False)
            # 尝试通过简单的 API 调用检查连接
            # 这里调用 get_collection 用来检测服务是否可用
            try:
                client.get_collection(collection_name=collection_name)
                print(f"集合 '{collection_name}' 已存在。删除并重新创建以确保向量维度正确...")
                client.delete_collection(collection_name=collection_name)
            except Exception:
                # 如果集合不存在或返回 404，会抛出异常，忽略以便创建
                pass

            # 尝试创建集合（如果已删除或不存在）
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance="Cosine")
            )
            print("成功创建/重新创建 Qdrant 集合。")
            last_exception = None
            break
        except Exception as e:
            last_exception = e
            wait = backoff_base * (2 ** (attempt - 1))
            print(f"Qdrant 连接尝试 {attempt}/{max_retries} 失败: {e}; {wait}s 后重试...")
            sleep(wait)

    if client is None or last_exception is not None:
        # 最后一轮重试仍然失败，做更详细的原始 HTTP 检查并抛出明确错误
        diagnostic_msgs = []
        try:
            resp = requests.get(f"{qdrant_url}/collections", timeout=5)
            diagnostic_msgs.append(f"raw GET /collections -> {resp.status_code}: {resp.content[:200]!r}")
        except Exception as e_raw:
            diagnostic_msgs.append(f"raw GET /collections 请求失败: {e_raw}")

        # 如果有最后的异常，包含在提示中
        diag = "; ".join(diagnostic_msgs)
        raise RuntimeError(
            "无法建立到 Qdrant 的可靠连接。最后一次异常: {}。诊断: {}\n".format(last_exception, diag)
            + "检查: 1) Qdrant 是否启动 (docker/服务), 2) 端口 6333 是否被占用或被代理/防火墙拦截, 3) 若使用反向代理(nginx)，查看其日志。"
        )
    # if not client.get_collection(collection_name):
    #     client.create_collection(
    #         collection_name=collection_name,
    #         vectors_config=VectorParams(size=vector_size, distance="Cosine")
    #     )
    #     print("创建 Qdrant 集合成功。")
    # else:
    #     print("检测到 Qdrant 中已有集合，跳过创建...")
    pdf_files = glob.glob("KB/*.pdf")
    docs = []

    for file_path in pdf_files:
        try:
            # 1. 使用 PyMuPDF4LLM 提取 Markdown 文本
            markdown_text = pymupdf4llm.to_markdown(file_path)

            # 2. 手动创建 LangChain 的 Document 对象
            # 这里的 page_content 就是转换后的 Markdown 文本
            # metadata 可以包含原始文件路径等信息
            doc = Document(
                page_content=markdown_text,
                metadata={"source": file_path}
            )
            docs.append(doc)

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
    print(f"成功加载了 {len(docs)} 个文档。")

    # 使用 SemanticChunker 进行语义分块
    # text_splitter = SemanticChunker(embeddings=embeddings)
    # # all_splits = text_splitter.split_documents(docs)
    # # ❗ 手动分批处理，以解决批量大小限制 ❗
    # all_splits = []
    # chunk_size = 1  # 每次处理一个文档，或者更小的批次
    # for i in range(0, len(docs), chunk_size):
    #     batch_docs = docs[i:i + chunk_size]
    #     try:
    #         batch_splits = text_splitter.split_documents(batch_docs)
    #         all_splits.extend(batch_splits)
    #     except Exception as e:
    #         print(f"处理批次 {i} 到 {i + chunk_size - 1} 时出错: {e}")
    #         # 如果某一批次失败，可以选择跳过或记录错误
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
        separators=["\n\n", "。", "，", " ", ",", "."]
    )
    all_splits = text_splitter.split_documents(docs)
    print(f"文档分割为 {len(all_splits)} 个子文档。\n")

    # 初始化 QdrantVectorStore 实例并写入
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name, embedding=embeddings)

    # 手动批处理文档并逐批添加（遇到写入错误时抛出异常）
    batch_size = 10
    for i in range(0, len(all_splits), batch_size):
        batch = all_splits[i:i + batch_size]
        print(f"正在处理第 {i} 到 {i + len(batch) - 1} 个文档...")
        try:
            vector_store.add_documents(batch)
        except Exception as e_add:
            # 在写入阶段提供尽可能多的诊断信息，然后抛出错误，避免静默回退
            # 尝试原始 HTTP 查询以收集更多上下文
            try:
                resp = requests.get(f"{qdrant_url}/collections", timeout=5)
                raw_info = f"HTTP {resp.status_code} {resp.content[:200]!r}"
            except Exception as e_diag:
                raw_info = f"raw HTTP check failed: {e_diag}"
            raise RuntimeError(
                f"向 Qdrant 写入向量库时失败: {e_add}; 诊断: {raw_info}. "
                "建议: 检查 Qdrant 日志、网络/防火墙设置、以及是否存在反向代理返回 502。"
            )

    print("向量库加载完成（Qdrant）。")
    # 尝试将已成功构建的向量库缓存到本地，便于下次直接加载
    # 只对可序列化的本地向量库进行缓存；QdrantVectorStore 是基于客户端连接的远端存储，
    # 通常包含无法序列化的底层锁/连接对象，跳过序列化并给出提示。
    try:
        if isinstance(vector_store, QdrantVectorStore):
            print(f"注意：QdrantVectorStore 为远端客户端驱动的存储，跳过本地序列化缓存。若需本地缓存，请使用 InMemoryVectorStore。")
        else:
            with open(cache_path, "wb") as f:
                pickle.dump(vector_store, f)
            print(f"已将向量库缓存到本地: {cache_path}")
    except (pickle.PicklingError, TypeError, AttributeError) as e_save:
        print(f"警告：无法将向量库缓存到本地 ({cache_path})：{e_save}")
    except Exception as e_save:
        print(f"警告：无法将向量库缓存到本地 ({cache_path})：{e_save}")

    return vector_store


class State(TypedDict):
    """RAG 工作流状态"""
    information: str
    vector_store: InMemoryVectorStore
    prompt: PromptTemplate
    client: OpenAI
    thinking: bool
    queries: List[str]
    context: List[Document]
    answer: str


def parse_info(info: str):
    if not info:
        return ""

    # 统一冒号，按行解析
    text = info.replace("：", ":")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    data = {}
    for ln in lines:
        if ":" in ln:
            k, v = ln.split(":", 1)
            data[k.strip()] = v.strip()

    tokens = []

    # 年龄 -> 年龄分组
    age_val = None
    for key in ("年龄", "Age"):
        if key in data:
            # 提取整数
            m = re.search(r"(\d{1,3})", data[key])
            if m:
                age_val = int(m.group(1))
            data.pop(key)
            break
    if age_val is not None:
        if age_val < 18:
            tokens.append("儿童")
        elif age_val < 45:
            tokens.append("成年")
        elif age_val < 65:
            tokens.append("中年")
        else:
            tokens.append("老年")

    # 性别
    gender_val = None
    for key in ("性别", "Gender"):
        if key in data:
            gender_val = data[key]
            data.pop(key)
            break
    if gender_val:
        gv = gender_val.strip().lower()
        if gv in ("女", "f", "female", "女士", "女性"):
            tokens.append("女性")
        elif gv in ("男", "m", "male", "先生", "男性"):
            tokens.append("男性")
        else:
            tokens.append(gender_val)

    # BMI 分类
    bmi_val = None
    for key in ("BMI",'bmi'):
        if key in data:
            m = re.search(r"(\d+(?:\.\d+)?)", data[key])
            if m:
                bmi_val = float(m.group(1))
            break
    if bmi_val is not None:
        if bmi_val >= 30:
            tokens.append("肥胖")
        elif bmi_val >= 25:
            tokens.append("超重")
        elif bmi_val >= 18.5:
            tokens.append("正常")
        else:
            tokens.append("偏瘦")
            
    # 糖尿病分类（仅判断 健康 / 糖尿病前期 / 糖尿病）
    a1c_val = None
    for key in ("糖化血红蛋白", "A1c", "A1c%"):
        if key in data:
            m = re.search(r"(\d+(?:\.\d+)?)", data[key])
            if m:
                a1c_val = float(m.group(1))
            break

    fasting_val = None
    for key in ("空腹血糖", "Fasting BG", "空腹血糖基线", "Baseline_Libre"):
        if key in data:
            m = re.search(r"(\d+(?:\.\d+)?)", data[key])
            if m:
                fasting_val = float(m.group(1))
            break

    # 判定阈值（mg/dL 与 %）
    is_diabetes = False
    is_prediabetes = False
    if a1c_val is not None:
        if a1c_val >= 6.5:
            is_diabetes = True
        elif 5.7 <= a1c_val < 6.5:
            is_prediabetes = True
    if fasting_val is not None:
        if fasting_val >= 126:
            is_diabetes = True
        elif 100 <= fasting_val < 126:
            is_prediabetes = True

    # 优先级：糖尿病 > 糖尿病前期 > 健康
    if is_diabetes:
        tokens.append("糖尿病")
    elif is_prediabetes:
        tokens.append("糖尿病前期")
    else:
        # 当既没有糖化血红蛋白也没有空腹血糖指标时，不强行判断“健康”
        if a1c_val is None and fasting_val is None:
            pass
        else:
            tokens.append("健康")

    for key in data.keys():
        tokens.append(key)

    # 如果没有任何指标被解析，退回为原始的键名摘要
    if len(tokens) == 0:
        # 尝试返回所有键的简短列表
        keys = list(data.keys())
        return keys

    return tokens


def analyze_query(state: State):
    """
    生成多条增强的检索查询：
    - 返回主 query（用于简洁检索）和 expanded_queries（用于多轮/多角度检索并合并结果）
    """
    info_tokens = parse_info(state['information'])
    core_phrase = '饮食 食物 餐 摄入'
    key_phrases = ['宏量营养素 碳水 蛋白质 纤维素 脂肪', '碳水化合物 碳水 主食 米面', '纤维素 膳食纤维 蔬菜植物', '蛋白质 蛋 肉', '脂肪 油']

    queries = []
    for token in info_tokens:
        for phrase in key_phrases:
            queries.append(f"{token} {core_phrase} {phrase}")
    return {'queries':queries}


def retrieve(state: State):
    """检索相关文档"""
    vector_store = state['vector_store']
    queries = state['queries']
    retrieved_docs = []
    for query in queries:
        retrieved_docs.extend(vector_store.similarity_search(query))
    # 对 retrieved_docs 去重并保留前x条（保持顺序）
    if not retrieved_docs:
        retrieved_docs = []
    else:
        # 统计每个片段被检索到的次数，按次数降序（次数相同按首次出现顺序）取前12
        counts = {}
        first_doc = {}
        first_idx = {}
        for idx, doc in enumerate(retrieved_docs):
            key = doc.page_content
            counts[key] = counts.get(key, 0) + 1
            if key not in first_doc:
                first_doc[key] = doc
                first_idx[key] = idx

        # 排序键：先按次数降序，再按首次出现的索引升序
        sorted_keys = sorted(counts.keys(), key=lambda k: (-counts[k], first_idx[k]))
        top_keys = sorted_keys[:12]
        retrieved_docs = [first_doc[k] for k in top_keys]
    return {"context": retrieved_docs}


def generate(state: State):
    """
    生成节点，根据state中的information、context和提示词模板生成整个提示词，然后输入llm进行推理
    """
    system_prompt = """你是一位专业的糖尿病营养师。请根据用户提供的患者信息和相关知识，为该患者制定一份个性化的饮食建议。

    重要：你必须严格按照以下格式输出，每餐分别给出碳水化合物、蛋白质、脂肪、纤维素的摄入量（单位：克）：

    早餐：
    碳水化合物：XX克
    蛋白质：XX克  
    脂肪：XX克
    纤维素：XX克

    午餐：
    碳水化合物：XX克
    蛋白质：XX克
    脂肪：XX克
    纤维素：XX克

    晚餐：
    碳水化合物：XX克
    蛋白质：XX克
    脂肪：XX克
    纤维素：XX克

    请确保数值合理，适合糖尿病患者的营养需求。"""
    client = state['client']
    prompt = state['prompt']
    thinking = state['thinking']
    docs_content = ""
    for i, doc in enumerate(state["context"]):
        docs_content += f"片段[{i+1}]:{doc.page_content}\n\n"
    messages = prompt.invoke({"information": state["information"], "context": docs_content}).to_messages()[0].content
    if thinking:
        completion = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": messages},
            ],
            stream=False,
        )
        # print(completion)
        answer = "-"*80 + "\n\n" + completion.choices[0].message.reasoning_content + "\n\n" + "-"*80 + "\n\n" + completion.choices[0].message.content
    else:
        completion = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": messages},
            ],
            stream=False,
        )
        answer = "-"*80 + "\n\n" + completion.choices[0].message.content
    
    return {"answer": answer}


def format_patient_info(patient_data):
    """格式化患者信息为文本"""
    gender_map = {1: "男", -1: "女"}
    gender = gender_map.get(patient_data['Gender'], "未知")
    
    info = f"""
    年龄：{patient_data['Age']}岁
    性别：{gender}
    BMI：{patient_data['BMI']:.1f}
    空腹血糖基线：{patient_data['Baseline_Libre']:.1f} mg/dL
    糖化血红蛋白：{patient_data['A1c']:.1f}%
    HOMA指数：{patient_data['HOMA']:.2f}
    空腹胰岛素：{patient_data['Insulin']:.1f} μIU/mL
    甘油三酯：{patient_data['TG']:.1f} mg/dL
    总胆固醇：{patient_data['Cholesterol']:.1f} mg/dL
    高密度脂蛋白：{patient_data['HDL']:.1f} mg/dL
    非高密度脂蛋白：{patient_data['Non HDL']:.1f} mg/dL
    低密度脂蛋白：{patient_data['LDL']:.1f} mg/dL
    极低密度脂蛋白：{patient_data['VLDL']:.1f} mg/dL
    胆固醇/HDL比值：{patient_data['CHO/HDL ratio']:.2f}
    空腹血糖：{patient_data['Fasting BG']:.1f} mg/dL
    """
    return info


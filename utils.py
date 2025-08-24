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

def construct_vector_store(embeddings):
    """构建向量数据库"""
    vector_store_path = "vector_store.pkl"
    vector_store = InMemoryVectorStore(embeddings)
    if os.path.exists(vector_store_path):
        print("检测到本地向量库，正在加载...")
        with open(vector_store_path, "rb") as f:
            all_splits, metadatas = pickle.load(f)
        # 重新构建向量库
        batch_size = 10
        for i in range(0, len(all_splits), batch_size):
            batch = all_splits[i:i+batch_size]
            vector_store.add_documents(documents=batch)
    else:
        print("未检测到本地向量库，正在分割文档并嵌入...")
        pdf_files = glob.glob("KB/*.pdf")
        docs = []
        for file_path in pdf_files:
            loader = PyPDFLoader(file_path)
            docs.extend(loader.load())
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True,
            separators=["\n\n", "。", "，", " ", ",", "."]
        )
        all_splits = text_splitter.split_documents(docs)
        print(f"Split documents into {len(all_splits)} sub-documents.")
        
        batch_size = 10
        for i in range(0, len(all_splits), batch_size):
            batch = all_splits[i:i+batch_size]
            vector_store.add_documents(documents=batch)
            
        # 保存分割结果和元数据到本地
        metadatas = [doc.metadata for doc in all_splits]
        with open(vector_store_path, "wb") as f:
            pickle.dump((all_splits, metadatas), f)
            
    print("向量库加载完成")
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
    """
    把 information 文本解析为简短描述字符串，例如：
    中年 女性 肥胖 空腹血糖基线 糖化血红蛋白 HOMA指数 ...

    支持 key 后使用中文冒号或英文冒号。仅保留出现的关键指标，按固定顺序输出。
    """
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
        seen = set()
        unique = []
        for doc in retrieved_docs:
            key = doc.page_content
            if key in seen:
                continue
            seen.add(key)
            unique.append(doc)
            if len(unique) >= 12:
                break
        retrieved_docs = unique
    return {"context": retrieved_docs}


def generate(state: State):
    """
    生成节点，根据state中的information、context和提示词模板生成整个提示词，然后输入llm进行推理
    """
    client = state['client']
    prompt = state['prompt']
    thinking = state['thinking']
    docs_content = ""
    for i, doc in enumerate(state["context"]):
        docs_content += f"片段[{i+1}]:{doc.page_content}\n\n"
    messages = prompt.invoke({"information": state["information"], "context": docs_content}).to_messages()[0].content
    if thinking:
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": """你是一位专业的糖尿病营养师。请根据用户提供的患者信息和相关知识，为该患者制定一份个性化的饮食建议。

    重要：在你的推理过程（不是最终输出）中，你必须显示指出引用了哪个片段，直接在引用处后添加'[i]'即可，其中i是引用的片段序号。
    你必须严格按照以下格式输出，每餐分别给出碳水化合物、蛋白质、脂肪、纤维素的摄入量（单位：克）：

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

    请确保数值合理，适合糖尿病患者的营养需求。"""},
                {"role": "user", "content": messages},
            ],
            extra_body={"enable_thinking": True},
        )
        answer = "-"*80 + "\n\n" + completion.choices[0].message.reasoning_content + "\n\n" + "-"*80 + "\n\n" + completion.choices[0].message.content
        # print(completion)
    else:
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": """你是一位专业的糖尿病营养师。请根据用户提供的患者信息和相关知识，为该患者制定一份个性化的饮食建议。

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

    请确保数值合理，适合糖尿病患者的营养需求。"""},
                {"role": "user", "content": messages},
            ],
            extra_body={"enable_thinking": False},
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


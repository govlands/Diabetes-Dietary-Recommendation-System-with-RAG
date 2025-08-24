from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.prompts import PromptTemplate
from langgraph.graph import START, StateGraph
import os
from openai import OpenAI
from pred_iauc import get_data_all_sub
from time import time
import numpy as np
from utils import *


def fetch_infos_from_cgmacros(n_infos:int=1):
    data_all_sub = get_data_all_sub(data_dir)
    feature_cols = ['Baseline_Libre', 'Age', 'Gender', 
                    'BMI', 'A1c', 'HOMA', 'Insulin', 'TG', 'Cholesterol', 'HDL', 'Non HDL', 
                    'LDL', 'VLDL', 'CHO/HDL ratio', 'Fasting BG']
    unique_subjects = data_all_sub['sub'].unique()
    np.random.seed(int(time()))
    selected_subjects = np.random.choice(unique_subjects, size=n_infos, replace=False)
    results = []
    for sub in selected_subjects:
        sub_info = data_all_sub[
            (data_all_sub['sub'] == sub)
        ].iloc[0]
        patient_info = sub_info[feature_cols].to_dict()
        info = format_patient_info(patient_info)
        results.append(info)
    return results


if __name__ == "__main__":
    original_dir = os.getcwd()
    data_dir = original_dir + '/cgmacros1.0/CGMacros'
    
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v4", dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
    )
    vector_store = construct_vector_store(embeddings)
    
    # 初始化RAG提示词模板
    template = """
    患者信息：
    {information}
    相关知识：
    {context}
    """
    prompt = PromptTemplate.from_template(template)

    # 将节点连接并生成一个工作流程图
    # graph_builder = StateGraph(State).add_sequence([analyze_query, retrieve, generate])
    graph_builder = StateGraph(State).add_sequence([analyze_query, retrieve])
    graph_builder.add_edge(START, "analyze_query") # 添加从START到检索的边
    graph = graph_builder.compile() #生成图像

    information = fetch_infos_from_cgmacros()[0]
    
    state = State(
        information=information,
        vector_store=vector_store,
        prompt=prompt,
        client=client,
        thinking=False
    )

    # 执行一次完整推理流程，获取最终结果
    result = graph.invoke(state)
    
    print("【病人信息】")
    print(information)
    
    print("【生成的queries】")
    for query in result.get('queries', []):
        print(query)
    print('-' * 40)

    # 输出检索到的文档内容（retrieve阶段）
    print("【检索到的相关知识片段】")
    for i, doc in enumerate(result.get("context", [])):
        print(f"\n片段{i+1}：{doc.metadata['source']}\t 第{doc.metadata['page']}页")
        print(doc.page_content)
        print("-" * 40)

    # 输出最终答案（answer阶段）
    print("\n【个性化饮食建议】")
    print(result.get("answer", ""))
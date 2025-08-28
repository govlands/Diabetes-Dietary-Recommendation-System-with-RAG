import os
import pandas as pd
import numpy as np
import re
import pickle
import json
from datetime import datetime
from time import time
from openai import OpenAI
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.prompts import PromptTemplate
from langgraph.graph import START, StateGraph
from pred_iauc import get_data_all_sub
from utils import *


def convert_numpy_types(obj):
    """递归转换numpy类型为Python原生类型，使其可以JSON序列化"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def generate_llm_only(client, patient_info):
    """仅使用LLM生成营养建议，不使用RAG"""
    completion = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": """你是一位专业的糖尿病营养师。请根据用户提供的样本信息，为该样本制定一份个性化的早餐建议。

重要：你必须严格按照以下格式输出早餐的营养素摄入量（单位：克）：

早餐：
碳水化合物：XX克
蛋白质：XX克  
脂肪：XX克
纤维素：XX克

请确保数值合理，适合糖尿病样本的营养需求。"""},
            {"role": "user", "content": f"样本信息：\n{patient_info}"},
        ],
    )
    return completion.choices[0].message.content


def parse_nutrition_output(text):
    """从LLM输出中解析营养素数值，返回字典格式"""
    patterns = {
        "碳水化合物": r"碳水化合物[：:]\s*(\d+(?:\.\d+)?)\s*克",
        "蛋白质": r"蛋白质[：:]\s*(\d+(?:\.\d+)?)\s*克", 
        "脂肪": r"脂肪[：:]\s*(\d+(?:\.\d+)?)\s*克",
        "纤维素": r"纤维素[：:]\s*(\d+(?:\.\d+)?)\s*克"
    }
    
    nutrition = {}
    default_values = {"碳水化合物": 0, "蛋白质": 0, "脂肪": 0, "纤维素": 0}
    
    for nutrient, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            nutrition[nutrient] = float(match.group(1))
        else:
            nutrition[nutrient] = default_values[nutrient]
            
    return nutrition


def nutrition_to_features(nutrition):
    """将营养素转换为模型特征格式 ['Carbs', 'Protein', 'Fat', 'Fiber']"""
    return {
        'Carbs': nutrition['碳水化合物'],
        'Protein': nutrition['蛋白质'],
        'Fat': nutrition['脂肪'],
        'Fiber': nutrition['纤维素']
    }


def create_sample_row(nutrition_features, patient_features):
    """创建完整的样本行，按照模型所需特征顺序"""
    model_features = ['Carbs', 'Protein', 'Fat', 'Fiber', 'Baseline_Libre', 'Age', 'Gender', 
                      'BMI', 'A1c', 'HOMA', 'Insulin', 'TG', 'Cholesterol', 'HDL', 'Non HDL', 
                      'LDL', 'VLDL', 'CHO/HDL ratio', 'Fasting BG']
    
    sample_row = {}
    # 添加营养素特征
    for feature in ['Carbs', 'Protein', 'Fat', 'Fiber']:
        sample_row[feature] = nutrition_features[feature]
    
    # 添加样本特征
    patient_feature_names = ['Baseline_Libre', 'Age', 'Gender', 'BMI', 'A1c', 'HOMA', 
                           'Insulin', 'TG', 'Cholesterol', 'HDL', 'Non HDL', 'LDL', 
                           'VLDL', 'CHO/HDL ratio', 'Fasting BG']
    for feature in patient_feature_names:
        sample_row[feature] = patient_features[feature]
    
    return sample_row


def main():
    # 配置参数
    n_samples = 120  # 可以修改样本数量
    
    print("开始生成测试样本...")
    print(f"计划生成 {n_samples} 个早餐建议样本")
    
    # 1. 加载数据
    print("\n1. 加载数据...")
    data_all_sub = get_data_all_sub()
    print(f"总数据量: {len(data_all_sub)} 条记录")
    
    # 筛选早餐数据
    breakfast_data = data_all_sub[data_all_sub['Meal Type'] == 1]
    print(f"早餐数据量: {len(breakfast_data)} 条记录")
    
    # 检查必需特征是否存在
    required_features = ['Carbs', 'Protein', 'Fat', 'Fiber', 'Baseline_Libre', 'Age', 'Weight', 
                        'Height', 'Gender', 'BMI', 'A1c', 'HOMA', 'Insulin', 'TG', 'Cholesterol', 
                        'HDL', 'Non HDL', 'LDL', 'VLDL', 'CHO/HDL ratio', 'Fasting BG', 'iAUC']
    
    missing_features = [f for f in required_features if f not in breakfast_data.columns]
    if missing_features:
        print(f"警告：缺少以下特征: {missing_features}")
    
    np.random.seed(int(time()))
    n_samples = min(n_samples, len(breakfast_data))
    selected_indices = np.random.choice(len(breakfast_data), size=n_samples, replace=False)
    selected_breakfast_data = breakfast_data.iloc[selected_indices].reset_index(drop=True)
    print(f"实际选择样本数量: {n_samples}")
    
    # 2. 初始化RAG和LLM系统
    print("\n2. 初始化RAG和LLM系统...")
    
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )
    
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v4", 
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
    )
    
    vector_store = construct_vector_store(embeddings, True)
    
    # 初始化RAG提示词模板
    template = """
    样本信息：
    {information}
    相关知识：
    {context}
    """
    prompt = PromptTemplate.from_template(template)
    
    # 构建RAG工作流
    graph_builder = StateGraph(State).add_sequence([analyze_query, retrieve, generate])
    graph_builder.add_edge(START, "analyze_query")
    graph = graph_builder.compile()
    
    print("RAG和LLM系统初始化完成!")
    
    # 3. 生成样本
    print("\n3. 开始生成样本...")
    
    # 存储结果的列表
    rag_samples = []
    llm_samples = []
    original_iauc = []
    sample_metadata = []
    
    patient_info_features = ['Baseline_Libre', 'Age', 'Weight', 'Height', 'Gender', 'BMI', 'A1c', 
                           'HOMA', 'Insulin', 'TG', 'Cholesterol', 'HDL', 'Non HDL', 'LDL', 
                           'VLDL', 'CHO/HDL ratio', 'Fasting BG']
    
    for i in range(n_samples):
        print(f"\n--- 处理样本 {i+1}/{n_samples} ---")
        
        # 获取该样本的早餐数据
        subject_breakfast = selected_breakfast_data.iloc[i]
        subject = subject_breakfast['sub']
        print(f"样本来源: {subject}")
        
        # 提取样本信息
        patient_data = {}
        for feature in patient_info_features:
            patient_data[feature] = subject_breakfast[feature]
        
        # 格式化样本信息文本
        patient_info_text = format_patient_info(patient_data)
        print(f"样本信息: {patient_info_text[:100]}...")
        
        # RAG生成营养建议
        print("使用RAG生成营养建议...")
        try:
            state = State(
                information=patient_info_text,
                vector_store=vector_store,
                prompt=prompt,
                client=client,
                thinking=True
            )
            rag_result = graph.invoke(state)
            rag_text = rag_result.get("answer", "")
            rag_nutrition = parse_nutrition_output(rag_text)
            print(f"RAG营养建议: {rag_nutrition}")
        except Exception as e:
            print(f"RAG生成失败: {e}，使用默认值")
            rag_nutrition = {"碳水化合物": 0, "蛋白质": 0, "脂肪": 0, "纤维素": 0}
        
        # LLM生成营养建议
        print("使用LLM生成营养建议...")
        try:
            llm_text = generate_llm_only(client, patient_info_text)
            llm_nutrition = parse_nutrition_output(llm_text)
            print(f"LLM营养建议: {llm_nutrition}")
        except Exception as e:
            print(f"LLM生成失败: {e}，使用默认值")
            llm_nutrition = {"碳水化合物": 0, "蛋白质": 0, "脂肪": 0, "纤维素": 0}
        
        # 转换为特征格式
        rag_features = nutrition_to_features(rag_nutrition)
        llm_features = nutrition_to_features(llm_nutrition)
        
        # 创建完整样本
        rag_sample = create_sample_row(rag_features, patient_data)
        llm_sample = create_sample_row(llm_features, patient_data)
        
        # 获取原始iAUC
        true_iauc = subject_breakfast['iAUC']
        
        # 保存结果
        rag_samples.append(rag_sample)
        llm_samples.append(llm_sample)
        original_iauc.append(true_iauc)
        
        # 保存元数据
        metadata = {
            'subject_id': int(subject),  # 确保是Python int
            'original_carbs': float(subject_breakfast['Carbs']),
            'original_protein': float(subject_breakfast['Protein']),
            'original_fat': float(subject_breakfast['Fat']),
            'original_fiber': float(subject_breakfast['Fiber']),
            'rag_nutrition': rag_nutrition,
            'llm_nutrition': llm_nutrition,
            'rag_text': rag_text if 'rag_text' in locals() else "",
            'llm_text': llm_text if 'llm_text' in locals() else ""
        }
        sample_metadata.append(metadata)
        
        print(f"原始iAUC: {true_iauc:.2f}")
        print(f"RAG样本特征: Carbs={rag_sample['Carbs']:.1f}, Protein={rag_sample['Protein']:.1f}")
        print(f"LLM样本特征: Carbs={llm_sample['Carbs']:.1f}, Protein={llm_sample['Protein']:.1f}")
    
    # 4. 保存数据到本地
    print("\n4. 保存数据到本地...")
    
    # 创建保存目录
    save_dir = 'generated_samples'
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 转换为DataFrame并保存
    rag_df = pd.DataFrame(rag_samples)
    llm_df = pd.DataFrame(llm_samples)
    iauc_df = pd.DataFrame({'iAUC': original_iauc})
    
    # 保存CSV文件
    rag_path = f'{save_dir}/rag_samples_{timestamp}.csv'
    llm_path = f'{save_dir}/llm_samples_{timestamp}.csv'
    iauc_path = f'{save_dir}/original_iauc_{timestamp}.csv'
    metadata_path = f'{save_dir}/sample_metadata_{timestamp}.json'
    
    rag_df.to_csv(rag_path, index=False)
    llm_df.to_csv(llm_path, index=False)
    iauc_df.to_csv(iauc_path, index=False)
    
    # 保存元数据
    with open(metadata_path, 'w', encoding='utf-8') as f:
        # 转换numpy类型为Python原生类型
        serializable_metadata = convert_numpy_types(sample_metadata)
        json.dump(serializable_metadata, f, indent=2, ensure_ascii=False)
    
    print(f"RAG样本保存到: {rag_path}")
    print(f"LLM样本保存到: {llm_path}")
    print(f"原始iAUC保存到: {iauc_path}")
    print(f"元数据保存到: {metadata_path}")
    
    # 5. 生成摘要报告
    print("\n5. 生成摘要报告...")
    print(f"成功生成 {len(rag_samples)} 个样本")
    print(f"RAG样本特征统计:")
    print(rag_df[['Carbs', 'Protein', 'Fat', 'Fiber']].describe())
    print(f"\nLLM样本特征统计:")
    print(llm_df[['Carbs', 'Protein', 'Fat', 'Fiber']].describe())
    print(f"\n原始iAUC统计:")
    print(iauc_df.describe())
    
    # 保存完整的打包文件
    package_path = f'{save_dir}/complete_package_{timestamp}.pkl'
    package_data = {
        'rag_samples': rag_df,
        'llm_samples': llm_df,
        'original_iauc': iauc_df,
        'metadata': sample_metadata,
        'timestamp': timestamp,
        'n_samples': n_samples
    }
    
    with open(package_path, 'wb') as f:
        pickle.dump(package_data, f)
    
    print(f"\n完整数据包保存到: {package_path}")
    print("样本生成完成！")


if __name__ == "__main__":
    main()

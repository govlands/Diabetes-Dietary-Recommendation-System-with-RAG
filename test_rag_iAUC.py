import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
from tabpfn_client import TabPFNRegressor, init
import warnings
from pred_iauc import get_data_all_sub
from sklearn.model_selection import train_test_split
import re
import random
from openai import OpenAI
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from langgraph.graph import START, StateGraph
from langchain_text_splitters import RecursiveCharacterTextSplitter
from time import time
from matplotlib import font_manager, rcParams


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
    query: str
    context: List[Document]
    answer: str


def analyze_query(state: State):
    """分析查询"""
    information = state["information"]
    query = information + " 早餐 午餐 晚餐 碳水化合物 蛋白质 脂肪 纤维素 摄入量"
    return {"query": query}


def retrieve(state: State):
    """检索相关文档"""
    retrieved_docs = vector_store.similarity_search(state["query"])
    return {"context": retrieved_docs}


def generate(state: State):
    """生成营养建议"""
    docs_content = ""
    for i, doc in enumerate(state["context"]):
        docs_content += f"片段[{i+1}]:{doc.page_content}\n\n"
    messages = prompt.invoke({"information": state["information"], "context": docs_content}).to_messages()[0].content
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
    )
    answer = completion.choices[0].message.content
    return {"answer": answer}


def generate_llm_only(patient_info):
    """仅使用LLM生成营养建议，不使用RAG"""
    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": """你是一位专业的糖尿病营养师。请根据用户提供的患者信息，为该患者制定一份个性化的饮食建议。

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
            {"role": "user", "content": f"患者信息：\n{patient_info}"},
        ],
    )
    return completion.choices[0].message.content


def parse_nutrition_output(text):
    """从LLM输出中解析营养素数值"""
    # 解析格式：早餐/午餐/晚餐中的碳水化合物、蛋白质、脂肪、纤维素
    meals = {"早餐": {}, "午餐": {}, "晚餐": {}}
    
    patterns = {
        "碳水化合物": r"碳水化合物[：:]\s*(\d+(?:\.\d+)?)\s*克",
        "蛋白质": r"蛋白质[：:]\s*(\d+(?:\.\d+)?)\s*克", 
        "脂肪": r"脂肪[：:]\s*(\d+(?:\.\d+)?)\s*克",
        "纤维素": r"纤维素[：:]\s*(\d+(?:\.\d+)?)\s*克"
    }
    
    # 分割不同餐次
    for meal in ["早餐", "午餐", "晚餐"]:
        meal_start = text.find(f"{meal}：")
        if meal_start == -1:
            continue
            
        # 找到下一个餐次的开始位置
        next_meals = [m for m in ["早餐", "午餐", "晚餐"] if m != meal]
        meal_end = len(text)
        for next_meal in next_meals:
            next_start = text.find(f"{next_meal}：", meal_start + 1)
            if next_start != -1:
                meal_end = min(meal_end, next_start)
                
        meal_text = text[meal_start:meal_end]
        
        # 解析每个营养素
        for nutrient, pattern in patterns.items():
            match = re.search(pattern, meal_text)
            if match:
                meals[meal][nutrient] = float(match.group(1))
            else:
                # 如果没找到，设为默认值
                default_values = {"碳水化合物": 45, "蛋白质": 20, "脂肪": 15, "纤维素": 8}
                meals[meal][nutrient] = default_values[nutrient]
                
    return meals


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


# def predict_iauc_for_meal(training_data, patient_data, nutrition_values, debug=False):
#     """
#     使用训练好的模型预测iAUC - 每次创建新的模型实例以避免缓存问题
#     training_data: (X_train, y_train) 训练数据
#     nutrition_values: {'碳水化合物': xx, '蛋白质': xx, '脂肪': xx, '纤维素': xx}
#     """
#     X_train, y_train = training_data
    
#     # 构建完整特征向量
#     feature_vector = []
    
#     # 营养素特征（需要乘以转换系数）
#     carbs = nutrition_values['碳水化合物'] * 4
#     protein = nutrition_values['蛋白质'] * 4  
#     fat = nutrition_values['脂肪'] * 9
#     fiber = nutrition_values['纤维素'] * 2
    
#     feature_vector.extend([carbs, protein, fat, fiber])
    
#     # 患者基础信息特征
#     patient_features = ['Baseline_Libre', 'Age', 'Gender', 'BMI', 'A1c', 'HOMA', 
#                        'Insulin', 'TG', 'Cholesterol', 'HDL', 'Non HDL', 'LDL', 
#                        'VLDL', 'CHO/HDL ratio', 'Fasting BG']
    
#     for feature in patient_features:
#         feature_vector.append(patient_data[feature])
    
#     # 转换为DataFrame格式（TabPFN需要）
#     feature_names = ['Carbs', 'Protein', 'Fat', 'Fiber'] + patient_features
#     X = np.array(feature_vector, dtype=np.float64).reshape(1, -1)
#     X_df = pd.DataFrame(X, columns=feature_names)
    
#     if debug:
#         print(f"    营养素: Carbs={carbs:.0f}, Protein={protein:.0f}, Fat={fat:.0f}, Fiber={fiber:.0f}")
#         print(f"    完整特征向量前4个: {feature_vector[:4]}")
    
#     # 创建新的模型实例并训练 - 避免缓存问题
#     import random
#     random_seed = int(time())
#     model = TabPFNRegressor(random_state=random_seed)
#     model.fit(X_train, y_train)
    
#     # 预测
#     pred_iauc = model.predict(X_df)[0]
    
#     if debug:
#         print(f"    使用随机种子: {random_seed}")
#         print(f"    预测结果: {pred_iauc:.2f}")
    
#     return pred_iauc

def predict_iauc_for_meal(model, patient_data, nutrition_values, debug=False):
    """
    使用训练好的模型预测iAUC - 每次创建新的模型实例以避免缓存问题
    training_data: (X_train, y_train) 训练数据
    nutrition_values: {'碳水化合物': xx, '蛋白质': xx, '脂肪': xx, '纤维素': xx}
    """
    
    # 构建完整特征向量
    feature_vector = []
    
    # 营养素特征（需要乘以转换系数）
    carbs = nutrition_values['碳水化合物'] * 4
    protein = nutrition_values['蛋白质'] * 4  
    fat = nutrition_values['脂肪'] * 9
    fiber = nutrition_values['纤维素'] * 2
    
    feature_vector.extend([carbs, protein, fat, fiber])
    
    # 患者基础信息特征
    patient_features = ['Baseline_Libre', 'Age', 'Gender', 'BMI', 'A1c', 'HOMA', 
                       'Insulin', 'TG', 'Cholesterol', 'HDL', 'Non HDL', 'LDL', 
                       'VLDL', 'CHO/HDL ratio', 'Fasting BG']
    
    for feature in patient_features:
        feature_vector.append(patient_data[feature])
    
    # 转换为DataFrame格式（TabPFN需要）
    feature_names = ['Carbs', 'Protein', 'Fat', 'Fiber'] + patient_features
    X = np.array(feature_vector, dtype=np.float64).reshape(1, -1)
    X_df = pd.DataFrame(X, columns=feature_names)
    
    if debug:
        print(f"    营养素: Carbs={carbs:.0f}, Protein={protein:.0f}, Fat={fat:.0f}, Fiber={fiber:.0f}")
        print(f"    完整特征向量前4个: {feature_vector[:4]}")
    
    random_seed = int(time())
    
    # 预测
    pred_iauc = model.predict(X_df)[0]
    
    if debug:
        print(f"    使用随机种子: {random_seed}")
        print(f"    预测结果: {pred_iauc:.2f}")
    
    return pred_iauc

def setup_matplotlib_chinese():
    candidates = [
        "Microsoft YaHei", "SimHei", "Arial Unicode MS",
        "Noto Sans CJK SC", "PingFang SC", "NotoSansCJKsc-Regular"
    ]
    available = {f.name: f.fname for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            rcParams["font.family"] = "sans-serif"
            rcParams["font.sans-serif"] = [name]
            rcParams["axes.unicode_minus"] = False
            print(f"matplotlib 中文字体已设置为: {name}")
            return

    # 按文件名在系统字体中尝试匹配常见中文字体
    try:
        sys_fonts = font_manager.findSystemFonts(fontpaths=None, fontext="ttf")
        sys_fonts += font_manager.findSystemFonts(fontpaths=None, fontext="ttc")
    except Exception:
        sys_fonts = font_manager.findSystemFonts(fontpaths=None, fontext="ttf")

    for fp in sys_fonts:
        fn = fp.lower()
        if any(k in fn for k in ("noto", "msyh", "simhei", "simsun", "pingfang", "hei", "zh")):
            prop = font_manager.FontProperties(fname=fp)
            name = prop.get_name()
            rcParams["font.family"] = "sans-serif"
            rcParams["font.sans-serif"] = [name]
            rcParams["axes.unicode_minus"] = False
            print(f"matplotlib 中文字体已设置为: {name} ({fp})")
            return

    # 兜底设置（如果系统确实没有中文字体，可能仍然无法显示中文）
    rcParams["font.sans-serif"] = ["SimHei"]
    rcParams["font.family"] = "sans-serif"
    rcParams["axes.unicode_minus"] = False
    print("未找到常见中文字体，已设置为 SimHei（若无效请安装中文字体）")


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    # 在程序启动处调用，启用 matplotlib 中文显示并修复负号显示问题
    setup_matplotlib_chinese()
    
    # 切换到数据目录
    original_dir = os.getcwd()
    os.chdir("cgmacros1.0/CGMacros")
    
    try:
        # 初始化TabPFN
        init()
        
        # 加载数据
        print("加载数据...")
        data_all_sub = get_data_all_sub()
        
        # 定义特征列
        feature_cols = ['Carbs', 'Protein', 'Fat', 'Fiber', 'Baseline_Libre', 'Age', 'Gender', 
                       'BMI', 'A1c', 'HOMA', 'Insulin', 'TG', 'Cholesterol', 'HDL', 'Non HDL', 
                       'LDL', 'VLDL', 'CHO/HDL ratio', 'Fasting BG']
        
        # 1. 训练早餐预测模型
        print("\n1. 训练早餐预测模型...")
        breakfast_mask = data_all_sub["Meal Type"] == 1
        X_breakfast = data_all_sub.loc[breakfast_mask, feature_cols]  # 保持DataFrame格式
        y_breakfast = data_all_sub.loc[breakfast_mask, 'iAUC'].values
        
        print(f"早餐数据集大小: {X_breakfast.shape}")
        
        # 训练TabPFN模型（使用DataFrame）
        breakfast_model = TabPFNRegressor(random_state=42)
        breakfast_model.fit(X_breakfast, y_breakfast)
        print("早餐预测模型训练完成!")
        
        # 切换回原目录以访问KB和其他文件
        os.chdir(original_dir)
        
        # 2. 初始化RAG系统
        print("\n2. 初始化RAG系统...")
        global client, embeddings, vector_store, prompt, graph
        
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
        
        # 构建RAG工作流
        graph_builder = StateGraph(State).add_sequence([analyze_query, retrieve, generate])
        graph_builder.add_edge(START, "analyze_query")
        graph = graph_builder.compile()
        
        print("RAG系统初始化完成!")
        
        # 切换回数据目录
        os.chdir("cgmacros1.0/CGMacros")
        
        # 3. 测试：随机选择3个不同受试者的早餐样本
        print("\n3. 开始测试...")
        unique_subjects = data_all_sub['sub'].unique()
        np.random.seed(int(time()))
        selected_subjects = np.random.choice(unique_subjects, size=3, replace=False)
        
        results = []
        
        for i, subject in enumerate(selected_subjects):
            print(f"\n--- 测试样本 {i+1}: 受试者 {subject} ---")
            
            # 获取该受试者的早餐数据
            subject_breakfast = data_all_sub[
                (data_all_sub['sub'] == subject) & (data_all_sub['Meal Type'] == 1)
            ].iloc[0]  # 取第一条记录
            
            # 提取患者信息（15个特征）
            patient_info_features = ['Baseline_Libre', 'Age', 'Gender', 'BMI', 'A1c', 'HOMA', 
                                   'Insulin', 'TG', 'Cholesterol', 'HDL', 'Non HDL', 'LDL', 
                                   'VLDL', 'CHO/HDL ratio', 'Fasting BG']
            patient_data = subject_breakfast[patient_info_features].to_dict()
            
            # 格式化患者信息
            patient_info_text = format_patient_info(patient_data)
            print(f"患者信息: {patient_info_text}")
            
            # 切换回原目录进行RAG预测
            os.chdir(original_dir)
            
            # RAG预测
            print("使用RAG预测营养素...")
            rag_result = graph.invoke({"information": patient_info_text})
            rag_nutrition = parse_nutrition_output(rag_result.get("answer", ""))
            print(f"RAG预测结果: {rag_nutrition}")
            
            # LLM预测
            print("使用普通LLM预测营养素...")
            llm_result = generate_llm_only(patient_info_text)
            llm_nutrition = parse_nutrition_output(llm_result)
            print(f"LLM预测结果: {llm_nutrition}")
            
            # 切换回数据目录进行模型预测
            os.chdir("cgmacros1.0/CGMacros")
            
            # 使用模型预测iAUC
            print("RAG营养素预测:")
            # rag_pred_iauc = predict_iauc_for_meal((X_breakfast, y_breakfast), patient_data, rag_nutrition['早餐'], debug=True)
            rag_pred_iauc = predict_iauc_for_meal(breakfast_model, patient_data, rag_nutrition['早餐'], debug=True)
            print("LLM营养素预测:")
            # llm_pred_iauc = predict_iauc_for_meal((X_breakfast, y_breakfast), patient_data, llm_nutrition['早餐'], debug=True)
            llm_pred_iauc = predict_iauc_for_meal(breakfast_model, patient_data, llm_nutrition['早餐'], debug=True)
            true_iauc = subject_breakfast['iAUC']
            
            print(f"RAG干涉后预测iAUC: {rag_pred_iauc:.2f}")
            print(f"LLM干涉后预测iAUC: {llm_pred_iauc:.2f}")
            print(f"无干涉iAUC: {true_iauc:.2f}")
            
            results.append({
                'subject': subject,
                'rag_pred': rag_pred_iauc,
                'llm_pred': llm_pred_iauc,
                'true_iauc': true_iauc,
                'rag_nutrition': rag_nutrition['早餐'],
                'llm_nutrition': llm_nutrition['早餐']
            })
        
        # 4. 可视化结果
        print("\n4. 生成可视化...")
        
        subjects = [r['subject'] for r in results]
        rag_preds = [r['rag_pred'] for r in results]
        llm_preds = [r['llm_pred'] for r in results]
        true_values = [r['true_iauc'] for r in results]
        
        # 创建对比图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 柱状图对比
        x = np.arange(len(subjects))
        width = 0.25
        
        ax1.bar(x - width, rag_preds, width, label='RAG预测', alpha=0.8)
        ax1.bar(x, llm_preds, width, label='LLM预测', alpha=0.8)
        ax1.bar(x + width, true_values, width, label='真实值', alpha=0.8)
        
        ax1.set_xlabel('受试者')
        ax1.set_ylabel('iAUC')
        ax1.set_title('iAUC预测对比')
        ax1.set_xticks(x)
        ax1.set_xticklabels(subjects)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 散点图对比
        ax2.scatter(true_values, rag_preds, label='RAG vs 无干涉', s=100, alpha=0.7)
        ax2.scatter(true_values, llm_preds, label='LLM vs 无干涉', s=100, alpha=0.7)
        
        # 添加y=x线
        min_val = min(min(true_values), min(rag_preds), min(llm_preds))
        max_val = max(max(true_values), max(rag_preds), max(llm_preds))
        ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='理想预测线')
        
        ax2.set_xlabel('无干涉iAUC')
        ax2.set_ylabel('预测iAUC')
        ax2.set_title('预测准确性对比')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 保存结果
        plt.savefig('../../plots/rag_vs_llm_iauc_prediction.png', dpi=300, bbox_inches='tight')
        
        # 5. 计算评估指标
        print("\n5. 评估指标:")
        rag_mae = mean_absolute_error(true_values, rag_preds)
        llm_mae = mean_absolute_error(true_values, llm_preds)
        rag_rmse = root_mean_squared_error(true_values, rag_preds)
        llm_rmse = root_mean_squared_error(true_values, llm_preds)
        rag_r = pearsonr(true_values, rag_preds)[0]
        llm_r = pearsonr(true_values, llm_preds)[0]
        
        print(f"RAG预测 - MAE: {rag_mae:.2f}, RMSE: {rag_rmse:.2f}, Pearson r: {rag_r:.3f}")
        print(f"LLM预测 - MAE: {llm_mae:.2f}, RMSE: {llm_rmse:.2f}, Pearson r: {llm_r:.3f}")
        
        # 详细结果表格
        print("\n详细结果:")
        print("受试者\tRAG预测\tLLM预测\t真实值\tRAG误差\tLLM误差")
        for r in results:
            rag_error = abs(r['rag_pred'] - r['true_iauc'])
            llm_error = abs(r['llm_pred'] - r['true_iauc'])
            print(f"{r['subject']}\t{r['rag_pred']:.1f}\t{r['llm_pred']:.1f}\t{r['true_iauc']:.1f}\t{rag_error:.1f}\t{llm_error:.1f}")
            
    finally:
        # 切换回原目录
        os.chdir(original_dir)


if __name__ == "__main__":
    main()

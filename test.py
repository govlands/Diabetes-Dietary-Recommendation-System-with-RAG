import torch
import numpy as np
from tabpfn import TabPFNRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from utils import get_data_all_sub, fetch_dataset_from_cgmacros
import pandas as pd

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CAT_INDICES = [6]  # Gender特征的索引位置

def load_breakfast_dataset():
    """加载早餐数据集"""
    print("=== 加载早餐数据集 ===")
    # 使用meal_type=1加载早餐数据
    X, y = fetch_dataset_from_cgmacros(meal_type=1)
    print(f"早餐数据形状: X={X.shape}, y={y.shape}")
    print(f"特征名称: ['Carbs', 'Protein', 'Fat', 'Fiber', 'Baseline_Libre', 'Age', 'Gender', 'BMI', 'A1c', 'HOMA', 'Insulin', 'TG', 'Cholesterol', 'HDL', 'Non HDL', 'LDL', 'VLDL', 'CHO/HDL ratio', 'Fasting BG']")
    print(f"前4个特征: ['Carbs', 'Protein', 'Fat', 'Fiber']")
    print(f"iAUC范围: {y.min():.2f} - {y.max():.2f}")
    
    # 检查Gender特征的取值
    gender_values = np.unique(X[:, 6])  # Gender在第6列
    print(f"Gender特征的唯一值: {gender_values}")
    print(f"Gender特征类型: {type(X[0, 6])}")
    
    return X, y

def train_regressors(X_train, y_train):
    """训练三个回归器"""
    print("\n=== 训练回归器 ===")
    
    # 检查Gender特征是否真的是分类特征
    gender_values = np.unique(X_train[:, 6])
    print(f"Gender特征的唯一值: {gender_values}")
    print(f"X_train数据类型: {X_train.dtype}")
    
    # 为CatBoost准备数据：将Gender列转换为整数
    X_train_cat = X_train.copy()
    X_train_cat[:, 6] = X_train_cat[:, 6].astype(int)  # 将Gender转为整数
    
    # 1. TabPFN回归器
    print("训练TabPFN回归器...")
    tabpfn_reg = TabPFNRegressor(
        categorical_features_indices=CAT_INDICES,
        device=DEVICE,
        n_estimators=4,
        random_state=42,
        ignore_pretraining_limits=True
    )
    tabpfn_reg.fit(X_train, y_train)
    
    # 2. CatBoost回归器 - 不使用分类特征，将所有特征视为数值特征
    print("训练CatBoost回归器...")
    catboost_reg = CatBoostRegressor(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        random_state=42,
        logging_level="Silent",
        allow_writing_files=False
        # 不指定cat_features，让CatBoost自动处理
    )
    catboost_reg.fit(X_train, y_train)
    
    # 3. XGBoost回归器
    print("训练XGBoost回归器...")
    xgb_reg = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        random_state=42,
        verbosity=0
    )
    xgb_reg.fit(X_train, y_train)
    
    return tabpfn_reg, catboost_reg, xgb_reg

def evaluate_regressors(models, X_test, y_test):
    """评估回归器性能"""
    print("\n=== 评估回归器性能 ===")
    model_names = ['TabPFN', 'CatBoost', 'XGBoost']
    
    for i, (model, name) in enumerate(zip(models, model_names)):
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        r, _ = pearsonr(y_test, y_pred)
        
        print(f"{name:>8}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}, R={r:.4f}")

def create_perturbed_samples(X, n_samples=10, perturbation_std=0.1):
    """
    创建扰动样本对
    
    参数:
    - X: 原始特征矩阵
    - n_samples: 要创建的样本对数
    - perturbation_std: 扰动的标准差（相对于特征标准差的比例）
    
    返回:
    - X_pairs: 形状为(n_samples, 2, n_features)的数组，每个样本对的两个样本
    """
    print(f"\n=== 创建{n_samples}对扰动样本 ===")
    
    # 随机选择n_samples个样本作为基础
    np.random.seed(42)
    base_indices = np.random.choice(len(X), size=n_samples, replace=False)
    base_samples = X[base_indices].copy()
    
    # 计算前4个特征的标准差
    feature_stds = np.std(X[:, :4], axis=0)
    print(f"前4个特征的标准差: {feature_stds}")
    
    # 为每个样本创建扰动版本
    X_pairs = np.zeros((n_samples, 2, X.shape[1]))
    
    for i in range(n_samples):
        # 第一个样本：原始样本
        X_pairs[i, 0] = base_samples[i].copy()
        
        # 第二个样本：扰动版本
        X_pairs[i, 1] = base_samples[i].copy()
        # 只扰动前4个特征
        perturbations1 = np.random.normal(0, perturbation_std * feature_stds, size=4)
        X_pairs[i, 1, :4] += perturbations1
        
        perturbations0 = np.random.normal(1, perturbation_std * feature_stds, size=4)
        X_pairs[i, 0, :4] += perturbations0
        
        # 确保扰动后的特征值不为负（营养成分不能为负）
        X_pairs[i, 1, :4] = np.maximum(X_pairs[i, 1, :4], 0.1)
    
    print(f"创建了{n_samples}对样本，每对除前4个特征外完全相同")
    print("扰动统计:")
    for j in range(4):
        feature_names = ['Carbs', 'Protein', 'Fat', 'Fiber']
        original_vals = X_pairs[:, 0, j]
        perturbed_vals = X_pairs[:, 1, j]
        avg_change = np.mean(np.abs(perturbed_vals - original_vals))
        print(f"  {feature_names[j]:>8}: 平均变化 = {avg_change:.3f}")
    
    return X_pairs

def test_prediction_consistency(models, X_pairs):
    """
    测试模型对扰动样本的预测一致性
    
    参数:
    - models: 训练好的模型列表
    - X_pairs: 扰动样本对
    """
    print(f"\n=== 测试预测一致性 ===")
    model_names = ['TabPFN', 'CatBoost', 'XGBoost']
    n_samples = X_pairs.shape[0]
    
    results = {}
    
    for i, (model, name) in enumerate(zip(models, model_names)):
        print(f"\n{name} 预测结果:")
        
        # 预测原始样本和扰动样本
        X_original = X_pairs[:, 0, :]  # 原始样本
        X_perturbed = X_pairs[:, 1, :]  # 扰动样本
        
        y_pred_original = model.predict(X_original)
        y_pred_perturbed = model.predict(X_perturbed)
        
        # 计算预测差异
        pred_differences = np.abs(y_pred_perturbed - y_pred_original)
        
        results[name] = {
            'original_preds': y_pred_original,
            'perturbed_preds': y_pred_perturbed,
            'differences': pred_differences
        }
        
        print(f"  预测差异统计:")
        print(f"    平均差异: {np.mean(pred_differences):.4f}")
        print(f"    最大差异: {np.max(pred_differences):.4f}")
        print(f"    最小差异: {np.min(pred_differences):.4f}")
        print(f"    标准差: {np.std(pred_differences):.4f}")
        
        # 显示每对样本的详细结果
        print(f"  详细结果:")
        print(f"    {'样本对':<6} {'原始预测':<12} {'扰动预测':<12} {'差异':<8}")
        print(f"    {'-'*6:<6} {'-'*12:<12} {'-'*12:<12} {'-'*8:<8}")
        for j in range(n_samples):
            print(f"    {j+1:>4}   {y_pred_original[j]:>10.4f}   {y_pred_perturbed[j]:>10.4f}   {pred_differences[j]:>6.4f}")
    
    return results

def compare_model_consistency(results):
    """比较不同模型的一致性"""
    print(f"\n=== 模型一致性对比 ===")
    
    model_names = list(results.keys())
    consistency_scores = {}
    
    for name in model_names:
        differences = results[name]['differences']
        # 一致性得分：平均预测差异的倒数（差异越小，一致性越高）
        avg_diff = np.mean(differences)
        consistency_score = 1.0 / (1.0 + avg_diff)  # 归一化到0-1
        consistency_scores[name] = consistency_score
        
        print(f"{name:>8}: 平均差异={avg_diff:>8.4f}, 一致性得分={consistency_score:>6.4f}")
    
    # 找出最一致的模型
    best_model = max(consistency_scores.items(), key=lambda x: x[1])
    print(f"\n最一致的模型: {best_model[0]} (得分: {best_model[1]:.4f})")
    
    return consistency_scores

def main():
    """主函数"""
    print("=" * 60)
    print("测试回归器对扰动样本的预测一致性")
    print("=" * 60)
    
    # 1. 加载早餐数据集
    X, y = load_breakfast_dataset()
    
    # 2. 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n训练集: {X_train.shape}, 测试集: {X_test.shape}")
    
    # 3. 训练三个回归器
    models = train_regressors(X_train, y_train)
    
    # 4. 评估模型性能
    evaluate_regressors(models, X_test, y_test)
    
    # 5. 创建扰动样本对
    n_test_samples = 15  # 可以调整测试样本数
    X_pairs = create_perturbed_samples(X_test, n_samples=n_test_samples, perturbation_std=0.05)
    
    # 6. 测试预测一致性
    results = test_prediction_consistency(models, X_pairs)
    
    # 7. 比较模型一致性
    consistency_scores = compare_model_consistency(results)
    
    print(f"\n{'='*60}")
    print("测试完成!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
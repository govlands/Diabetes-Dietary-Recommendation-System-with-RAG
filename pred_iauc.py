import torch
import numpy as np
from tabpfn import TabPFNRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from utils import fetch_dataset_from_cgmacros, get_save_path
import warnings
warnings.filterwarnings('ignore')

def load_dataset(meal_type=1, test_size=0.2, random_state=42):
    """加载指定meal type的数据集并划分训练测试集"""
    print(f"=== 加载meal type={meal_type}的数据集 ===")
    X, y = fetch_dataset_from_cgmacros(meal_type=meal_type)
    
    print(f"数据集形状: X={X.shape}, y={y.shape}")
    print(f"特征维度: {X.shape[1]}")
    print(f"目标变量(iAUC)范围: {y.min():.2f} - {y.max():.2f}")
    
    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def initialize_regressors(random_state=42, device="cuda" if torch.cuda.is_available() else "cpu"):
    """初始化四个回归器"""
    print(f"\n=== 初始化回归器 ===")
    print(f"使用设备: {device}")
    
    # 1. TabPFN回归器
    tabpfn_reg = TabPFNRegressor(
        categorical_features_indices=[6],  # Gender特征
        device=device,
        n_estimators=4,
        random_state=random_state,
        ignore_pretraining_limits=True
    )
    
    # 2. CatBoost回归器（基础配置）
    catboost_reg = CatBoostRegressor(
        random_state=random_state,
        logging_level="Silent",
        allow_writing_files=False
    )
    
    # 3. XGBoost回归器（基础配置）
    xgb_reg = XGBRegressor(
        random_state=random_state,
        verbosity=0
    )
    
    # 4. RandomForest回归器（基础配置）
    rf_reg = RandomForestRegressor(
        random_state=random_state,
        n_jobs=-1
    )
    
    regressors = {
        'TabPFN': tabpfn_reg,
        'CatBoost': catboost_reg,
        'XGBoost': xgb_reg,
        'RandomForest': rf_reg
    }
    
    print(f"初始化完成: {list(regressors.keys())}")
    return regressors

def get_param_grids():
    """定义各回归器的参数搜索空间"""
    param_grids = {
        'CatBoost': {
            'iterations': [100, 200, 500, 1000],
            'depth': [4, 6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'l2_leaf_reg': [1, 3, 5, 9],
            'subsample': [0.6, 0.8, 1.0],
            'rsm': [0.6, 0.8, 1.0]
        },
        'XGBoost': {
            'n_estimators': [100, 200, 500, 1000],
            'max_depth': [3, 6, 9, 12],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'reg_alpha': [0, 0.1, 1, 5],
            'reg_lambda': [1, 3, 5, 10]
        },
        'RandomForest': {
            'n_estimators': [100, 200, 500, 1000],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', 0.5, 0.8]
        }
    }
    return param_grids

def tune_regressor(regressor, param_grid, X_train, y_train, cv=5, n_iter=50, scoring='neg_mean_absolute_error', random_state=42):
    """对单个回归器进行随机搜索调参"""
    print(f"开始随机搜索调参...")
    
    random_search = RandomizedSearchCV(
        estimator=regressor,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        random_state=random_state,
        verbose=1
    )
    
    random_search.fit(X_train, y_train)
    
    print(f"最优参数: {random_search.best_params_}")
    print(f"最优CV得分: {-random_search.best_score_:.4f}")
    
    return random_search.best_estimator_

def train_and_tune_regressors(regressors, X_train, y_train):
    """训练和调优回归器"""
    print(f"\n=== 训练和调优回归器 ===")
    
    param_grids = get_param_grids()
    tuned_regressors = {}
    
    # TabPFN不需要调参，直接训练
    print(f"\n训练TabPFN...")
    tabpfn = regressors['TabPFN']
    tabpfn.fit(X_train, y_train)
    tuned_regressors['TabPFN'] = tabpfn
    print(f"TabPFN训练完成")
    
    # 其他回归器进行随机搜索
    for name in ['CatBoost', 'XGBoost', 'RandomForest']:
        print(f"\n调优{name}...")
        regressor = regressors[name]
        param_grid = param_grids[name]
        
        tuned_reg = tune_regressor(
            regressor, param_grid, X_train, y_train
        )
        tuned_regressors[name] = tuned_reg
        print(f"{name}调优完成")
    
    return tuned_regressors

def evaluate_regressors(regressors, X_test, y_test):
    """评估所有回归器在测试集上的表现"""
    print(f"\n=== 评估回归器性能 ===")
    
    results = {}
    
    for name, regressor in regressors.items():
        print(f"\n评估{name}...")
        
        # 预测
        y_pred = regressor.predict(X_test)
        
        # 计算指标
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        r, _ = pearsonr(y_test, y_pred)
        
        results[name] = {
            'predictions': y_pred,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'r': r
        }
        
        print(f"{name}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}, R={r:.4f}")
    
    return results

def find_best_regressor(results):
    """根据RMSE找出最优回归器"""
    best_name = min(results.keys(), key=lambda k: results[k]['rmse'])
    best_score = results[best_name]['rmse']
    
    print(f"\n=== 最优回归器 ===")
    print(f"最优模型: {best_name}")
    print(f"最优RMSE: {best_score:.4f}")
    
    return best_name, results[best_name]

def plot_best_predictions(best_name, best_result, y_test):
    """绘制最优回归器的预测vs真实散点图"""
    print(f"\n=== 绘制{best_name}预测结果 ===")
    
    y_pred = best_result['predictions']
    rmse = best_result['rmse']
    mae = best_result['mae']
    r2 = best_result['r2']
    r = best_result['r']
    
    plt.figure(figsize=(10, 8))
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 绘制散点图
    plt.scatter(y_test, y_pred, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    # 完美预测线
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='完美预测线')
    
    # 置信带
    errors = np.abs(y_test - y_pred)
    std_error = np.std(errors)
    range_vals = np.array([min_val, max_val])
    plt.fill_between(range_vals, range_vals - std_error, range_vals + std_error, 
                     alpha=0.2, color='red', label='±1 标准差')
    
    # 设置标签和标题
    plt.xlabel('真实iAUC', fontsize=14)
    plt.ylabel('预测iAUC', fontsize=14)
    plt.title(f'{best_name} - 预测 vs 真实值', fontsize=16, fontweight='bold')
    
    # 添加性能指标文本框
    metrics_text = f'RMSE = {rmse:.4f}\nMAE = {mae:.4f}\nR² = {r2:.4f}\nR = {r:.4f}'
    plt.text(0.05, 0.95, metrics_text, 
             transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
             fontsize=12, verticalalignment='top', fontweight='bold')
    
    # 添加样本数信息
    plt.text(0.95, 0.05, f'样本数: {len(y_test)}', 
             transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
             fontsize=10, horizontalalignment='right')
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 保存图片
    plot_path = get_save_path('plots', f'{best_name}_prediction_scatter', 'png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"图片已保存至: {plot_path}")

def print_summary_table(results):
    """打印所有回归器的性能汇总表"""
    print(f"\n{'='*70}")
    print(f"所有回归器性能汇总")
    print(f"{'='*70}")
    print(f"{'模型':<15} {'RMSE':<10} {'MAE':<10} {'R²':<10} {'R':<10}")
    print(f"{'-'*70}")
    
    for name, result in results.items():
        print(f"{name:<15} {result['rmse']:<10.4f} {result['mae']:<10.4f} "
              f"{result['r2']:<10.4f} {result['r']:<10.4f}")
    
    print(f"{'='*70}")

def main():
    """主函数"""
    print("="*80)
    print("iAUC预测回归器对比测试")
    print("="*80)
    
    # 1. 定义待测试数据集的meal type
    MEAL_TYPE = 5  # 早餐
    print(f"测试meal type: {MEAL_TYPE} ")
    
    # 2. 加载数据集并划分
    X_train, X_test, y_train, y_test = load_dataset(meal_type=MEAL_TYPE)
    
    # 3. 初始化四个回归器
    regressors = initialize_regressors()
    
    # 4. 训练和调优回归器
    tuned_regressors = train_and_tune_regressors(regressors, X_train, y_train)
    
    # 5. 评估所有回归器
    results = evaluate_regressors(tuned_regressors, X_test, y_test)
    
    # 6. 找出最优回归器
    best_name, best_result = find_best_regressor(results)
    
    # 7. 绘制最优回归器的预测散点图
    plot_best_predictions(best_name, best_result, y_test)
    
    # 8. 打印汇总表
    print_summary_table(results)
    
    print(f"\n{'='*80}")
    print(f"测试完成！最优模型: {best_name}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()

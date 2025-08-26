import os
import pickle
import matplotlib.pyplot as plt
import sklearn
import pandas as pd
import numpy as np
from scipy.stats import zscore
from scipy.stats import pearsonr
from scipy.stats import ttest_ind
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, r2_score
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from tabpfn_client import TabPFNRegressor, init
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, RandomizedSearchCV, GroupShuffleSplit
import scipy
import warnings
import numpy as np
from utils import *

def perform_grid_search(X, y, cv_folds=5, enable_tabpfn=False, randomized_threshold=200, random_search_iters=50, model_list=None):
    
    # Enlarged CatBoost grid (can be large -> will switch to RandomizedSearchCV if combinations exceed threshold)
    param_grids = {
        'RandomForest': {
            'n_estimators': [100, 200, 500],
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5, 10],
            'max_features': ['sqrt', 'log2', 0.3]
        },
        'XGBoost': {
            'n_estimators': [100, 200, 500, 1000],
            'max_depth': [3, 6, 9, 12],
            'learning_rate': [0.001, 0.01, 0.03, 0.05, 0.1],
            'subsample': [0.5, 0.7, 0.8, 1.0],
            'colsample_bytree': [0.4, 0.6, 0.8, 1.0],
            'colsample_bylevel': [0.4, 0.6, 0.8, 1.0],
            'reg_alpha': [0, 0.01, 0.1, 1, 5],
            'reg_lambda': [0.5, 1, 3, 5, 10],
            'gamma': [0, 0.1, 1, 5],
            'min_child_weight': [1, 3, 5, 10],
            'verbosity': [0]
        },
        'CatBoost': {
            'iterations': [100, 200, 500, 1000],
            'depth': [4, 6, 8, 10],
            'learning_rate': [0.001, 0.01, 0.03, 0.05, 0.1],
            'l2_leaf_reg': [1, 3, 5, 7, 9],
            'bagging_temperature': [0, 1, 2, 5],
            'border_count': [32, 64, 128],
            'random_strength': [0, 1, 5, 10],
            'subsample': [0.6, 0.8, 1.0],   # 行采样
            'rsm': [0.6, 0.8, 1.0]          # 特征子采样比例
        },
        'TabPFN': {}
    }
    
    # Define base models with pipelines
    base_models = {
        'RandomForest': RandomForestRegressor(random_state=42),
        'XGBoost': XGBRegressor(random_state=42),
        'CatBoost': CatBoostRegressor(
            random_state=42,
            logging_level="Silent",
            allow_writing_files=False
        )
    }
    if enable_tabpfn:
        base_models['TabPFN'] = TabPFNRegressor(random_state=42)
    
    # 如果调用者传入了 model_list，只处理其中存在于 base_models 的模型
    if model_list is not None:
        # 保持输入顺序，但过滤掉不可用的名字
        requested = list(model_list)
        available = [m for m in requested if m in base_models]
        missing = [m for m in requested if m not in base_models]
        if missing:
            print(f"Warning: the following requested models are not available and will be skipped: {missing}")
        if 'TabPFN' in requested and 'TabPFN' not in base_models:
            print("Warning: TabPFN requested in model_list but enable_tabpfn=False, skipping TabPFN")
        models_to_run = available
    else:
        models_to_run = list(base_models.keys())
    
    cv = KFold(n_splits=cv_folds, random_state=42, shuffle=True)
    
    optimized_models = {}
    cv_scores = {}
    
    for name in models_to_run:
        model = base_models[name]
        print(f"Optimizing {name}...")
        grid = param_grids.get(name, None)
        if grid:  # 有参数网格，使用 GridSearchCV 或 RandomizedSearchCV
            # 计算网格组合总数
            sizes = [len(v) for v in grid.values() if isinstance(v, (list, tuple, np.ndarray))]
            total_combinations = int(np.prod(sizes)) if sizes else 0
            if total_combinations > randomized_threshold:
                print(f"Large grid detected ({total_combinations} combos). Using RandomizedSearchCV with n_iter={random_search_iters}.")
                searcher = RandomizedSearchCV(
                    model,
                    param_distributions=grid,
                    n_iter=random_search_iters,
                    cv=cv,
                    scoring="neg_mean_absolute_error",
                    n_jobs=-1,
                    verbose=0,
                    random_state=42
                )
            else:
                searcher = GridSearchCV(
                    model,
                    grid,
                    cv=cv,
                    scoring="neg_mean_absolute_error",
                    n_jobs=-1,
                    verbose=0
                )
            searcher.fit(X, y)
            optimized_models[name] = searcher.best_estimator_
            # 保留 cv 结果：GridSearchCV 有 cv_results_，RandomizedSearchCV 也有
            try:
                cv_scores[name] = searcher.cv_results_
            except Exception:
                cv_scores[name] = None
            mean_score = searcher.best_score_
            print(f"{name} best score (neg MAE): {mean_score:.4f}")
            print(f"{name} best params: {searcher.best_params_}")
        else:
            # 对于没有网格的模型（例如 TabPFN），直接做交叉验证并在全量数据上训练返回模型
            scores = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error", n_jobs=1, verbose=0)
            cv_scores[name] = scores
            print(f"{name} CV Score (neg MAE): {scores.mean():.4f} ± {scores.std():.4f}")
            # 在全部数据上 fit 并返回已训练模型
            try:
                model.fit(X, y)
                optimized_models[name] = model
            except Exception as e:
                print(f"Warning: failed to fit {name} on full data: {e}")
                optimized_models[name] = model
    
    return optimized_models, cv_scores


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    original_dir = os.getcwd()
    data_dir = original_dir + '/cgmacros1.0/CGMacros'
    os.chdir(data_dir)
    pd.set_option('display.max_rows', None)
    enable_tabpfn = True
    model_list = ['CatBoost', 'TabPFN']
    
    data_all_sub = get_data_all_sub(data_dir)
    groups = data_all_sub['sub'].values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, holdout_idx = next(gss.split(data_all_sub, groups=groups))
    train_df = data_all_sub.iloc[train_idx].reset_index(drop=True)
    holdout_df = data_all_sub.iloc[holdout_idx].reset_index(drop=True)


    # Split data by meal type and train separate models
    feature_cols_spec = ['Carbs', 'Protein', 'Fat', 'Fiber', 'Baseline_Libre', 'Age', 'Gender', 'BMI', 'A1c', 'HOMA', 'Insulin', 'TG', 'Cholesterol',
                   'HDL', 'Non HDL', 'LDL', 'VLDL', 'CHO/HDL ratio', 'Fasting BG']
    feature_cols_uni = ['Carbs', 'Protein', 'Fat', 'Fiber', 'Baseline_Libre', 'Age', 'Gender', 'Meal Type', 'BMI', 'A1c', 'HOMA', 'Insulin', 'TG', 'Cholesterol',
                   'HDL', 'Non HDL', 'LDL', 'VLDL', 'CHO/HDL ratio', 'Fasting BG']
    
    meal_type_names = ["Breakfast", "Lunch", "Dinner"]
    meal_type_values = [1, 2, 3]
    meal_results = {}
    
    if enable_tabpfn: init()
    
    print(f"\n{'='*60}")
    print("Training separate models for each meal type")
    print(f"{'='*60}")
    
    # Check meal type distribution
    meal_type_counts = data_all_sub["Meal Type"].value_counts().sort_index()
    print("Meal type distribution:")
    for meal_val, count in meal_type_counts.items():
        meal_name = ["Breakfast", "Lunch", "Dinner"][int(meal_val) - 1]
        print(f"  {meal_name} (type {meal_val}): {count} samples")
    
    # 开始搜索
    for i, (meal_name, meal_val) in enumerate(zip(meal_type_names, meal_type_values)):
        print(f"\n{'='*40}")
        print(f"Processing {meal_name} (meal_type={meal_val})")
        print(f"{'='*40}")
        
        # Filter data for this meal type
        train_mask = train_df["Meal Type"] == meal_val
        test_mask = holdout_df["Meal Type"] == meal_val
        X_meal = train_df.loc[train_mask, feature_cols_spec]
        y_meal = train_df.loc[train_mask, 'iAUC']
        
        print(f"{meal_name} dataset shape: {X_meal.shape}")
        
        if len(X_meal) < 10:  # Skip if too few samples
            print(f"Warning: Too few samples for {meal_name} ({len(X_meal)}), skipping...")
            continue
        
        # Perform grid search for this meal type
        optimized_models, scores = perform_grid_search(X_meal, y_meal, enable_tabpfn=enable_tabpfn, model_list=model_list)
        meal_results[meal_name] = {
            'models': optimized_models,
            'scores': scores,
            'X': X_meal,
            'y': y_meal,
            'train_mask': train_mask,
            'test_mask': test_mask
        }
        
    print(f"\n{'='*40}")
    print(f"Processing all meal")
    print(f"{'='*40}")
    X_all = train_df.loc[:, feature_cols_uni]
    y_all = train_df.loc[:, 'iAUC']
    print(f"All meal dataset shape: {X_all.shape}")
    
    # Perform grid search for this meal type
    optimized_models, scores = perform_grid_search(X_all, y_all, enable_tabpfn=enable_tabpfn, model_list=model_list)
    meal_results["All meal"] = {
        'models': optimized_models,
        'scores': scores,
        'X': X_all,
        'y': y_all,
        'train_mask': [],
        'test_mask': []
    }

    # 开始测试工作：比较专门模型 vs 通用模型
    print(f"\n{'='*60}")
    print("专门模型 vs 通用模型测试")
    print(f"{'='*60}")
    
    # 创建models目录
    models_dir = "../../models/"
    os.makedirs(models_dir, exist_ok=True)
    
    final_models = {}
    final_results = {}
    
    # 找到最优通用模型
    uni_result = meal_results['All meal']
    best_uni_mae = float('inf')
    best_uni_model = None
    best_uni_name = None
    
    for model_name, model in uni_result['models'].items():
        # 计算该模型的MAE（基于交叉验证分数）
        scores_info = uni_result['scores'][model_name]
        if isinstance(scores_info, dict):
            # GridSearchCV结果
            mean_scores = np.array(scores_info.get("mean_test_score", []))
            if mean_scores.size > 0:
                best_cv_score = np.max(mean_scores)  # neg_mae，越大越好
                mae = -best_cv_score
            else:
                continue
        else:
            # 直接的CV分数数组
            mae = -np.mean(scores_info)
        
        print(f"  通用-{model_name}: MAE = {mae:.4f}")
        
        if mae < best_uni_mae:
            best_uni_mae = mae
            best_uni_model = model
            best_uni_name = model_name
    
    for meal_name in ["Breakfast", "Lunch", "Dinner"]:
        if meal_name not in meal_results:
            print(f"Warning: No data for {meal_name}, skipping...")
            continue
            
        print(f"\n{'='*40}")
        print(f"测试 {meal_name}")
        print(f"{'='*40}")
        
        # 获取专门结果
        spec_result = meal_results[meal_name]
        
        # 找到最优专门模型
        best_spec_mae = float('inf')
        best_spec_model = None
        best_spec_name = None
        
        for model_name, model in spec_result['models'].items():
            # 计算该模型的MAE（基于交叉验证分数）
            scores_info = spec_result['scores'][model_name]
            if isinstance(scores_info, dict):
                # GridSearchCV结果
                mean_scores = np.array(scores_info.get("mean_test_score", []))
                if mean_scores.size > 0:
                    best_cv_score = np.max(mean_scores)  # neg_mae，越大越好
                    mae = -best_cv_score
                else:
                    continue
            else:
                # 直接的CV分数数组
                mae = -np.mean(scores_info)
            
            print(f"  专门-{model_name}: MAE = {mae:.4f}")
            
            if mae < best_spec_mae:
                best_spec_mae = mae
                best_spec_model = model
                best_spec_name = model_name
        
        print(f"\n最优专门模型: {best_spec_name} (MAE: {best_spec_mae:.4f})")
        print(f"最优通用模型: {best_uni_name} (MAE: {best_uni_mae:.4f})")
        
        # 生成该餐的测试数据集
        test_mask = meal_results[meal_name]["test_mask"]
        X_meal_spec = holdout_df.loc[test_mask, feature_cols_spec]
        X_meal_uni = holdout_df.loc[test_mask, feature_cols_uni]
        y_meal = holdout_df.loc[test_mask, 'iAUC']
        
        print(f"测试数据集大小: {X_meal_uni.shape}")
        
        # 在测试数据集上比较两个模型的实际MAE
        # 专门模型：使用专门特征 (去掉Meal Type)
        # 检查是否为TabPFN模型，如果是则传入DataFrame，否则传入numpy array
        if 'TabPFN' in best_spec_name:
            y_pred_spec = best_spec_model.predict(X_meal_spec)  # DataFrame
        else:
            y_pred_spec = best_spec_model.predict(X_meal_spec.values)  # numpy array
        mae_spec_actual = np.mean(np.abs(y_meal.values - y_pred_spec))
        
        # 通用模型：使用全部特征 (包含Meal Type)
        if 'TabPFN' in best_uni_name:
            y_pred_uni = best_uni_model.predict(X_meal_uni)  # DataFrame
        else:
            y_pred_uni = best_uni_model.predict(X_meal_uni.values)  # numpy array
        mae_uni_actual = np.mean(np.abs(y_meal.values - y_pred_uni))
        
        print(f"\n实际测试结果:")
        print(f"  专门模型实际MAE: {mae_spec_actual:.4f}")
        print(f"  通用模型实际MAE: {mae_uni_actual:.4f}")
        
        # mae_spec_actual = best_spec_mae
        # mae_uni_actual = best_uni_mae
        
        # 选择最优模型
        if mae_spec_actual <= mae_uni_actual:
            final_model = best_spec_model
            final_model_name = f"{best_spec_name} (spec)"
            final_model_type = "specialized"
            final_mae = mae_spec_actual
            final_features = feature_cols_spec  # 专门模型特征
            X_final = X_meal_spec
            y_pred_final = y_pred_spec
            print(f"  → 选择专门模型 (MAE更优: {mae_spec_actual:.4f} vs {mae_uni_actual:.4f})")
        else:
            final_model = best_uni_model
            final_model_name = f"{best_uni_name} (uni)"
            final_model_type = "universal"
            final_mae = mae_uni_actual
            final_features = feature_cols_uni  # 通用模型特征
            X_final = X_meal_uni
            y_pred_final = y_pred_uni
            print(f"  → 选择通用模型 (MAE更优: {mae_uni_actual:.4f} vs {mae_spec_actual:.4f})")
        
        # 计算最终模型的评估指标
        pearson_corr = pearsonr(y_meal.values, y_pred_final)[0]
        rmse = root_mean_squared_error(y_meal.values, y_pred_final)
        r2 = r2_score(y_meal.values, y_pred_final)
        
        print(f"\n最终模型性能:")
        print(f"  模型: {final_model_name}")
        print(f"  MAE: {final_mae:.4f}")
        print(f"  Pearson correlation: {pearson_corr:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²: {r2:.4f}")
        
        # 保存模型信息
        model_info = {
            'model': final_model,
            'model_name': final_model_name,
            'model_type': final_model_type,
            'features': final_features,
            'hyperparams': final_model.get_params(),
            'mae': final_mae,
            'pearson_r': pearson_corr,
            'rmse': rmse,
            'r2': r2
        }
        
        final_models[meal_name] = model_info
        final_results[meal_name] = {
            'model_name': final_model_name,
            'model_type': final_model_type,
            'mae': final_mae,
            'pearson_r': pearson_corr,
            'rmse': rmse,
            'r2': r2
        }
        
        # 保存模型到本地
        model_filename = f"{models_dir}{meal_name.lower()}_{final_model_type}_model.pkl"
        with open(model_filename, 'wb') as f:
            pickle.dump(model_info, f)
        print(f"模型信息已保存到: {model_filename}")
        
        # 绘制散点图
        plt.figure(figsize=(10, 8))

        # 根据A1c值着色（使用 holdout 的 A1c，与 y_meal/y_pred_final 对齐）
        a1c_vals = holdout_df.loc[test_mask, "A1c"].values
        colors = []
        for v in a1c_vals:
            if v < 5.7:
                colors.append("blue")
            elif 5.7 <= v <= 6.4:
                colors.append("green")
            else:
                colors.append("red")

        # 分健康状态绘制
        for color, label in [("blue", "healthy"), ("green", "Pre-diabetes"), ("red", "Diabetes")]:
            indices = [i for i, c in enumerate(colors) if c == color]
            if indices:
                y_true_subset = [y_meal.values[i] for i in indices]
                y_pred_subset = [y_pred_final[i] for i in indices]
                plt.scatter(y_true_subset, y_pred_subset, c=color, s=30,
                           label=label, alpha=0.7)

        # 绘制对角线
        min_val = min(min(y_meal.values), min(y_pred_final))
        max_val = max(max(y_meal.values), max(y_pred_final))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Ideal prediction line')

        plt.xlabel('actual iAUC (mg/dl·h)')
        plt.ylabel('predicted iAUC (mg/dl·h)')
        plt.title(f'{meal_name} - {final_model_name}\n(r = {pearson_corr:.3f}, R² = {r2:.3f}, MAE = {final_mae:.2f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(f"{meal_name.lower()}_final_model_scatter.png", dpi=150, bbox_inches='tight')
        plt.show()
    
    # 保存最终结果汇总
    print(f"\n{'='*60}")
    print("最终模型选择汇总")
    print(f"{'='*60}")
    
    summary_data = []
    for meal_name, info in final_results.items():
        print(f"\n{meal_name}:")
        print(f"  选择的模型: {info['model_name']}")
        print(f"  模型类型: {info['model_type']}")
        print(f"  MAE: {info['mae']:.4f}")
        print(f"  Pearson r: {info['pearson_r']:.4f}")
        print(f"  R²: {info['r2']:.4f}")
        print(f"  RMSE: {info['rmse']:.4f}")
        
        # 准备保存的数据
        row = {
            'meal_type': meal_name,
            'selected_model': info['model_name'],
            'model_type': info['model_type'],
            'mae': info['mae'],
            'pearson_r': info['pearson_r'],
            'r2': info['r2'],
            'rmse': info['rmse']
        }
        summary_data.append(row)
    
    # 保存到CSV
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv("final_model_selection_summary.csv", index=False)
    print(f"\n最终模型选择汇总已保存到 final_model_selection_summary.csv")
    
    print(f"\n{'='*60}")
    print("所有测试完成！")
    print("生成的文件:")
    print("- breakfast_final_model_scatter.png: 早餐最终模型散点图")
    print("- lunch_final_model_scatter.png: 午餐最终模型散点图")
    print("- dinner_final_model_scatter.png: 晚餐最终模型散点图")
    print("- final_model_selection_summary.csv: 最终模型选择汇总")
    print("- ../../models/ 目录下的模型文件")
    print(f"{'='*60}")

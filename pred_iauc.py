import os
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
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, RandomizedSearchCV
# import statsmodels.api as sm
# import statsmodels.formula.api as smf
# import seaborn as sns
# from sklearn.cross_decomposition import PLSRegression
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import explained_variance_score
# from sklearn.ensemble import RandomForestRegressor
# from scipy.stats import zscore
# import glob
# import shap
import scipy


import warnings
import numpy as np

def areaUnderCurve(a, b):
    total = 0
    temp = 0
    for i in range(len(a)-1):
        if (b[i+1]-b[0]>=0) and (b[i]-b[0]>=0):
            temp = ((b[i]-b[0]+b[i+1]-b[0])/2)*(a[i+1]-a[i])
        elif (b[i+1]-b[0] < 0) and (b[i]-b[0] >= 0):
            temp = (b[i]-b[0])*((b[i]-b[0])/(b[i]-b[i+1])*(a[i+1]-a[i])/2)
        elif (b[i+1]-b[0] >= 0) and (b[i]-b[0] < 0):
            temp = (b[i+1]-b[0])*((b[i+1]-b[0])/(b[i+1]-b[i])*(a[i+1]-a[i])/2)
        elif (b[i]-b[0] < 0) and (b[i+1]-b[0] < 0):
            temp = 0
        total = total + temp
    return total

def calc_iauc(cgm, sampling_interval):
    a = []
    for i in range(len(cgm)):
        a.append(i * sampling_interval[i])
    return areaUnderCurve(a, cgm)

def calc_auc(cgm, sampling_interval):
    return np.trapz(cgm, dx=sampling_interval)

def perform_grid_search(X, y, cv_folds=5, enable_tabpfn=False, randomized_threshold=500, random_search_iters=50):
    
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
    
    cv = KFold(n_splits=cv_folds, random_state=42, shuffle=True)
    
    optimized_models = {}
    cv_scores = {}
    
    for name, model in base_models.items():
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

def get_data_all_sub():
    # 构建data_all_sub(训练集)
    if os.path.exists("data_all_sub.csv"):
        data_all_sub = pd.read_csv("data_all_sub.csv")
    else:
        data_all_sub = pd.DataFrame(columns = ["sub", "Libre GL", "Meal Type", "Carb", "Protein", "Fat", "Fiber"])
        hours = 2
        libre_samples = hours * 4 + 1

        for sub in sorted(os.listdir(".")):
            if sub[:8] != "CGMacros":
                continue
            data = pd.read_csv(os.path.join(sub, sub+'.csv'))
            # print(data)
            data_sub = pd.DataFrame(columns = ["sub", "Libre GL", "Meal Type", "Carb", "Protein", "Fat", "Fiber"])    
            for index in data[(data["Meal Type"] == "Breakfast") | (data["Meal Type"] == "breakfast") | (data["Meal Type"] == "Lunch") | (data["Meal Type"] == "lunch") | (data["Meal Type"] == "Dinner") | (data["Meal Type"] == "dinner")].index:
                data_meal = {}
                data_meal["sub"] = sub[-3:]
                data_meal["Libre GL"] = data["Libre GL"][index:index+135:15].to_list()
                if len(data_meal["Libre GL"]) < 9:
                    continue
                data_meal["iAUC"] = calc_iauc(data_meal["Libre GL"], [15 for i in range(libre_samples)])
                data_meal["AUC"] = calc_auc(data_meal["Libre GL"], 15)
                data_meal["Carb"] = data["Carbs"][index] * 4
                data_meal["Protein"] = data["Protein"][index] * 4
                data_meal["Fat"] = data["Fat"][index] * 9
                data_meal["Fiber"] = data["Fiber"][index] * 2
                data_meal["Calories"] = data["Calories"][index]
                x = data["Meal Type"][index]
                if x == "Breakfast" or x == "breakfast":
                    data_meal["Meal Type"] = 1
                elif x == "lunch" or x == "Lunch":
                    data_meal["Meal Type"] = 2
                else:
                    data_meal["Meal Type"] = 3
                data_sub = pd.concat([data_sub, pd.DataFrame([data_meal])], ignore_index=True)
            if data_sub["Carb"].iloc[0] == 24 and data_sub["Protein"].iloc[0] == 22 and data_sub["Fat"].iloc[0] == 10.5 and data_sub["Fiber"].iloc[0] == 0.0:
                data_sub = data_sub.iloc[1:]
            data_all_sub = pd.concat([data_all_sub, data_sub], ignore_index=True)
        # print(data_all_sub)
        data_all_sub = data_all_sub[data_all_sub["iAUC"] > 0]
        data_all_sub.reset_index(inplace=True)

        # 记录病人数据
        df = pd.read_csv("bio.csv")
        a1c = df["A1c PDL (Lab)"].dropna().to_numpy()
        fasting_glucose = df["Fasting GLU - PDL (Lab)"].dropna().to_numpy()
        fasting_insulin = df["Insulin "].dropna().to_numpy() # in uIU/mL (ideal range: 2.6 - 24.9)
        fasting_insulin = [float(str(x).strip(' (low)')) for x in fasting_insulin]

        HOMA = (fasting_insulin * fasting_glucose)/405

        tg = df["Triglycerides"].dropna().to_numpy()
        cholesterol = df["Cholesterol"].dropna().to_numpy()
        HDL = df["HDL"].dropna().to_numpy()
        non_HDL = df["Non HDL "].dropna().to_numpy()
        ldl = df["LDL (Cal)"].dropna().to_numpy()
        vldl = df["VLDL (Cal)"].dropna().to_numpy()
        cho_hdl_ratio = df["Cho/HDL Ratio"].dropna().to_numpy()

        patients = []
        for i in range(len(a1c)):
            if a1c[i] < 5.7:
                patients.append("H")
            if a1c[i] >= 5.7 and a1c[i] <=6.4:
                patients.append("P")
            if a1c[i] > 6.4:
                patients.append("T2D")
        patients = np.array(patients)
        h_index = np.where(patients == "H")[0]
        p_index = np.where(patients == "P")[0]
        t_index = np.where(patients == "T2D")[0]

        weights = df["Body weight "].dropna().to_numpy()
        heights = df["Height "].dropna().to_numpy()
        weight_kg = weights * 0.453592
        total_heights = []    
        for i in range(len(heights)):
            inches = heights[i]
            h = float(inches)
            total_heights.append(h * 0.0254)
        BMI = []
        for height, weight in zip(total_heights, weight_kg):
            bmi = weight/(height**2)
            BMI.append(bmi)
        BMI = np.array(BMI)

        age = df["Age"].to_numpy()
        gender = df["Gender"].to_list()
        gender = [1 if x == 'M'  else -1 for x in gender]

        libre_data = data_all_sub["Libre GL"]
        interp_gl = []
        for i in range(len(data_all_sub["Libre GL"])):
            interp_gl.append(data_all_sub["Libre GL"][i][0])
        data_all_sub["Baseline_Libre"] = interp_gl

        subjects = data_all_sub["sub"].unique()

        new_age = []
        new_gender = []
        new_BMI = []
        new_a1c = []
        new_HOMA = []
        new_fasting_insulin = []
        new_tg = []
        new_cholestrol = []
        new_HDL = []
        new_non_HDL = []
        new_ldl = []
        new_vldl = []
        new_cho_hdl_ratio = []
        new_fasting_glucose = []

        for i in range(len(subjects)):
            match_length = len(data_all_sub[data_all_sub["sub"] == subjects[i]])
            new_age.extend([age[i]] * match_length)
            new_gender.extend([gender[i]] * match_length)
            new_BMI.extend([BMI[i]] * match_length)
            new_a1c.extend([a1c[i]] * match_length)
            new_HOMA.extend([HOMA[i]] * match_length)
            new_fasting_insulin.extend([fasting_insulin[i]] * match_length)
            new_tg.extend([tg[i]] * match_length)
            new_cholestrol.extend([cholesterol[i]] * match_length)
            new_HDL.extend([HDL[i]] * match_length)
            new_non_HDL.extend([non_HDL[i]] * match_length)
            new_ldl.extend([ldl[i]] * match_length)
            new_vldl.extend([vldl[i]] * match_length)
            new_cho_hdl_ratio.extend([cho_hdl_ratio[i]] * match_length)
            new_fasting_glucose.extend([fasting_glucose[i]] * match_length)


        data_all_sub["Age"] = new_age
        data_all_sub["Gender"] = new_gender
        data_all_sub["BMI"] = new_BMI
        data_all_sub["A1c"] = new_a1c
        data_all_sub["HOMA"] = new_HOMA
        data_all_sub["Insulin"] = new_fasting_insulin
        data_all_sub["TG"] = new_tg
        data_all_sub["Cholesterol"] = new_cholestrol
        data_all_sub["HDL"] = new_HDL
        data_all_sub["Non HDL"] = new_non_HDL
        data_all_sub["LDL"] = new_ldl
        data_all_sub["VLDL"] = new_vldl
        data_all_sub["CHO/HDL ratio"] = new_cho_hdl_ratio
        data_all_sub["Fasting BG"] = new_fasting_glucose

        data_all_sub["Carbs"] = data_all_sub["Carb"] * 4
        data_all_sub["Protein"] = data_all_sub["Protein"] * 4
        data_all_sub["Fats"] = data_all_sub["Fat"] * 9
        data_all_sub["Fiber"] = data_all_sub["Fiber"] * 2
        data_all_sub["Net Carb"] = data_all_sub["Carb"] - data_all_sub["Fiber"]
        data_all_sub.to_csv("data_all_sub.csv")
    return data_all_sub

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    os.chdir("cgmacros1.0/CGMacros")
    pd.set_option('display.max_rows', None)
    enable_tabpfn = False
    
    data_all_sub = get_data_all_sub()

    # Split data by meal type and train separate models
    feature_cols = ['Carbs', 'Protein', 'Fat', 'Fiber', 'Baseline_Libre', 'Age', 'Gender', 'BMI', 'A1c', 'HOMA', 'Insulin', 'TG', 'Cholesterol',
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
    
    for i, (meal_name, meal_val) in enumerate(zip(meal_type_names, meal_type_values)):
        print(f"\n{'='*40}")
        print(f"Processing {meal_name} (meal_type={meal_val})")
        print(f"{'='*40}")
        
        # Filter data for this meal type
        meal_mask = data_all_sub["Meal Type"] == meal_val
        X_meal = data_all_sub.loc[meal_mask, feature_cols]
        y_meal = data_all_sub.loc[meal_mask, 'iAUC']
        
        print(f"{meal_name} dataset shape: {X_meal.shape}")
        
        if len(X_meal) < 10:  # Skip if too few samples
            print(f"Warning: Too few samples for {meal_name} ({len(X_meal)}), skipping...")
            continue
        
        # Perform grid search for this meal type
        optimized_models, scores = perform_grid_search(X_meal, y_meal, enable_tabpfn=enable_tabpfn)
        meal_results[meal_name] = {
            'models': optimized_models,
            'scores': scores,
            'X': X_meal,
            'y': y_meal,
            'mask': meal_mask
        }

    # Plot comparison results for all meal types
    
    # 1) Create MAE comparison plot across meal types and models
    fig, axes = plt.subplots(1, len(meal_results), figsize=(15, 5))
    if len(meal_results) == 1:
        axes = [axes]
    
    all_predictions = []
    all_ground_truths = []
    all_colors = []
    all_meal_labels = []
    
    for idx, (meal_name, result) in enumerate(meal_results.items()):
        scores = result['scores']
        model_names = list(scores.keys())
        mae_means = []
        mae_stds = []
        
        # Calculate MAE for each model
        for name in model_names:
            s = scores[name]
            if isinstance(s, dict):
                mean_tests = np.array(s.get("mean_test_score", []))
                std_tests = np.array(s.get("std_test_score", []))
                if mean_tests.size > 0:
                    best_idx = int(np.argmax(mean_tests))
                    best_mean = mean_tests[best_idx]
                    best_std = std_tests[best_idx] if std_tests.size > 0 else 0.0
                    mae_means.append(-best_mean)
                    mae_stds.append(best_std)
                else:
                    mae_means.append(np.nan)
                    mae_stds.append(np.nan)
            else:
                arr = np.array(s)
                mae_means.append(-arr.mean())
                mae_stds.append(arr.std())
        
        # Plot MAE comparison for this meal type
        ax = axes[idx]
        ax.errorbar(range(len(model_names)), mae_means, yerr=mae_stds, marker='o', linestyle='-', capsize=4)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=30)
        ax.set_ylabel("MAE")
        ax.set_title(f"{meal_name} Model Comparison")
        ax.grid(True, alpha=0.3)
        
        # Find best model for this meal type
        best_idx = int(np.argmin(mae_means))
        best_name = model_names[best_idx]
        print(f"\nBest model for {meal_name}: {best_name} (MAE={mae_means[best_idx]:.4f} ± {mae_stds[best_idx]:.4f})")
        
        # Get best model and make predictions
        if best_name == "TabPFN":
            best_model = TabPFNRegressor(random_state=42)
        else:
            best_model = result['models'].get(best_name)
        
        X_meal = result['X']
        y_meal = result['y']
        meal_mask = result['mask']
        
        # Train and predict
        best_model.fit(X_meal.values, y_meal.values)
        y_pred = best_model.predict(X_meal.values)
        
        # Calculate metrics
        pearson_corr = pearsonr(y_meal.values, y_pred)[0]
        rmse = root_mean_squared_error(y_meal.values, y_pred)
        r2 = r2_score(y_meal.values, y_pred)
        print(f"  Pearson correlation: {pearson_corr:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R^2: {r2:.4f}")
        
        # Collect data for combined scatter plot
        all_predictions.extend(y_pred)
        all_ground_truths.extend(y_meal.values)
        all_meal_labels.extend([meal_name] * len(y_pred))
        
        # Assign colors based on A1c values for this meal type
        a1c_vals = data_all_sub.loc[meal_mask, "A1c"].values
        meal_colors = []
        for v in a1c_vals:
            if v < 5.7:
                meal_colors.append("blue")
            elif 5.7 <= v <= 6.4:
                meal_colors.append("green")
            else:
                meal_colors.append("red")
        all_colors.extend(meal_colors)
    
    plt.tight_layout()
    plt.savefig("meal_type_mae_comparison.png", dpi=150)
    plt.show()
    
    # 2) Create combined scatter plot for all meal types
    fig, axes = plt.subplots(1, len(meal_results), figsize=(18, 5))
    if len(meal_results) == 1:
        axes = [axes]
    
    meal_colors_map = {"Breakfast": "orange", "Lunch": "purple", "Dinner": "brown"}
    
    # Individual scatter plots for each meal type
    for idx, (meal_name, result) in enumerate(meal_results.items()):
        ax = axes[idx]
        meal_mask = result['mask']
        
        # Get predictions for this meal type
        start_idx = sum(len(meal_results[m]['y']) for m in list(meal_results.keys())[:idx])
        end_idx = start_idx + len(result['y'])
        
        meal_preds = all_predictions[start_idx:end_idx]
        meal_truths = all_ground_truths[start_idx:end_idx]
        meal_colors_health = all_colors[start_idx:end_idx]
        
        # Plot by health status
        for col, lab in [("blue","healthy"), ("green","preD"), ("red","T2D")]:
            indices = [i for i, c in enumerate(meal_colors_health) if c == col]
            if indices:
                meal_pred_subset = [meal_preds[i] for i in indices]
                meal_truth_subset = [meal_truths[i] for i in indices]
                ax.scatter(meal_truth_subset, meal_pred_subset, c=col, s=15, label=lab, alpha=0.7)
        
        ax.set_xlabel("Ground truth iAUC (mg/dl.h)")
        ax.set_ylabel("Predicted iAUC (mg/dl.h)")
        ax.set_title(f"{meal_name} Predictions")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Add correlation as text
        corr = pearsonr(meal_truths, meal_preds)[0]
        ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig("meal_type_scatter_comparison.png", dpi=150)
    plt.show()
    
    # 3) Create overall combined scatter plot
    plt.figure(figsize=(10, 8))
    
    # Plot by meal type with different markers
    markers = ['o', 's', '^']
    for idx, (meal_name, result) in enumerate(meal_results.items()):
        start_idx = sum(len(meal_results[m]['y']) for m in list(meal_results.keys())[:idx])
        end_idx = start_idx + len(result['y'])
        
        meal_preds = all_predictions[start_idx:end_idx]
        meal_truths = all_ground_truths[start_idx:end_idx]
        
        plt.scatter(meal_truths, meal_preds, 
                   c=meal_colors_map[meal_name], 
                   marker=markers[idx], 
                   s=20, 
                   label=meal_name, 
                   alpha=0.7)
    
    plt.xlabel("Ground truth iAUC (mg/dl.h)")
    plt.ylabel("Predicted iAUC (mg/dl.h)")
    plt.title("Combined Predictions by Meal Type")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add overall correlation
    overall_corr = pearsonr(all_ground_truths, all_predictions)[0]
    plt.text(0.05, 0.95, f'Overall r = {overall_corr:.3f}', 
             transform=plt.gca().transAxes,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig("combined_meal_type_scatter.png", dpi=150)
    plt.show()
    
    # 4) Save results to CSV files
    for meal_name, result in meal_results.items():
        meal_mask = result['mask']
        start_idx = sum(len(meal_results[m]['y']) for m in list(meal_results.keys())[:idx])
        end_idx = start_idx + len(result['y'])
        
        meal_preds = all_predictions[start_idx:end_idx]
        meal_truths = all_ground_truths[start_idx:end_idx]
        meal_colors_health = all_colors[start_idx:end_idx]
        
        status = []
        for c in meal_colors_health:
            if c == "blue":
                status.append("healthy")
            elif c == "green":
                status.append("preD")
            else:
                status.append("T2D")
        
        out_df = pd.DataFrame({
            "ground truth": meal_truths,
            "prediction": meal_preds,
            "health status": status,
            "meal_type": meal_name
        })
        out_df.to_csv(f"iAUC_{meal_name.lower()}.csv", index=False)
        print(f"Saved {meal_name} predictions to iAUC_{meal_name.lower()}.csv")
    
    # Save combined results
    combined_df = pd.DataFrame({
        "ground truth": all_ground_truths,
        "prediction": all_predictions,
        "meal_type": all_meal_labels
    })
    combined_df.to_csv("iAUC_combined.csv", index=False)
    print("Saved combined predictions to iAUC_combined.csv")
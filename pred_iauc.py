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
        # 'RandomForest': {
        #     'n_estimators': [100, 200, 500],
        #     'max_depth': [None, 5, 10, 20],
        #     'min_samples_split': [2, 5, 10],
        #     'max_features': ['sqrt', 'log2', 0.3]
        # },
        # 'XGBoost': {
        #     'n_estimators': [100, 200, 500],
        #     'max_depth': [3, 6, 9],
        #     'learning_rate': [0.01, 0.05, 0.1, 0.2],
        #     'subsample': [0.6, 0.8, 1.0],
        #     'colsample_bytree': [0.6, 0.8, 1.0]
        # },
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
        'RandomForest': RandomForestRegressor(random_state=random_seed),
        'XGBoost': XGBRegressor(random_state=random_seed),
        'CatBoost': CatBoostRegressor(
            random_state=random_seed,
            logging_level="Silent",
            allow_writing_files=False
        )
    }
    if enable_tabpfn:
        base_models['TabPFN'] = TabPFNRegressor(random_state=random_seed)
    
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
                print(f"Large CatBoost grid detected ({total_combinations} combos). Using RandomizedSearchCV with n_iter={random_search_iters}.")
                searcher = RandomizedSearchCV(
                    model,
                    param_distributions=grid,
                    n_iter=random_search_iters,
                    cv=cv,
                    scoring="neg_mean_absolute_error",
                    n_jobs=-1,
                    verbose=0,
                    random_state=random_seed
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
    os.chdir("cgmacros1.0/CGMacros")
    pd.set_option('display.max_rows', None)
    random_seed = 42
    enable_tabpfn = False
    
    # 构建data_all_sub(训练集)
    if os.path.exists("data_all_sub.csv"):
        data_all_sub = pd.read_csv("data_all_sub.csv")
    else:
        data_all_sub = pd.DataFrame(columns = ["sub", "Libre GL", "Carb", "Protein", "Fat", "Fiber"])
        hours = 2
        libre_samples = hours * 4 + 1

        for sub in sorted(os.listdir(".")):
            if sub[:8] != "CGMacros":
                continue
            data = pd.read_csv(os.path.join(sub, sub+'.csv'))
            # print(data)
            data_sub = pd.DataFrame(columns = ["sub", "Libre GL", "Carb", "Protein", "Fat", "Fiber"])    
            for index in data[(data["Meal Type"] == "Breakfast") | (data["Meal Type"] == "breakfast")].index:
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

    X = data_all_sub[['sub', 'Carbs', 'Protein', 'Fat', 'Fiber', 'Baseline_Libre', 'Age', 'Gender', 'BMI', 'A1c', 'HOMA', 'Insulin', 'TG', 'Cholesterol',
    'HDL', 'Non HDL', 'LDL', 'VLDL', 'CHO/HDL ratio', 'Fasting BG']].iloc[:, 1:]
    y = data_all_sub['iAUC']

    if enable_tabpfn: init()
    optimized_models, scores = perform_grid_search(X, y, enable_tabpfn=enable_tabpfn)

    # 绘制各模型 MAE 折线图，选择最优模型并保存预测结果与散点图

    # 1) 计算每个模型的 MAE（scores 中为 neg_mean_absolute_error）
    model_names = list(scores.keys())
    mae_means = []
    mae_stds = []
    for name in model_names:
        s = scores[name]
        if isinstance(s, dict):
            # s 是 cv_results_，取 best 的 mean_test_score / std_test_score
            mean_tests = np.array(s.get("mean_test_score", []))
            std_tests = np.array(s.get("std_test_score", []))
            if mean_tests.size > 0:
                best_idx = int(np.argmax(mean_tests))
                best_mean = mean_tests[best_idx]
                best_std = std_tests[best_idx] if std_tests.size > 0 else 0.0
                mae_means.append(-best_mean)   # 记得 mean_test_score 是 neg_mean_absolute_error
                mae_stds.append(best_std)
            else:
                mae_means.append(np.nan)
                mae_stds.append(np.nan)
        else:
            # s 是 cross_val_score 的返回值（array-like），元素为 neg_mean_absolute_error
            arr = np.array(s)
            mae_means.append(-arr.mean())
            mae_stds.append(arr.std())

    plt.figure(figsize=(8,4))
    plt.errorbar(range(len(model_names)), mae_means, yerr=mae_stds, marker='o', linestyle='-', capsize=4)
    plt.xticks(range(len(model_names)), model_names, rotation=30)
    plt.ylabel("MAE")
    plt.title("Model MAE Comparison (CV)")
    plt.tight_layout()
    plt.savefig("model_mae_comparison.png", dpi=150)
    plt.show()

    # 2) 选取 MAE 最低的模型名
    best_idx = int(np.argmin(mae_means))
    best_name = model_names[best_idx]
    print(f"Best model by CV MAE: {best_name} (MAE={mae_means[best_idx]:.4f} ± {mae_stds[best_idx]:.4f})")

    # 3) 获取该模型的 estimator（包含 TabPFN）
    if best_name == "TabPFN":
        best_model = TabPFNRegressor(random_state=42)
    else:
        # optimized_models 存在 RandomForest, XGBoost, CatBoost
        best_model = optimized_models.get(best_name)
        if best_model is None:
            # 兜底：如果名字不在 optimized_models（不太可能），尝试实例化 by name
            if best_name == "RandomForest":
                best_model = RandomForestRegressor(random_state=random_seed)
            elif best_name == "XGBoost":
                best_model = XGBRegressor(random_state=random_seed)
            elif best_name == "CatBoost":
                best_model = CatBoostRegressor(random_state=random_seed)
            else:
                raise ValueError(f"Unknown best model: {best_name}")

    # 4) 训练最佳模型并在整个 X 上预测
    X_values = X.values
    y_values = y.values
    best_model.fit(X_values, y_values)
    y_pred = best_model.predict(X_values)

    # 5) 根据 A1c 为每个样本分配颜色/健康状态（使用 data_all_sub 中的 A1c 列）
    a1c_vals = data_all_sub["A1c"].values
    colors = []
    status = []
    for v in a1c_vals:
        if v < 5.7:
            colors.append("blue"); status.append("healthy")
        elif 5.7 <= v <= 6.4:
            colors.append("green"); status.append("preD")
        else:
            colors.append("red"); status.append("T2D")
    colors = np.array(colors)
    status = np.array(status)

    # 6) 计算并打印评估指标
    pearson_corr = pearsonr(y_values, y_pred)[0]
    rmse = root_mean_squared_error(y_values, y_pred)
    r2 = r2_score(y_values, y_pred)
    print(f"Pearson correlation: {pearson_corr:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R^2: {r2:.4f}")

    # 7) 保存预测结果到 iAUC.csv
    out_df = pd.DataFrame({
        "ground truth": y_values,
        "prediction": y_pred,
        "health status": status
    })
    out_df.to_csv("iAUC.csv", index=False)
    print("Saved predictions to iAUC.csv")

    # 8) 绘制该模型的预测散点图，按健康状态上色
    plt.figure(figsize=(6,6))
    for col, lab in [("blue","healthy"), ("green","preD"), ("red","T2D")]:
        idx = np.where(colors == col)[0]
        if len(idx) > 0:
            plt.scatter(y_values[idx], y_pred[idx], c=col, s=10, label=lab)

    plt.ylabel("Predicted iAUC (mg/dl.h)")
    plt.xlabel("Ground truth iAUC (mg/dl.h)")
    plt.title(f"{best_name} predictions (corr={pearson_corr:.2f})")
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("best_model_scatter.png", dpi=150)
    plt.show()
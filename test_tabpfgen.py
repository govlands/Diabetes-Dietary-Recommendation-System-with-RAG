from tabpfgen import TabPFGen
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
from catboost import CatBoostRegressor
from tabpfn import TabPFNRegressor
import warnings
import numpy as np
from pred_iauc import get_data_all_sub
from sklearn.model_selection import train_test_split
from datetime import datetime


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    os.chdir("cgmacros1.0/CGMacros")
    pd.set_option('display.max_rows', None)
    
    # Initialize generator
    generator = TabPFGen(n_sgld_steps=500, device='auto')
    
    data_all_sub = get_data_all_sub()
    feature_cols = ['Carbs', 'Protein', 'Fat', 'Fiber', 'Baseline_Libre', 'Age', 'Gender', 'BMI', 'A1c', 'HOMA', 'Insulin', 'TG', 'Cholesterol',
                   'HDL', 'Non HDL', 'LDL', 'VLDL', 'CHO/HDL ratio', 'Fasting BG']
    meal_name = "Breakfast"
    meal_type = 1
    mask = data_all_sub["Meal Type"] == meal_type
    X = data_all_sub.loc[mask, feature_cols].to_numpy()
    y = data_all_sub.loc[mask, 'iAUC'].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_real, y_real = X_train, y_train
    print(f"X.shape:{X.shape}")
    
    ratios = [0, 0.25, 0.5, 0.75, 1]
    N =7
    num_real = X_real.shape[0]
    results = {r:{} for r in ratios}
    
    X_synth, y_synth = generator.generate_regression(
        X_real, y_real,
        n_samples=2 * num_real * max(ratios),
    )
    
    for r in ratios:
        test_info = {}
        test_info['ratio'] = r
        num_synth = int(num_real * r)
        print(f"\n=== Ratio {r} ({int(r*100)}%) | num_real={num_real} | num_synth={num_synth} ===")
        maes = []
        rmses = []
        ps = []
        r2s = []
        for repeat in range(N):
            seed = 1000*repeat + int(r*10)
            print(f"  Repeat {repeat+1}/{N}, seed={seed} ...")
            rng = np.random.default_rng(seed)
            if num_synth > 0:
                available = X_synth.shape[0]
                idx = rng.choice(available, size=num_synth, replace=False)
                X_synth_subset = X_synth[idx]
                y_synth_subset = y_synth[idx]
                print(f"    Selected synthetic subset shapes: X {X_synth_subset.shape}, y {y_synth_subset.shape}")

            aug_on = num_synth > 0
            X_aug = np.vstack((X_real, X_synth_subset)) if aug_on else X_real
            y_aug = np.concatenate((y_real, y_synth_subset)) if aug_on else y_real

            print(f"    Augmented training data shapes: X_aug {X_aug.shape}, y_aug {y_aug.shape}")
            # model_name = 'TabPFN'
            # model = TabPFNRegressor(random_state=seed)
            model_name = 'CatBoost'
            model = CatBoostRegressor(random_state=seed, subsample=1.0, rsm=1.0, random_strength=0, learning_rate=0.1, l2_leaf_reg=7, iterations=200, depth=8, border_count=32, bagging_temperature=0, logging_level='Silent')
            print("    Training TabPFNRegressor...")
            model.fit(X_aug, y_aug)
            print("    Training complete. Predicting on test set...")
            y_pred = model.predict(X_test)

            mae_val = mean_absolute_error(y_test, y_pred)
            rmse_val = root_mean_squared_error(y_test, y_pred)
            p_val = pearsonr(y_test, y_pred)[0]
            r2_val = r2_score(y_test, y_pred)

            maes.append(mae_val)
            rmses.append(rmse_val)
            ps.append(p_val)
            r2s.append(r2_val)

            print(f"    Repeat results -> MAE: {mae_val:.4f}, RMSE: {rmse_val:.4f}, Pearson r: {p_val:.4f}, R2: {r2_val:.4f}")

        test_info['mae'] = np.mean(maes)
        test_info['rmse'] = np.mean(rmses)
        test_info['p'] = np.mean(ps)
        test_info['r2'] = np.mean(r2s)
        results[r] = test_info
        print(f"  Summary for ratio {int(r*100)}% -> MAE: {test_info['mae']:.4f}, RMSE: {test_info['rmse']:.4f}, Pearson r: {test_info['p']:.4f}, R2: {test_info['r2']:.4f}")

    # Prepare metrics for plotting
    maes = [results[r]['mae'] for r in ratios]
    rmses = [results[r]['rmse'] for r in ratios]
    rs = [results[r]['p'] for r in ratios]
    r2s = [results[r]['r2'] for r in ratios]

    # Pretty x labels
    x = np.array(ratios)
    xticks = [f"{int(r*100)}%" for r in ratios]

    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.ravel()

    print("\nAll ratios processed. Preparing plots...")

    axs[0].plot(x, maes, marker='o', linestyle='-')
    axs[0].set_title('MAE vs Synthetic Ratio')
    axs[0].set_xlabel('Synthetic ratio')
    axs[0].set_ylabel('MAE')
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(xticks)
    axs[0].grid(True)
    best_idx = int(np.argmin(maes))
    axs[0].annotate(f"best: {xticks[best_idx]}\n{maes[best_idx]:.3f}",
                    xy=(x[best_idx], maes[best_idx]), xytext=(0, -30),
                    textcoords='offset points', ha='center', arrowprops=dict(arrowstyle='->'))

    axs[1].plot(x, rmses, marker='o', linestyle='-', color='tab:orange')
    axs[1].set_title('RMSE vs Synthetic Ratio')
    axs[1].set_xlabel('Synthetic ratio')
    axs[1].set_ylabel('RMSE')
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(xticks)
    axs[1].grid(True)
    best_idx = int(np.argmin(rmses))
    axs[1].annotate(f"best: {xticks[best_idx]}\n{rmses[best_idx]:.3f}",
                    xy=(x[best_idx], rmses[best_idx]), xytext=(0, -30),
                    textcoords='offset points', ha='center', arrowprops=dict(arrowstyle='->'))

    axs[2].plot(x, r2s, marker='o', linestyle='-', color='tab:green')
    axs[2].set_title('R^2 vs Synthetic Ratio')
    axs[2].set_xlabel('Synthetic ratio')
    axs[2].set_ylabel('R^2')
    axs[2].set_xticks(x)
    axs[2].set_xticklabels(xticks)
    axs[2].grid(True)
    best_idx = int(np.argmax(r2s))
    axs[2].annotate(f"best: {xticks[best_idx]}\n{r2s[best_idx]:.3f}",
                    xy=(x[best_idx], r2s[best_idx]), xytext=(0, 10),
                    textcoords='offset points', ha='center', arrowprops=dict(arrowstyle='->'))

    axs[3].plot(x, rs, marker='o', linestyle='-', color='tab:red')
    axs[3].set_title('Pearson r vs Synthetic Ratio')
    axs[3].set_xlabel('Synthetic ratio')
    axs[3].set_ylabel('Pearson r')
    axs[3].set_xticks(x)
    axs[3].set_xticklabels(xticks)
    axs[3].grid(True)
    best_idx = int(np.argmax(rs))
    axs[3].annotate(f"best: {xticks[best_idx]}\n{rs[best_idx]:.3f}",
                    xy=(x[best_idx], rs[best_idx]), xytext=(0, 10),
                    textcoords='offset points', ha='center', arrowprops=dict(arrowstyle='->'))

    plt.tight_layout()
    plt.show()

    # Optionally save the figure
    timestamp = datetime.now().strftime("%Y%m%d_%H_%M")
    filename = f"../../plots/{model_name}_performance_vs_synth_ratio_{timestamp}.png"
    fig.savefig(filename, dpi=200)
            

    # Visualize results
    # visualize_regression_results(
    #     X, y, X_synth, y_synth,
    #     feature_names=feature_cols
    # )
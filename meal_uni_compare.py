import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, wilcoxon
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
import warnings

# Import functions from pred_iauc.py
sys.path.append('.')
from pred_iauc import get_data_all_sub, perform_grid_search

def create_specialized_models_with_known_params():
    """Create specialized models using the best hyperparameters found from per-meal training"""
    
    # Best hyperparameters from per-meal training results
    best_params = {
        'Breakfast': {
            'CatBoost': {
                'subsample': 1.0, 'rsm': 1.0, 'random_strength': 0, 'learning_rate': 0.1,
                'l2_leaf_reg': 7, 'iterations': 200, 'depth': 8, 'border_count': 32, 'bagging_temperature': 0
            }
        },
        'Lunch': {
            'CatBoost': {
                'subsample': 1.0, 'rsm': 0.6, 'random_strength': 1, 'learning_rate': 0.01,
                'l2_leaf_reg': 1, 'iterations': 1000, 'depth': 4, 'border_count': 32, 'bagging_temperature': 0
            }
        },
        'Dinner': {
            'XGBoost': {
                'verbosity': 0, 'subsample': 0.8, 'reg_lambda': 5, 'reg_alpha': 1, 'n_estimators': 100,
                'min_child_weight': 1, 'max_depth': 3, 'learning_rate': 0.05, 'gamma': 1,
                'colsample_bytree': 1.0, 'colsample_bylevel': 1.0
            }
        }
    }
    
    # Best models for each meal type based on CV results
    best_models_per_meal = {
        'Breakfast': 'CatBoost',
        'Lunch': 'CatBoost', 
        'Dinner': 'XGBoost'
    }
    
    specialized_models = {}
    
    for meal_name in ['Breakfast', 'Lunch', 'Dinner']:
        best_model_name = best_models_per_meal[meal_name]
        params = best_params[meal_name][best_model_name]
        
        if best_model_name == 'CatBoost':
            specialized_models[meal_name] = CatBoostRegressor(
                random_state=42,
                logging_level="Silent",
                allow_writing_files=False,
                **params
            )
        elif best_model_name == 'XGBoost':
            specialized_models[meal_name] = XGBRegressor(
                random_state=42,
                **params
            )
    
    return specialized_models, best_models_per_meal

def compare_models_statistical(unified_pred, specialized_pred, y_true):
    """Perform statistical comparison between unified and specialized models"""
    unified_errors = np.abs(y_true - unified_pred)
    specialized_errors = np.abs(y_true - specialized_pred)
    
    # Paired Wilcoxon signed-rank test
    try:
        stat, p_value = wilcoxon(unified_errors, specialized_errors, alternative='two-sided')
        
        # Effect size (mean difference in MAE)
        mae_diff = unified_errors.mean() - specialized_errors.mean()
        
        return {
            'wilcoxon_stat': stat,
            'wilcoxon_p': p_value,
            'mae_difference': mae_diff,
            'unified_mae': unified_errors.mean(),
            'specialized_mae': specialized_errors.mean()
        }
    except:
        return {
            'wilcoxon_stat': np.nan,
            'wilcoxon_p': np.nan,
            'mae_difference': np.nan,
            'unified_mae': unified_errors.mean(),
            'specialized_mae': specialized_errors.mean()
        }

def create_comparison_plots(results, meal_names):
    """Create comprehensive comparison plots"""
    
    # 1. MAE Comparison Bar Plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # MAE comparison
    ax1 = axes[0, 0]
    x_pos = np.arange(len(meal_names))
    unified_maes = [results[meal]['unified']['test_mae'] for meal in meal_names]
    specialized_maes = [results[meal]['specialized']['test_mae'] for meal in meal_names]
    
    width = 0.35
    ax1.bar(x_pos - width/2, unified_maes, width, label='Unified Model', alpha=0.8, color='skyblue')
    ax1.bar(x_pos + width/2, specialized_maes, width, label='Specialized Model', alpha=0.8, color='lightcoral')
    
    ax1.set_xlabel('Meal Type')
    ax1.set_ylabel('Test MAE')
    ax1.set_title('MAE Comparison: Unified vs Specialized Models')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(meal_names)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (u_mae, s_mae) in enumerate(zip(unified_maes, specialized_maes)):
        ax1.text(i - width/2, u_mae + 10, f'{u_mae:.1f}', ha='center', va='bottom')
        ax1.text(i + width/2, s_mae + 10, f'{s_mae:.1f}', ha='center', va='bottom')
    
    # Pearson correlation comparison
    ax2 = axes[0, 1]
    unified_pearsons = [results[meal]['unified']['test_pearson'] for meal in meal_names]
    specialized_pearsons = [results[meal]['specialized']['test_pearson'] for meal in meal_names]
    
    ax2.bar(x_pos - width/2, unified_pearsons, width, label='Unified Model', alpha=0.8, color='skyblue')
    ax2.bar(x_pos + width/2, specialized_pearsons, width, label='Specialized Model', alpha=0.8, color='lightcoral')
    
    ax2.set_xlabel('Meal Type')
    ax2.set_ylabel('Pearson Correlation')
    ax2.set_title('Pearson Correlation: Unified vs Specialized Models')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(meal_names)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (u_r, s_r) in enumerate(zip(unified_pearsons, specialized_pearsons)):
        ax2.text(i - width/2, u_r + 0.01, f'{u_r:.3f}', ha='center', va='bottom')
        ax2.text(i + width/2, s_r + 0.01, f'{s_r:.3f}', ha='center', va='bottom')
    
    # R² comparison
    ax3 = axes[1, 0]
    unified_r2s = [results[meal]['unified']['test_r2'] for meal in meal_names]
    specialized_r2s = [results[meal]['specialized']['test_r2'] for meal in meal_names]
    
    ax3.bar(x_pos - width/2, unified_r2s, width, label='Unified Model', alpha=0.8, color='skyblue')
    ax3.bar(x_pos + width/2, specialized_r2s, width, label='Specialized Model', alpha=0.8, color='lightcoral')
    
    ax3.set_xlabel('Meal Type')
    ax3.set_ylabel('R² Score')
    ax3.set_title('R² Score: Unified vs Specialized Models')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(meal_names)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (u_r2, s_r2) in enumerate(zip(unified_r2s, specialized_r2s)):
        ax3.text(i - width/2, u_r2 + 0.01, f'{u_r2:.3f}', ha='center', va='bottom')
        ax3.text(i + width/2, s_r2 + 0.01, f'{s_r2:.3f}', ha='center', va='bottom')
    
    # Statistical significance plot
    ax4 = axes[1, 1]
    p_values = [results[meal]['statistical']['wilcoxon_p'] for meal in meal_names]
    colors = ['red' if p < 0.05 else 'gray' for p in p_values]
    
    bars = ax4.bar(meal_names, [-np.log10(p) if not np.isnan(p) else 0 for p in p_values], 
                   color=colors, alpha=0.7)
    ax4.axhline(y=-np.log10(0.05), color='red', linestyle='--', label='p = 0.05')
    ax4.set_xlabel('Meal Type')
    ax4.set_ylabel('-log10(p-value)')
    ax4.set_title('Statistical Significance (Wilcoxon Test)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add p-value labels
    for i, (bar, p) in enumerate(zip(bars, p_values)):
        if not np.isnan(p):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'p={p:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('unified_vs_specialized_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 2. Scatter plots for each meal type
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, meal in enumerate(meal_names):
        ax = axes[i]
        
        unified_pred = results[meal]['unified']['y_pred']
        specialized_pred = results[meal]['specialized']['y_pred']
        y_true = results[meal]['unified']['y_true']
        
        # Plot unified model predictions
        ax.scatter(y_true, unified_pred, alpha=0.6, label='Unified Model', 
                  color='skyblue', s=30)
        
        # Plot specialized model predictions  
        ax.scatter(y_true, specialized_pred, alpha=0.6, label='Specialized Model', 
                  color='lightcoral', s=30)
        
        # Add diagonal line
        min_val = min(y_true.min(), unified_pred.min(), specialized_pred.min())
        max_val = max(y_true.max(), unified_pred.max(), specialized_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        ax.set_xlabel('Ground Truth iAUC')
        ax.set_ylabel('Predicted iAUC')
        ax.set_title(f'{meal} Predictions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add correlation text
        unified_r = results[meal]['unified']['test_pearson']
        specialized_r = results[meal]['specialized']['test_pearson']
        ax.text(0.05, 0.95, f'Unified r = {unified_r:.3f}\nSpecialized r = {specialized_r:.3f}', 
                transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig('unified_vs_specialized_scatter.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    # Change to the correct directory
    os.chdir("cgmacros1.0/CGMacros")
    
    print("Loading dataset...")
    data_all_sub = get_data_all_sub()
    if data_all_sub is None:
        return
    
    # Define features (including meal_type for unified models, excluding for specialized)
    feature_cols_unified = ['Carbs', 'Protein', 'Fat', 'Fiber', 'Baseline_Libre', 'Age', 'Gender', 'BMI', 
                           'A1c', 'HOMA', 'Insulin', 'TG', 'Cholesterol', 'HDL', 'Non HDL', 'LDL', 
                           'VLDL', 'CHO/HDL ratio', 'Fasting BG', 'Meal Type']
    
    feature_cols_specialized = ['Carbs', 'Protein', 'Fat', 'Fiber', 'Baseline_Libre', 'Age', 'Gender', 'BMI', 
                               'A1c', 'HOMA', 'Insulin', 'TG', 'Cholesterol', 'HDL', 'Non HDL', 'LDL', 
                               'VLDL', 'CHO/HDL ratio', 'Fasting BG']
    
    meal_type_names = ["Breakfast", "Lunch", "Dinner"]
    meal_type_values = [1, 2, 3]
    
    print(f"\n{'='*60}")
    print("Step 1: Training unified model using perform_grid_search on complete dataset")
    print(f"{'='*60}")
    
    # Prepare unified data (all meals with Meal Type feature)
    X_unified = data_all_sub[feature_cols_unified]
    y_unified = data_all_sub['iAUC']
    
    print(f"Unified dataset shape: {X_unified.shape}")
    print("Running perform_grid_search on complete dataset...")
    
    # Train unified models using perform_grid_search
    unified_optimized_models, unified_scores = perform_grid_search(X_unified, y_unified, enable_tabpfn=False)
    
    # Find best unified model
    best_unified_model_name = None
    best_unified_score = float('inf')
    
    for name, scores in unified_scores.items():
        if isinstance(scores, dict):
            mean_tests = np.array(scores.get("mean_test_score", []))
            if mean_tests.size > 0:
                best_mean = mean_tests.max()  # max because we want the least negative (best MAE)
                mae_score = -best_mean
        else:
            arr = np.array(scores)
            mae_score = -arr.mean()
        
        if mae_score < best_unified_score:
            best_unified_score = mae_score
            best_unified_model_name = name
    
    print(f"Best unified model: {best_unified_model_name} (MAE={best_unified_score:.4f})")
    best_unified_model = unified_optimized_models[best_unified_model_name]
    
    print(f"\n{'='*60}")
    print("Step 2: Creating specialized models with known optimal parameters")
    print(f"{'='*60}")
    
    # Create specialized models using known optimal parameters
    specialized_models, best_models_per_meal = create_specialized_models_with_known_params()
    
    print(f"\n{'='*60}")
    print("Step 3: Comparing unified vs specialized models on each meal type")
    print(f"{'='*60}")
    
    results = {}
    
    for meal_name, meal_val in zip(meal_type_names, meal_type_values):
        print(f"\n{'='*40}")
        print(f"Processing {meal_name} (meal_type={meal_val})")
        print(f"{'='*40}")
        
        # Filter data for this meal type
        meal_mask = data_all_sub["Meal Type"] == meal_val
        X_meal_specialized = data_all_sub.loc[meal_mask, feature_cols_specialized]
        X_meal_unified = data_all_sub.loc[meal_mask, feature_cols_unified]
        y_meal = data_all_sub.loc[meal_mask, 'iAUC']
        
        print(f"{meal_name} dataset shape: {X_meal_specialized.shape}")
        
        # Get specialized model for this meal
        specialized_model = specialized_models[meal_name]
        specialized_model_name = best_models_per_meal[meal_name]
        
        print(f"Specialized model: {specialized_model_name}")
        print(f"Unified model: {best_unified_model_name}")
        
        # Evaluate unified model (trained on all data, tested on meal-specific data)
        print("Evaluating unified model on this meal type...")
        # Unified model is already trained on complete dataset
        unified_pred = best_unified_model.predict(X_meal_unified)
        
        unified_mae = mean_absolute_error(y_meal, unified_pred)
        unified_rmse = root_mean_squared_error(y_meal, unified_pred)
        unified_r2 = r2_score(y_meal, unified_pred)
        unified_pearson = pearsonr(y_meal, unified_pred)[0]
        
        # Evaluate specialized model (trained and tested on meal-specific data)
        print("Evaluating specialized model on this meal type...")
        specialized_model.fit(X_meal_specialized, y_meal)
        specialized_pred = specialized_model.predict(X_meal_specialized)
        
        specialized_mae = mean_absolute_error(y_meal, specialized_pred)
        specialized_rmse = root_mean_squared_error(y_meal, specialized_pred)
        specialized_r2 = r2_score(y_meal, specialized_pred)
        specialized_pearson = pearsonr(y_meal, specialized_pred)[0]
        
        # Statistical comparison
        statistical_results = compare_models_statistical(unified_pred, specialized_pred, y_meal)
        
        results[meal_name] = {
            'unified': {
                'test_mae': unified_mae,
                'test_rmse': unified_rmse,
                'test_r2': unified_r2,
                'test_pearson': unified_pearson,
                'y_pred': unified_pred,
                'y_true': y_meal.values
            },
            'specialized': {
                'test_mae': specialized_mae,
                'test_rmse': specialized_rmse,
                'test_r2': specialized_r2,
                'test_pearson': specialized_pearson,
                'y_pred': specialized_pred,
                'y_true': y_meal.values
            },
            'statistical': statistical_results,
            'unified_model_name': best_unified_model_name,
            'specialized_model_name': specialized_model_name
        }
        
        # Print results
        print(f"\nUnified Model ({best_unified_model_name}) Results:")
        print(f"  Test MAE: {unified_mae:.4f}")
        print(f"  Test RMSE: {unified_rmse:.4f}")
        print(f"  Test R²: {unified_r2:.4f}")
        print(f"  Test Pearson: {unified_pearson:.4f}")
        
        print(f"\nSpecialized Model ({specialized_model_name}) Results:")
        print(f"  Test MAE: {specialized_mae:.4f}")
        print(f"  Test RMSE: {specialized_rmse:.4f}")
        print(f"  Test R²: {specialized_r2:.4f}")
        print(f"  Test Pearson: {specialized_pearson:.4f}")
        
        print(f"\nStatistical Comparison:")
        print(f"  MAE Difference (Unified - Specialized): {statistical_results['mae_difference']:.4f}")
        print(f"  Wilcoxon p-value: {statistical_results['wilcoxon_p']:.4f}")
        
        if statistical_results['wilcoxon_p'] < 0.05:
            better_model = "Specialized" if statistical_results['mae_difference'] > 0 else "Unified"
            print(f"  → {better_model} model is significantly better (p < 0.05)")
        else:
            print(f"  → No significant difference between models (p ≥ 0.05)")
    
    # Create comprehensive plots
    print(f"\nCreating comparison plots...")
    create_comparison_plots(results, meal_type_names)
    
    # Save detailed results to CSV
    print(f"\nSaving detailed results...")
    summary_data = []
    for meal in meal_type_names:
        summary_data.append({
            'Meal_Type': meal,
            'Model_Type': f'Unified_{results[meal]["unified_model_name"]}',
            'Test_MAE': results[meal]['unified']['test_mae'],
            'Test_RMSE': results[meal]['unified']['test_rmse'],
            'Test_R2': results[meal]['unified']['test_r2'],
            'Test_Pearson': results[meal]['unified']['test_pearson']
        })
        summary_data.append({
            'Meal_Type': meal,
            'Model_Type': f'Specialized_{results[meal]["specialized_model_name"]}',
            'Test_MAE': results[meal]['specialized']['test_mae'],
            'Test_RMSE': results[meal]['specialized']['test_rmse'],
            'Test_R2': results[meal]['specialized']['test_r2'],
            'Test_Pearson': results[meal]['specialized']['test_pearson']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('unified_vs_specialized_results.csv', index=False)
    print("Saved results to unified_vs_specialized_results.csv")
    
    # Statistical comparison summary
    stat_data = []
    for meal in meal_type_names:
        stat_data.append({
            'Meal_Type': meal,
            'Unified_Model': results[meal]['unified_model_name'],
            'Specialized_Model': results[meal]['specialized_model_name'],
            'MAE_Difference': results[meal]['statistical']['mae_difference'],
            'Unified_MAE': results[meal]['statistical']['unified_mae'],
            'Specialized_MAE': results[meal]['statistical']['specialized_mae'],
            'Wilcoxon_P_Value': results[meal]['statistical']['wilcoxon_p'],
            'Significant': results[meal]['statistical']['wilcoxon_p'] < 0.05,
            'Better_Model': 'Specialized' if results[meal]['statistical']['mae_difference'] > 0 else 'Unified'
        })
    
    stat_df = pd.DataFrame(stat_data)
    stat_df.to_csv('statistical_comparison.csv', index=False)
    print("Saved statistical comparison to statistical_comparison.csv")
    
    print(f"\n{'='*60}")
    print("Analysis Complete!")
    print(f"{'='*60}")
    print("Generated files:")
    print("- unified_vs_specialized_comparison.png")
    print("- unified_vs_specialized_scatter.png") 
    print("- unified_vs_specialized_results.csv")
    print("- statistical_comparison.csv")

if __name__ == "__main__":
    main()

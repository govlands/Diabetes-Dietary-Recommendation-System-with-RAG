import torch
import numpy as np
from tabpfn import TabPFNRegressor
import json
import os
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
from utils import get_save_path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CAT_INDICES = [6]

def _torch_load_compat(path, map_location):
    """
    Load a checkpoint in a way compatible with PyTorch 2.6+ safe loading.
    Tries weights_only=False first (may be required for full checkpoints).
    Falls back to temporarily allowlisting numpy.core.multiarray.scalar if needed.
    """
    try:
        # PyTorch >=2.6 supports weights_only arg; try full object load
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # Older torch: no weights_only arg
        return torch.load(path, map_location=map_location)
    except RuntimeError as e:
        # Try allowing the specific numpy scalar global (use only if you trust file)
        try:
            import numpy as _np
            # add_safe_globals may exist per error message; guard with hasattr
            if hasattr(torch.serialization, "add_safe_globals"):
                torch.serialization.add_safe_globals([_np.core.multiarray.scalar])
            return torch.load(path, map_location=map_location)
        except Exception:
            # re-raise original for visibility
            raise e

def load_models(path=None, device=DEVICE, X_sample=None, y_sample=None):
    """Load finetuned and origin TabPFN regressors. If the internal model_ is not
    initialized, a small X_sample/y_sample can be provided to trigger initialization
    via get_preprocessed_datasets before loading the state_dict.
    """
    print("Loading model:", path)
    map_loc = torch.device(device)
    data = _torch_load_compat(path, map_loc)
    reg_config = data.get("regressor_config", {})
    # ensure device set
    reg_config["device"] = device
    # create finetuned_reg with same config; this mirrors how it was instantiated in training
    finetuned_reg = TabPFNRegressor(**reg_config, fit_mode="batched", differentiable_input=False)
    origin_config = dict(reg_config)
    origin_config['n_estimators'] = 1
    origin_reg = TabPFNRegressor(**origin_config, fit_mode="batched", differentiable_input=False)
    # load saved state dict into internal model; ensure model_ exists first
    model_state = data.get("model_state_dict")
    if model_state is not None:
        # If model_ not initialized, try to initialize using a tiny sample
        if not hasattr(finetuned_reg, "model_"):
            if X_sample is not None and y_sample is not None:
                # Ensure at least 2 samples (TabPFN requires minimum 2)
                Xs = np.asarray(X_sample)
                ys = np.asarray(y_sample)
                if Xs.ndim == 1:
                    Xs = Xs.reshape(1, -1)
                if ys.ndim == 0:
                    ys = ys.reshape(1,)
                if Xs.shape[0] < 2:
                    # duplicate the first row to make 2 samples
                    Xs = np.vstack([Xs, Xs[0:1]])
                    ys = np.concatenate([ys, ys[0:1]])

                # robust splitter that works for small arrays
                def _small_splitter(X, y, test_size=0.5, random_state=None):
                    X = np.asarray(X)
                    y = np.asarray(y)
                    n = len(X)
                    if n < 2:
                        return X, X, y, y
                    split = max(1, int(n * (1 - test_size)))
                    return X[:split], X[split:], y[:split], y[split:]

                try:
                    _ = finetuned_reg.get_preprocessed_datasets(
                        Xs, ys, _small_splitter, max_data_size=max(1, Xs.shape[0] - 1)
                    )
                except Exception as e:
                    print(f"Warning: model_ initialization via get_preprocessed_datasets failed: {e}")
        # attempt to load state dict if model_ now exists
        if hasattr(finetuned_reg, "model_"):
            finetuned_reg.model_.load_state_dict(model_state)
        else:
            raise AttributeError("Failed to initialize internal model_ for TabPFNRegressor; cannot load state_dict.")

    finetuned_reg.model_.to(map_loc)
    finetuned_reg.model_.eval()
    return finetuned_reg, origin_reg

def load_datasets(path=None, device=DEVICE):
    """
    Returns: X_train, X_test, y_train, y_test
    Expects the checkpoint to contain a 'datasets' entry (tuple/list).
    """
    print("Loading datasets from:", path)
    map_loc = torch.device(device)
    data = _torch_load_compat(path, map_loc)
    datasets = data.get('datasets')
    if datasets is None:
        raise RuntimeError(f"No 'datasets' key found in checkpoint: {path}")
    # Expect datasets to be (X_train, X_test, y_train, y_test)
    return tuple(datasets)
    
def get_baseline_regs(X=None, y=None):
    
    param_grids = {
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
    }
    
    base_models = {
        'XGBoost': XGBRegressor(random_state=42),
        'CatBoost': CatBoostRegressor(
            random_state=42,
            logging_level="Silent",
            allow_writing_files=False
        )
    }
    
    tuned_models = {}
    
    for model_name, model in base_models.items():
        print(f"Tuning {model_name}...")
        param_grid = param_grids[model_name]
        
        # RandomizedSearchCV for hyperparameter tuning
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=50,  # Number of parameter settings that are sampled
            cv=5,       # 5-fold cross validation
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        
        # Fit the random search
        random_search.fit(X, y)
        
        # Get the best model
        best_model = random_search.best_estimator_
        tuned_models[model_name] = best_model
        
        print(f"Best parameters for {model_name}: {random_search.best_params_}")
        print(f"Best CV score for {model_name}: {-random_search.best_score_:.4f}")
    
    return tuned_models
    
def compare_models(X_test, y_test, model_dict):
    # Create plots directory if it doesn't exist
    plots_dir = 'plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Make predictions and calculate metrics
    results = {}
    predictions = {}
    errors = {}

    for model_name, model in model_dict.items():
        print(f"Evaluating {model_name}...")

        y_pred = model.predict(X_test)
        
        predictions[model_name] = y_pred
        errors[model_name] = y_test - y_pred
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        r, _ = pearsonr(y_test, y_pred)
        
        results[model_name] = {
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'R': r
        }
        
        print(f"{model_name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}, R: {r:.4f}")

    # 1. Metrics comparison bar plots (x-axis shows model names) - Enhanced for visual differences
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    metrics = ['RMSE', 'MAE', 'R²', 'R']
    model_names = list(model_dict.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))

    for i, metric in enumerate(metrics):
        ax = axes[i // 2, i % 2]
        values = [results[model][metric] for model in model_names]
        x = np.arange(len(model_names))
        bars = ax.bar(x, values, color=colors, edgecolor='black', linewidth=1.2)
        ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel(metric, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        
        # Enhanced y-axis range to amplify visual differences
        if len(values) > 1:
            val_min, val_max = min(values), max(values)
            val_range = val_max - val_min
            if metric in ['RMSE', 'MAE']:
                # For error metrics, zoom in around the values
                margin = max(val_range * 0.3, val_range * 0.1) if val_range > 0 else val_max * 0.1
                ax.set_ylim(max(0, val_min - margin), val_max + margin)
            elif metric in ['R²', 'R']:
                # For correlation metrics, focus on the high-performance range
                if val_min > 0.5:  # If all models perform reasonably well
                    margin = max(val_range * 0.5, 0.05)
                    ax.set_ylim(val_min - margin, min(1.0, val_max + margin))
        
        # Add value labels on bars with better positioning
        ymax = ax.get_ylim()[1]
        ymin = ax.get_ylim()[0]
        for bar, value in zip(bars, values):
            # Position label slightly above bar, but within plot area
            label_y = min(bar.get_height() + (ymax - ymin) * 0.02, ymax * 0.98)
            ax.text(bar.get_x() + bar.get_width() / 2, label_y,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = get_save_path('plots', 'metrics_bar', 'png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.show()

    # 2. Residuals plots - Enhanced with zoom and better scaling
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for i, (model_name, y_pred) in enumerate(predictions.items()):
        ax = axes[i]
        residuals = y_test - y_pred
        
        # Use different marker styles and transparency for better distinction
        scatter = ax.scatter(y_pred, residuals, alpha=0.7, color=colors[i], 
                           s=30, edgecolors='black', linewidth=0.5)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Predicted Values', fontsize=12)
        ax.set_ylabel('Residuals', fontsize=12)
        ax.set_title(f'{model_name} - Residuals Plot', fontsize=14, fontweight='bold')
        
        # Enhanced axis limits to better show residual patterns
        y_std = np.std(residuals)
        y_mean = np.mean(residuals)
        # Focus on ±2.5 standard deviations to highlight patterns
        ax.set_ylim(y_mean - 2.5 * y_std, y_mean + 2.5 * y_std)
        
        # Add statistics text box
        rmse_val = np.sqrt(np.mean(residuals**2))
        ax.text(0.05, 0.95, f'RMSE: {rmse_val:.3f}\nMean: {y_mean:.3f}\nStd: {y_std:.3f}', 
                transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=10, verticalalignment='top')
        
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = get_save_path('plots', 'residuals_plot', 'png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.show()

    # 3. True vs Predicted plots - Enhanced with focused view and confidence intervals
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for i, (model_name, y_pred) in enumerate(predictions.items()):
        ax = axes[i]
        scatter = ax.scatter(y_test, y_pred, alpha=0.7, color=colors[i], 
                           s=30, edgecolors='black', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min(min(y_test), min(y_pred))
        max_val = max(max(y_test), max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r-', lw=2, label='Perfect Prediction')
        
        # Add confidence bands around perfect prediction line
        range_vals = np.array([min_val, max_val])
        ax.fill_between(range_vals, range_vals - np.std(y_test - y_pred), 
                       range_vals + np.std(y_test - y_pred), alpha=0.2, color='red', 
                       label='±1 Std Error')
        
        ax.set_xlabel('True Values', fontsize=12)
        ax.set_ylabel('Predicted Values', fontsize=12)
        ax.set_title(f'{model_name} - True vs Predicted', fontsize=14, fontweight='bold')
        
        # Enhanced axis limits - focus on data range with some padding
        data_min = min(min_val, min(y_pred))
        data_max = max(max_val, max(y_pred))
        padding = (data_max - data_min) * 0.05
        ax.set_xlim(data_min - padding, data_max + padding)
        ax.set_ylim(data_min - padding, data_max + padding)
        
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='upper left')
        
        # Enhanced R² display with additional metrics
        r2 = results[model_name]['R²']
        rmse = results[model_name]['RMSE']
        mae = results[model_name]['MAE']
        ax.text(0.05, 0.95, f'R² = {r2:.4f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}', 
                transform=ax.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                fontsize=10, verticalalignment='top', fontweight='bold')

    plt.tight_layout()
    path = get_save_path('plots', 'true_vs_pred', 'png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.show()

    # 4. Error distribution histograms - Enhanced with overlays and statistical comparisons
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    # Calculate global error range for consistent scaling
    all_errors = np.concatenate(list(errors.values()))
    global_min, global_max = np.percentile(all_errors, [1, 99])  # Use 1st-99th percentile to avoid outliers
    
    for i, (model_name, error) in enumerate(errors.items()):
        ax = axes[i]
        
        # Enhanced histogram with better binning
        n_bins = 40
        counts, bins, patches = ax.hist(error, bins=n_bins, alpha=0.7, color=colors[i], 
                                      edgecolor='black', linewidth=0.8, density=True)
        
        # Add normal distribution overlay for comparison
        mu, sigma = np.mean(error), np.std(error)
        x = np.linspace(global_min, global_max, 100)
        normal_dist = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        ax.plot(x, normal_dist, 'r-', linewidth=2, label='Normal Distribution')
        
        # Zero error line
        ax.axvline(x=0, color='green', linestyle='--', linewidth=2, label='Zero Error')
        ax.axvline(x=mu, color='orange', linestyle='-', linewidth=2, label=f'Mean Error')
        
        ax.set_xlabel('Prediction Error', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'{model_name} - Error Distribution', fontsize=14, fontweight='bold')
        
        # Consistent x-axis range for all subplots to enable comparison
        ax.set_xlim(global_min, global_max)
        
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        # Enhanced statistics with additional metrics
        mean_error = np.mean(error)
        std_error = np.std(error)
        median_error = np.median(error)
        mae_error = np.mean(np.abs(error))
        
        stats_text = f'Mean: {mean_error:.3f}\nStd: {std_error:.3f}\nMedian: {median_error:.3f}\nMAE: {mae_error:.3f}'
    plt.tight_layout()
    path = get_save_path('plots', 'error_distribution', 'png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.show()

    # 5. NEW: Relative Performance Comparison - Radar Chart and Difference Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Radar chart for multi-metric comparison
    metrics_radar = ['RMSE', 'MAE', 'R²', 'R']
    num_vars = len(metrics_radar)
    
    # Compute the angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # complete the circle
    
    ax1 = plt.subplot(121, polar=True)
    ax1.set_theta_offset(np.pi / 2)
    ax1.set_theta_direction(-1)
    
    # Normalize metrics for radar chart (0-1 scale, higher is better)
    normalized_results = {}
    for model_name in model_names:
        norm_values = []
        for metric in metrics_radar:
            val = results[model_name][metric]
            if metric in ['RMSE', 'MAE']:
                # For error metrics, invert so lower error = higher score
                all_vals = [results[m][metric] for m in model_names]
                max_val = max(all_vals)
                min_val = min(all_vals)
                if max_val != min_val:
                    norm_val = 1 - (val - min_val) / (max_val - min_val)
                else:
                    norm_val = 1.0
            else:  # R², R
                norm_val = val
            norm_values.append(norm_val)
        norm_values += norm_values[:1]  # complete the circle
        normalized_results[model_name] = norm_values
    
    # Plot each model
    for i, (model_name, norm_values) in enumerate(normalized_results.items()):
        ax1.plot(angles, norm_values, 'o-', linewidth=2, label=model_name, color=colors[i])
        ax1.fill(angles, norm_values, alpha=0.1, color=colors[i])
    
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(metrics_radar)
    ax1.set_ylim(0, 1)
    ax1.set_title('Multi-Metric Performance Comparison\n(Normalized, Higher=Better)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    ax1.grid(True)
    
    # Difference plot showing relative improvement over baseline
    ax2 = plt.subplot(122)
    if len(model_names) > 1:
        baseline_model = model_names[-1]  # Use last model as baseline (often worst performing)
        baseline_rmse = results[baseline_model]['RMSE']
        
        improvements = {}
        for model_name in model_names[:-1]:  # Exclude baseline from comparison
            rmse_improvement = ((baseline_rmse - results[model_name]['RMSE']) / baseline_rmse) * 100
            improvements[model_name] = rmse_improvement
        
        model_names_subset = list(improvements.keys())
        improvement_values = list(improvements.values())
        
        bars = ax2.bar(range(len(improvement_values)), improvement_values, 
                      color=colors[:len(improvement_values)], edgecolor='black', linewidth=1.2)
        ax2.set_xlabel('Models', fontsize=12)
        ax2.set_ylabel('RMSE Improvement vs Baseline (%)', fontsize=12)
        ax2.set_title(f'Relative RMSE Improvement\n(Baseline: {baseline_model})', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(improvement_values)))
        ax2.set_xticklabels(model_names_subset, rotation=45, ha='right')
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=1)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(bars, improvement_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (max(improvement_values) * 0.01),
                    f'{value:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    path = get_save_path('plots', 'performance_comparison', 'png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.show()

    # Print summary table
    print("\n" + "="*60)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*60)
    print(f"{'Model':<20} {'RMSE':<10} {'MAE':<10} {'R²':<10} {'R':<10}")
    print("-"*60)
    for model_name, metrics in results.items():
        print(f"{model_name:<20} {metrics['RMSE']:<10.4f} {metrics['MAE']:<10.4f} "
              f"{metrics['R²']:<10.4f} {metrics['R']:<10.4f}")
    
def main():
    model_path = "models/tabpfn_model_state_20250827_182232.pt"
    X_train, X_test, y_train, y_test = load_datasets(path=model_path)
    tuned_tabpfn, origin_tabpfn = load_models(path=model_path, X_sample=X_train, y_sample=y_train)
    tuned_tabpfn.fit(X_train, y_train)
    origin_tabpfn.fit(X_train, y_train)
    tuned_baseline_models = get_baseline_regs(X_train, y_train)
    xgb_reg = tuned_baseline_models['XGBoost']
    cat_reg = tuned_baseline_models['CatBoost']
    model_dict = {
        'finetuned tabpfn': tuned_tabpfn,
        'origin tabpfn': origin_tabpfn,
        'XGBoost': xgb_reg,
        'CatBoost': cat_reg,
    }
    compare_models(X_test, y_test, model_dict)
    
if __name__ == '__main__':
    main()
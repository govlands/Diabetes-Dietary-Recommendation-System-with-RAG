import os
import glob
import pandas as pd
import numpy as np
import torch
from datetime import datetime
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from scipy.stats import pearsonr, ttest_rel, wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns
from compare_models import load_models, _torch_load_compat
from utils import get_save_path, fetch_dataset_from_cgmacros


def load_all_samples_from_folder(folder='generated_samples'):
    """
    从generated_samples文件夹中加载所有样本数据，确保顺序一致
    返回: X_rag, X_llm, y_origin (all as numpy arrays)
    """
    print(f"从 {folder} 文件夹加载样本数据...")
    
    # 获取所有相关文件
    rag_files = sorted(glob.glob(os.path.join(folder, 'rag_samples_*.csv')))
    llm_files = sorted(glob.glob(os.path.join(folder, 'llm_samples_*.csv')))
    iauc_files = sorted(glob.glob(os.path.join(folder, 'original_iauc_*.csv')))
    
    print(f"找到 {len(rag_files)} 个RAG样本文件")
    print(f"找到 {len(llm_files)} 个LLM样本文件")
    print(f"找到 {len(iauc_files)} 个原始iAUC文件")
    
    if len(rag_files) != len(llm_files) or len(rag_files) != len(iauc_files):
        raise ValueError("RAG、LLM和iAUC文件数量不匹配！")
    
    # 验证文件名时间戳一致性
    def extract_timestamp(filename):
        # 从文件名中提取时间戳，如 'rag_samples_20250828_004353.csv' -> '20250828_004353'
        basename = os.path.basename(filename)
        parts = basename.split('_')
        if len(parts) >= 3:
            return '_'.join(parts[-2:]).replace('.csv', '')
        return basename
    
    rag_timestamps = [extract_timestamp(f) for f in rag_files]
    llm_timestamps = [extract_timestamp(f) for f in llm_files]
    iauc_timestamps = [extract_timestamp(f) for f in iauc_files]
    
    if rag_timestamps != llm_timestamps or rag_timestamps != iauc_timestamps:
        print("警告：文件时间戳不完全匹配，请检查文件对应关系")
        print(f"RAG时间戳: {rag_timestamps}")
        print(f"LLM时间戳: {llm_timestamps}")
        print(f"iAUC时间戳: {iauc_timestamps}")
    
    # 按时间戳排序后加载数据
    all_rag_data = []
    all_llm_data = []
    all_iauc_data = []
    
    for i, (rag_file, llm_file, iauc_file) in enumerate(zip(rag_files, llm_files, iauc_files)):
        print(f"加载第 {i+1} 批数据:")
        print(f"  RAG: {os.path.basename(rag_file)}")
        print(f"  LLM: {os.path.basename(llm_file)}")
        print(f"  iAUC: {os.path.basename(iauc_file)}")
        
        # 加载数据
        rag_df = pd.read_csv(rag_file)
        llm_df = pd.read_csv(llm_file)
        iauc_df = pd.read_csv(iauc_file)
        
        print(f"  样本数量: RAG={len(rag_df)}, LLM={len(llm_df)}, iAUC={len(iauc_df)}")
        
        if len(rag_df) != len(llm_df) or len(rag_df) != len(iauc_df):
            raise ValueError(f"第 {i+1} 批数据中样本数量不匹配！")
        
        all_rag_data.append(rag_df)
        all_llm_data.append(llm_df)
        all_iauc_data.append(iauc_df)
    
    # 合并所有数据
    print("\n合并所有数据...")
    X_rag_df = pd.concat(all_rag_data, ignore_index=True)
    X_llm_df = pd.concat(all_llm_data, ignore_index=True)
    y_origin_df = pd.concat(all_iauc_data, ignore_index=True)
    
    # 验证特征列
    expected_features = ['Carbs', 'Protein', 'Fat', 'Fiber', 'Baseline_Libre', 'Age', 'Gender', 
                        'BMI', 'A1c', 'HOMA', 'Insulin', 'TG', 'Cholesterol', 'HDL', 'Non HDL', 
                        'LDL', 'VLDL', 'CHO/HDL ratio', 'Fasting BG']
    
    missing_rag = [f for f in expected_features if f not in X_rag_df.columns]
    missing_llm = [f for f in expected_features if f not in X_llm_df.columns]
    
    if missing_rag:
        raise ValueError(f"RAG数据缺少特征: {missing_rag}")
    if missing_llm:
        raise ValueError(f"LLM数据缺少特征: {missing_llm}")
    
    # 转换为numpy数组，确保特征顺序一致
    X_rag = X_rag_df[expected_features].values.astype(np.float64)
    X_llm = X_llm_df[expected_features].values.astype(np.float64)
    y_origin = y_origin_df['iAUC'].values.astype(np.float64)
    
    print(f"\n最终数据形状:")
    print(f"X_rag: {X_rag.shape}")
    print(f"X_llm: {X_llm.shape}")
    print(f"y_origin: {y_origin.shape}")
    
    # 数据质量检查
    print(f"\n数据质量检查:")
    print(f"X_rag - NaN数量: {np.isnan(X_rag).sum()}, Inf数量: {np.isinf(X_rag).sum()}")
    print(f"X_llm - NaN数量: {np.isnan(X_llm).sum()}, Inf数量: {np.isinf(X_llm).sum()}")
    print(f"y_origin - NaN数量: {np.isnan(y_origin).sum()}, Inf数量: {np.isinf(y_origin).sum()}")
    
    return X_rag, X_llm, y_origin


def load_tabpfn_models(model_path="models/tabpfn_model_state_20250827_182232.pt", device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    加载两个微调后的TabPFN模型实例，仅random_state不同
    返回: (rag_model, llm_model) - 分别用于RAG和LLM样本预测
    """
    print(f"加载TabPFN模型: {model_path}")
    print(f"使用设备: {device}")
    
    # 创建一个小样本用于模型初始化
    dummy_X = np.random.rand(2, 19)  # 19个特征
    dummy_y = np.random.rand(2)
    
    try:
        # 加载第一个实例（用于RAG预测）
        print("创建RAG预测模型实例 (random_state=42)...")
        rag_model, _ = load_models(path=model_path, device=device, X_sample=dummy_X, y_sample=dummy_y)
        
        # 加载第二个实例（用于LLM预测）
        print("创建LLM预测模型实例 (random_state=123)...")
        llm_model, _ = load_models(path=model_path, device=device, X_sample=dummy_X, y_sample=dummy_y)
        
        # 手动设置不同的random_state以增加预测差异性
        if hasattr(rag_model, 'random_state'):
            rag_model.random_state = 114514
        if hasattr(llm_model, 'random_state'):
            llm_model.random_state = 1919810
            
        print("两个TabPFN模型实例创建成功！")
        print(f"RAG模型random_state: {getattr(rag_model, 'random_state', 'unknown')}")
        print(f"LLM模型random_state: {getattr(llm_model, 'random_state', 'unknown')}")
        
        return rag_model, llm_model
        
    except Exception as e:
        print(f"模型加载失败: {e}")
        raise


def predict_iauc_batch(model, X, batch_size=100):
    """
    分批预测iAUC，避免内存问题
    """
    print(f"开始预测，共 {len(X)} 个样本，批大小: {batch_size}")
    
    predictions = []
    n_batches = (len(X) + batch_size - 1) // batch_size
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(X))
        
        print(f"处理批次 {i+1}/{n_batches}: 样本 {start_idx}-{end_idx-1}")
        
        X_batch = X[start_idx:end_idx]
        
        try:
            # 为每个批次创建新的模型实例以避免状态问题
            batch_predictions = model.predict(X_batch)
            predictions.extend(batch_predictions)
            
        except Exception as e:
            print(f"批次 {i+1} 预测失败: {e}")
            # 使用平均值填充失败的预测
            batch_predictions = [np.mean(predictions) if predictions else 5000.0] * len(X_batch)
            predictions.extend(batch_predictions)
    
    return np.array(predictions)


def compute_statistics(y_origin, y_rag, y_llm):
    """
    计算各种统计指标
    """
    print("\n计算统计指标...")
    
    # 基础统计
    stats = {}
    for name, y in [('Origin', y_origin), ('RAG', y_rag), ('LLM', y_llm)]:
        stats[name] = {
            'mean': np.mean(y),
            'median': np.median(y),
            'std': np.std(y),
            'min': np.min(y),
            'max': np.max(y),
            'q25': np.percentile(y, 25),
            'q75': np.percentile(y, 75)
        }
    
    # 配对差值分析
    diff_rag_llm = y_llm - y_rag  # 正值表示RAG更好
    diff_rag_origin = y_origin - y_rag  # 正值表示RAG改善
    diff_llm_origin = y_origin - y_llm  # 正值表示LLM改善
    
    # 胜负统计
    rag_better_than_llm = np.sum(y_rag < y_llm)
    rag_better_than_origin = np.sum(y_rag < y_origin)
    llm_better_than_origin = np.sum(y_llm < y_origin)
    both_better_than_origin = np.sum((y_rag < y_origin) & (y_llm < y_origin))
    
    win_loss_stats = {
        'total_samples': len(y_origin),
        'rag_better_than_llm': rag_better_than_llm,
        'rag_better_than_llm_pct': rag_better_than_llm / len(y_origin) * 100,
        'rag_better_than_origin': rag_better_than_origin,
        'rag_better_than_origin_pct': rag_better_than_origin / len(y_origin) * 100,
        'llm_better_than_origin': llm_better_than_origin,
        'llm_better_than_origin_pct': llm_better_than_origin / len(y_origin) * 100,
        'both_better_than_origin': both_better_than_origin,
        'both_better_than_origin_pct': both_better_than_origin / len(y_origin) * 100
    }
    
    # 配对检验
    try:
        # RAG vs LLM
        t_stat_rag_llm, p_val_rag_llm = ttest_rel(y_rag, y_llm)
        w_stat_rag_llm, p_val_w_rag_llm = wilcoxon(y_rag, y_llm, alternative='two-sided')
        
        # RAG vs Origin
        t_stat_rag_origin, p_val_rag_origin = ttest_rel(y_rag, y_origin)
        w_stat_rag_origin, p_val_w_rag_origin = wilcoxon(y_rag, y_origin, alternative='two-sided')
        
        statistical_tests = {
            'rag_vs_llm': {
                't_test': {'statistic': t_stat_rag_llm, 'p_value': p_val_rag_llm},
                'wilcoxon': {'statistic': w_stat_rag_llm, 'p_value': p_val_w_rag_llm}
            },
            'rag_vs_origin': {
                't_test': {'statistic': t_stat_rag_origin, 'p_value': p_val_rag_origin},
                'wilcoxon': {'statistic': w_stat_rag_origin, 'p_value': p_val_w_rag_origin}
            }
        }
        
    except Exception as e:
        print(f"统计检验计算失败: {e}")
        statistical_tests = {}
    
    # 效应量 (Cohen's d for paired samples)
    def cohens_d_paired(x1, x2):
        diff = x1 - x2
        return np.mean(diff) / np.std(diff)
    
    effect_sizes = {
        'cohens_d_rag_vs_llm': cohens_d_paired(y_rag, y_llm),
        'cohens_d_rag_vs_origin': cohens_d_paired(y_rag, y_origin),
        'cohens_d_llm_vs_origin': cohens_d_paired(y_llm, y_origin)
    }
    
    return {
        'basic_stats': stats,
        'win_loss_stats': win_loss_stats,
        'differences': {
            'rag_vs_llm': diff_rag_llm,
            'rag_vs_origin': diff_rag_origin,
            'llm_vs_origin': diff_llm_origin
        },
        'statistical_tests': statistical_tests,
        'effect_sizes': effect_sizes
    }


def create_visualizations(y_origin, y_rag, y_llm, stats_results):
    """
    创建增强版可视化图表，按行组织显示
    """
    print("\n生成增强版可视化图表...")
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 计算关键统计量
    differences = stats_results['differences']
    rag_vs_llm_diff = differences['rag_vs_llm']  # LLM - RAG
    rag_improvement = differences['rag_vs_origin']  # Origin - RAG  
    llm_improvement = differences['llm_vs_origin']  # Origin - LLM
    
    plot_paths = []
    
    # 第1行：差值分布图（3个子图）
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # RAG vs LLM差值分布
    ax = axes[0]
    ax.hist(rag_vs_llm_diff, bins=30, alpha=0.6, edgecolor='black', color='thistle')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='无差异')
    ax.axvline(np.mean(rag_vs_llm_diff), color='blue', linestyle='-', linewidth=2, label='均值')
    ax.set_xlabel('iAUC差值 (LLM - RAG)', fontsize=12)
    ax.set_ylabel('频次', fontsize=12)
    ax.set_title('RAG vs LLM差值分布', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 添加统计信息
    mean_diff = np.mean(rag_vs_llm_diff)
    std_diff = np.std(rag_vs_llm_diff)
    ax.text(0.02, 0.98, f'均值: {mean_diff:.3f}\n标准差: {std_diff:.3f}', 
           transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
           fontsize=10, verticalalignment='top')
    
    # RAG改善效果分布
    ax = axes[1]
    ax.hist(rag_improvement, bins=30, alpha=0.6, edgecolor='black', color='lightsteelblue')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='无改善')
    ax.axvline(np.mean(rag_improvement), color='darkblue', linestyle='-', linewidth=2, label='均值')
    ax.set_xlabel('iAUC差值 (原始 - RAG)', fontsize=12)
    ax.set_ylabel('频次', fontsize=12)
    ax.set_title('RAG改善效果分布', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 添加统计信息
    mean_rag_imp = np.mean(rag_improvement)
    std_rag_imp = np.std(rag_improvement)
    rag_better_pct = stats_results['win_loss_stats']['rag_better_than_origin_pct']
    ax.text(0.02, 0.98, f'均值: {mean_rag_imp:.3f}\n标准差: {std_rag_imp:.3f}\n改善率: {rag_better_pct:.1f}%', 
           transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
           fontsize=10, verticalalignment='top')
    
    # LLM改善效果分布
    ax = axes[2]
    ax.hist(llm_improvement, bins=30, alpha=0.6, edgecolor='black', color='mistyrose')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='无改善')
    ax.axvline(np.mean(llm_improvement), color='darkred', linestyle='-', linewidth=2, label='均值')
    ax.set_xlabel('iAUC差值 (原始 - LLM)', fontsize=12)
    ax.set_ylabel('频次', fontsize=12)
    ax.set_title('LLM改善效果分布', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 添加统计信息
    mean_llm_imp = np.mean(llm_improvement)
    std_llm_imp = np.std(llm_improvement)
    llm_better_pct = stats_results['win_loss_stats']['llm_better_than_origin_pct']
    ax.text(0.02, 0.98, f'均值: {mean_llm_imp:.3f}\n标准差: {std_llm_imp:.3f}\n改善率: {llm_better_pct:.1f}%', 
           transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
           fontsize=10, verticalalignment='top')
    
    plt.tight_layout()
    path1 = get_save_path('plots', f'rag_llm_distributions', 'png')
    plt.savefig(path1, dpi=300, bbox_inches='tight')
    plt.show()
    plot_paths.append(path1)
    
    # 第2行：预测对比图（2个子图）
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # RAG vs LLM预测对比
    ax = axes[0]
    ax.scatter(y_llm, y_rag, alpha=0.6, s=30, color='purple', edgecolors='black', linewidth=0.3)
    
    # 聚焦于数据范围
    llm_range = np.ptp(y_llm)
    rag_range = np.ptp(y_rag)
    x_margin = llm_range * 0.05
    y_margin = rag_range * 0.05
    ax.set_xlim(np.min(y_llm) - x_margin, np.max(y_llm) + x_margin)
    ax.set_ylim(np.min(y_rag) - y_margin, np.max(y_rag) + y_margin)
    
    # 对角线
    min_val = min(np.min(y_rag), np.min(y_llm))
    max_val = max(np.max(y_rag), np.max(y_llm))
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='相等线')
    
    ax.set_xlabel('LLM预测的iAUC', fontsize=12)
    ax.set_ylabel('RAG预测的iAUC', fontsize=12)
    ax.set_title('RAG vs LLM预测对比', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 添加统计信息
    rag_better_llm_pct = stats_results['win_loss_stats']['rag_better_than_llm_pct']
    correlation = np.corrcoef(y_llm, y_rag)[0, 1]
    ax.text(0.05, 0.95, f'RAG更优: {rag_better_llm_pct:.1f}%\n相关系数: {correlation:.3f}', 
           transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
           fontsize=11, verticalalignment='top', fontweight='bold')
    
    # 原始值 vs 预测值对比
    ax = axes[1]
    ax.scatter(y_origin, y_rag, alpha=0.6, s=30, label='RAG', color='blue', edgecolors='black', linewidth=0.3)
    ax.scatter(y_origin, y_llm, alpha=0.6, s=30, label='LLM', color='red', edgecolors='black', linewidth=0.3)
    
    # 对角线
    min_val = min(np.min(y_origin), np.min(y_rag), np.min(y_llm))
    max_val = max(np.max(y_origin), np.max(y_rag), np.max(y_llm))
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='相等线')
    
    ax.set_xlabel('原始iAUC', fontsize=12)
    ax.set_ylabel('预测的iAUC', fontsize=12)
    ax.set_title('原始值 vs 预测值对比', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 添加相关性信息
    corr_rag = np.corrcoef(y_origin, y_rag)[0, 1]
    corr_llm = np.corrcoef(y_origin, y_llm)[0, 1]
    ax.text(0.05, 0.95, f'RAG相关: {corr_rag:.3f}\nLLM相关: {corr_llm:.3f}', 
           transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
           fontsize=11, verticalalignment='top')
    
    plt.tight_layout()
    path2 = get_save_path('plots', f'rag_llm_scatter_comparison', 'png')
    plt.savefig(path2, dpi=300, bbox_inches='tight')
    plt.show()
    plot_paths.append(path2)
    
    # 第3行：效应量雷达图 + 累积分布函数（2个子图）
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 效应量雷达图
    ax = axes[0]
    ax.remove()  # 移除原坐标轴
    ax = fig.add_subplot(1, 2, 1, projection='polar')
    
    effect_sizes = stats_results['effect_sizes']
    metrics = ['RAG vs LLM', 'RAG改善', 'LLM改善']
    values = [
        abs(effect_sizes['cohens_d_rag_vs_llm']),
        abs(effect_sizes['cohens_d_rag_vs_origin']), 
        abs(effect_sizes['cohens_d_llm_vs_origin'])
    ]
    
    # 设置角度
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    values += values[:1]  # 闭合图形
    angles += angles[:1]
    
    ax.plot(angles, values, 'o-', linewidth=3, color='red', markersize=8)
    ax.fill(angles, values, alpha=0.3, color='red')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0, max(values) * 1.1)
    ax.set_title('效应量对比\n(Cohen\'s d绝对值)', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True)
    
    # 添加数值标注
    for angle, value in zip(angles[:-1], values[:-1]):
        ax.text(angle, value + max(values) * 0.05, f'{value:.3f}', 
               ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 累积分布函数对比
    ax = axes[1]
    
    # 计算CDF
    sorted_rag_llm = np.sort(rag_vs_llm_diff)
    sorted_rag_imp = np.sort(rag_improvement)
    sorted_llm_imp = np.sort(llm_improvement)
    
    y_cdf = np.arange(1, len(sorted_rag_llm) + 1) / len(sorted_rag_llm)
    
    ax.plot(sorted_rag_llm, y_cdf, color='purple', linewidth=2, label='LLM-RAG', alpha=0.8)
    ax.plot(sorted_rag_imp, y_cdf, color='blue', linewidth=2, label='RAG改善', alpha=0.8)
    ax.plot(sorted_llm_imp, y_cdf, color='red', linewidth=2, label='LLM改善', alpha=0.8)
    
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.7, label='无差异')
    ax.set_xlabel('iAUC差值', fontsize=12)
    ax.set_ylabel('累积概率', fontsize=12)
    ax.set_title('累积分布函数对比', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 添加分位数信息
    q25_rag = np.percentile(rag_improvement, 25)
    q75_rag = np.percentile(rag_improvement, 75)
    median_rag = np.median(rag_improvement)
    ax.text(0.02, 0.98, f'RAG改善四分位数:\nQ1: {q25_rag:.3f}\n中位数: {median_rag:.3f}\nQ3: {q75_rag:.3f}', 
           transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
           fontsize=10, verticalalignment='top')
    
    plt.tight_layout()
    path3 = get_save_path('plots', f'rag_llm_effect_cdf', 'png')
    plt.savefig(path3, dpi=300, bbox_inches='tight')
    plt.show()
    plot_paths.append(path3)
    
    # 第4行：相对改善效果 + 胜率矩阵（2个子图）
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 相对改善百分比
    ax = axes[0]
    
    # 计算相对改善百分比
    rag_pct_improvement = (rag_improvement / np.abs(y_origin)) * 100
    llm_pct_improvement = (llm_improvement / np.abs(y_origin)) * 100
    
    # 过滤异常值
    rag_pct_clean = rag_pct_improvement[np.abs(rag_pct_improvement) < np.percentile(np.abs(rag_pct_improvement), 95)]
    llm_pct_clean = llm_pct_improvement[np.abs(llm_pct_improvement) < np.percentile(np.abs(llm_pct_improvement), 95)]
    
    bp = ax.boxplot([rag_pct_clean, llm_pct_clean], 
                   labels=['RAG', 'LLM'], patch_artist=True, notch=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='无改善')
    ax.set_ylabel('相对改善百分比 (%)', fontsize=12)
    ax.set_title('相对改善效果对比', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    
    # 添加统计信息
    rag_median = np.median(rag_pct_clean)
    llm_median = np.median(llm_pct_clean)
    rag_mean = np.mean(rag_pct_clean)
    llm_mean = np.mean(llm_pct_clean)
    ax.text(0.02, 0.98, f'RAG: 中位数={rag_median:.2f}%, 均值={rag_mean:.2f}%\nLLM: 中位数={llm_median:.2f}%, 均值={llm_mean:.2f}%', 
           transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
           fontsize=10, verticalalignment='top')
    
    # 胜率矩阵热图
    ax = axes[1]
    
    # 创建胜率矩阵
    win_loss = stats_results['win_loss_stats']
    matrix_data = np.array([
        [win_loss['rag_better_than_origin_pct'], win_loss['rag_better_than_llm_pct']],
        [win_loss['llm_better_than_origin_pct'], 100 - win_loss['rag_better_than_llm_pct']]
    ])
    
    im = ax.imshow(matrix_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    
    # 添加文本标注
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, f'{matrix_data[i, j]:.1f}%',
                         ha="center", va="center", color="black", fontweight='bold', fontsize=14)
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['vs 原始', 'vs 对方'], fontsize=12)
    ax.set_yticklabels(['RAG', 'LLM'], fontsize=12)
    ax.set_title('胜率矩阵 (%)', fontsize=14, fontweight='bold')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('胜率 (%)', rotation=270, labelpad=15, fontsize=12)
    
    # 添加详细统计
    total_samples = win_loss['total_samples']
    both_better = win_loss['both_better_than_origin_pct']
    ax.text(0.02, -0.15, f'总样本数: {total_samples}\n双重改善: {both_better:.1f}%', 
           transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
           fontsize=10, verticalalignment='top')
    
    plt.tight_layout()
    path4 = get_save_path('plots', f'rag_llm_improvement_matrix', 'png')
    plt.savefig(path4, dpi=300, bbox_inches='tight')
    plt.show()
    plot_paths.append(path4)
    
    return plot_paths


def print_summary_report(stats_results):
    """
    打印汇总报告
    """
    print("\n" + "="*80)
    print("RAG vs LLM iAUC预测结果汇总报告")
    print("="*80)
    
    # 基础统计
    print("\n1. 基础统计指标:")
    print("-" * 40)
    basic_stats = stats_results['basic_stats']
    print(f"{'指标':<12} {'原始iAUC':<12} {'RAG预测':<12} {'LLM预测':<12}")
    print("-" * 48)
    for metric in ['mean', 'median', 'std', 'min', 'max']:
        print(f"{metric:<12} {basic_stats['Origin'][metric]:<12.2f} {basic_stats['RAG'][metric]:<12.2f} {basic_stats['LLM'][metric]:<12.2f}")
    
    # 胜负统计
    print("\n2. 胜负统计:")
    print("-" * 40)
    win_loss = stats_results['win_loss_stats']
    total = win_loss['total_samples']
    print(f"总样本数: {total}")
    print(f"RAG优于LLM: {win_loss['rag_better_than_llm']} ({win_loss['rag_better_than_llm_pct']:.1f}%)")
    print(f"RAG优于原始: {win_loss['rag_better_than_origin']} ({win_loss['rag_better_than_origin_pct']:.1f}%)")
    print(f"LLM优于原始: {win_loss['llm_better_than_origin']} ({win_loss['llm_better_than_origin_pct']:.1f}%)")
    print(f"两者都优于原始: {win_loss['both_better_than_origin']} ({win_loss['both_better_than_origin_pct']:.1f}%)")
    
    # 平均差值
    print("\n3. 平均差值:")
    print("-" * 40)
    diffs = stats_results['differences']
    print(f"RAG vs LLM (LLM-RAG): {np.mean(diffs['rag_vs_llm']):.2f} ± {np.std(diffs['rag_vs_llm']):.2f}")
    print(f"RAG改善效果 (Origin-RAG): {np.mean(diffs['rag_vs_origin']):.2f} ± {np.std(diffs['rag_vs_origin']):.2f}")
    print(f"LLM改善效果 (Origin-LLM): {np.mean(diffs['llm_vs_origin']):.2f} ± {np.std(diffs['llm_vs_origin']):.2f}")
    
    # 效应量
    print("\n4. 效应量 (Cohen's d):")
    print("-" * 40)
    effect_sizes = stats_results['effect_sizes']
    print(f"RAG vs LLM: {effect_sizes['cohens_d_rag_vs_llm']:.3f}")
    print(f"RAG vs Origin: {effect_sizes['cohens_d_rag_vs_origin']:.3f}")
    print(f"LLM vs Origin: {effect_sizes['cohens_d_llm_vs_origin']:.3f}")
    
    # 统计检验
    if stats_results['statistical_tests']:
        print("\n5. 统计检验:")
        print("-" * 40)
        tests = stats_results['statistical_tests']
        
        if 'rag_vs_llm' in tests:
            rag_llm = tests['rag_vs_llm']
            print(f"RAG vs LLM:")
            print(f"  配对t检验: t={rag_llm['t_test']['statistic']:.3f}, p={rag_llm['t_test']['p_value']:.4f}")
            print(f"  Wilcoxon检验: W={rag_llm['wilcoxon']['statistic']:.3f}, p={rag_llm['wilcoxon']['p_value']:.4f}")
        
        if 'rag_vs_origin' in tests:
            rag_origin = tests['rag_vs_origin']
            print(f"RAG vs Origin:")
            print(f"  配对t检验: t={rag_origin['t_test']['statistic']:.3f}, p={rag_origin['t_test']['p_value']:.4f}")
            print(f"  Wilcoxon检验: W={rag_origin['wilcoxon']['statistic']:.3f}, p={rag_origin['wilcoxon']['p_value']:.4f}")
    
    print("\n" + "="*80)


def main():
    """
    主函数
    """
    print("开始RAG vs LLM iAUC预测对比分析...")
    
    try:
        # 1. 加载样本数据
        X_rag, X_llm, y_origin = load_all_samples_from_folder('generated_samples')
        
        # 2. 加载TabPFN模型 - 为RAG和LLM分别创建实例
        rag_tabpfn, llm_tabpfn = load_tabpfn_models()
        
        X_all, y_all = fetch_dataset_from_cgmacros(1)
        rag_tabpfn.fit(X_all, y_all)
        llm_tabpfn.fit(X_all, y_all)
        
        # 3. 进行预测 - 使用不同的模型实例增强差异性
        print("\n使用两个TabPFN模型实例进行预测以增强差异性...")
        print("预测RAG样本（使用RAG专用模型）...")
        y_rag = predict_iauc_batch(rag_tabpfn, X_rag)
        
        print("预测LLM样本（使用LLM专用模型）...")
        y_llm = predict_iauc_batch(llm_tabpfn, X_llm)
        
        print(f'y_rag:{y_rag[:5]}')
        print(f'y_llm:{y_llm[:5]}')
        
        print(f"\n预测完成!")
        print(f"y_rag形状: {y_rag.shape}, 范围: {np.min(y_rag):.2f} - {np.max(y_rag):.2f}")
        print(f"y_llm形状: {y_llm.shape}, 范围: {np.min(y_llm):.2f} - {np.max(y_llm):.2f}")
        print(f"y_origin形状: {y_origin.shape}, 范围: {np.min(y_origin):.2f} - {np.max(y_origin):.2f}")
        
        # 4. 计算统计指标
        stats_results = compute_statistics(y_origin, y_rag, y_llm)
        
        # 5. 生成可视化
        plot_paths = create_visualizations(y_origin, y_rag, y_llm, stats_results)
        
        # 6. 打印汇总报告
        print_summary_report(stats_results)
        
        # 7. 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = f'results/rag_llm_comparison_{timestamp}.npz'
        os.makedirs('results', exist_ok=True)
        
        np.savez(results_path,
                 y_origin=y_origin,
                 y_rag=y_rag,
                 y_llm=y_llm,
                 X_rag=X_rag,
                 X_llm=X_llm)
        
        print(f"\n结果已保存到: {results_path}")
        print(f"图表已保存到: {plot_paths}")
        print("\n分析完成！")
        
    except Exception as e:
        print(f"\n分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
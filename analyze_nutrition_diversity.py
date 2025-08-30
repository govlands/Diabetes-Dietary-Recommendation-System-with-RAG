#!/usr/bin/env python3
"""
分析RAG vs LLM营养建议的多样性和个性化程度
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

def load_nutrition_data(folder='generated_samples'):
    """
    加载RAG和LLM的营养建议数据
    """
    rag_files = sorted(glob.glob(os.path.join(folder, 'rag_samples_*.csv')))
    
    all_rag_data = []
    all_llm_data = []
    
    for rag_file in rag_files:
        llm_file = rag_file.replace('rag_samples_', 'llm_samples_')
        
        rag_df = pd.read_csv(rag_file)
        llm_df = pd.read_csv(llm_file)
        
        all_rag_data.append(rag_df)
        all_llm_data.append(llm_df)
    
    rag_combined = pd.concat(all_rag_data, ignore_index=True)
    llm_combined = pd.concat(all_llm_data, ignore_index=True)
    
    return rag_combined, llm_combined

def analyze_value_distribution(rag_data, llm_data, nutrient='Carbs'):
    """
    分析特定营养素的数值分布特征
    """
    rag_values = rag_data[nutrient].values
    llm_values = llm_data[nutrient].values
    
    print(f"\n=== {nutrient} 分布分析 ===")
    
    # 基础统计
    print(f"RAG {nutrient}:")
    print(f"  范围: {np.min(rag_values):.1f} - {np.max(rag_values):.1f}")
    print(f"  均值: {np.mean(rag_values):.2f} ± {np.std(rag_values):.2f}")
    print(f"  中位数: {np.median(rag_values):.1f}")
    print(f"  唯一值数量: {len(np.unique(rag_values))}")
    
    print(f"\nLLM {nutrient}:")
    print(f"  范围: {np.min(llm_values):.1f} - {np.max(llm_values):.1f}")
    print(f"  均值: {np.mean(llm_values):.2f} ± {np.std(llm_values):.2f}")
    print(f"  中位数: {np.median(llm_values):.1f}")
    print(f"  唯一值数量: {len(np.unique(llm_values))}")
    
    # 多样性指标
    # 1. 香农熵 (Shannon Entropy)
    def shannon_entropy(values):
        value_counts = Counter(values)
        total = len(values)
        entropy = -sum((count/total) * np.log2(count/total) for count in value_counts.values())
        return entropy
    
    rag_entropy = shannon_entropy(rag_values)
    llm_entropy = shannon_entropy(llm_values)
    
    print(f"\n多样性指标:")
    print(f"  RAG 香农熵: {rag_entropy:.3f}")
    print(f"  LLM 香农熵: {llm_entropy:.3f}")
    print(f"  熵比值 (RAG/LLM): {rag_entropy/llm_entropy:.3f}")
    
    # 2. 基尼系数 (衡量分布不均匀程度)
    def gini_coefficient(values):
        values = np.array(values)
        n = len(values)
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * np.sort(values))) / (n * np.sum(values)) - (n + 1) / n
    
    rag_gini = gini_coefficient(rag_values)
    llm_gini = gini_coefficient(llm_values)
    
    print(f"  RAG 基尼系数: {rag_gini:.3f}")
    print(f"  LLM 基尼系数: {llm_gini:.3f}")
    
    # 3. 变异系数 (标准差/均值)
    rag_cv = np.std(rag_values) / np.mean(rag_values)
    llm_cv = np.std(llm_values) / np.mean(llm_values)
    
    print(f"  RAG 变异系数: {rag_cv:.3f}")
    print(f"  LLM 变异系数: {llm_cv:.3f}")
    
    # 4. 最常见值的占比
    rag_counter = Counter(rag_values)
    llm_counter = Counter(llm_values)
    
    rag_most_common = rag_counter.most_common(5)
    llm_most_common = llm_counter.most_common(5)
    
    print(f"\n最常见的5个值:")
    print(f"  RAG: {rag_most_common}")
    print(f"  LLM: {llm_most_common}")
    
    rag_top_ratio = rag_most_common[0][1] / len(rag_values)
    llm_top_ratio = llm_most_common[0][1] / len(llm_values)
    
    print(f"  RAG 最常见值占比: {rag_top_ratio:.3f}")
    print(f"  LLM 最常见值占比: {llm_top_ratio:.3f}")
    
    return {
        'rag_entropy': rag_entropy,
        'llm_entropy': llm_entropy,
        'rag_gini': rag_gini,
        'llm_gini': llm_gini,
        'rag_cv': rag_cv,
        'llm_cv': llm_cv,
        'rag_unique': len(np.unique(rag_values)),
        'llm_unique': len(np.unique(llm_values)),
        'rag_top_ratio': rag_top_ratio,
        'llm_top_ratio': llm_top_ratio,
        'rag_values': rag_values,
        'llm_values': llm_values
    }

def analyze_clustering_patterns(rag_data, llm_data, nutrients=['Carbs', 'Protein', 'Fat', 'Fiber']):
    """
    分析营养建议的聚类模式
    """
    print(f"\n=== 聚类模式分析 ===")
    
    rag_features = rag_data[nutrients].values
    llm_features = llm_data[nutrients].values
    
    # 尝试不同的聚类数量
    silhouette_scores = {}
    
    for data_type, features in [('RAG', rag_features), ('LLM', llm_features)]:
        scores = []
        k_range = range(2, min(11, len(np.unique(features, axis=0))))
        
        for k in k_range:
            if k < len(features):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(features)
                score = silhouette_score(features, cluster_labels)
                scores.append(score)
            else:
                scores.append(0)
        
        silhouette_scores[data_type] = scores
        best_k = k_range[np.argmax(scores)]
        best_score = max(scores)
        
        print(f"{data_type} 最佳聚类数: {best_k}, 轮廓系数: {best_score:.3f}")
    
    return silhouette_scores

def create_diversity_visualizations(rag_data, llm_data, nutrients=['Carbs', 'Protein', 'Fat', 'Fiber']):
    """
    创建多样性分析的可视化图表
    """
    print("\n生成多样性分析可视化...")
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 为每个营养素分析
    diversity_metrics = {}
    
    for i, nutrient in enumerate(nutrients):
        diversity_metrics[nutrient] = analyze_value_distribution(rag_data, llm_data, nutrient)
    
    plot_paths = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    
    # =================== 图表1: 分布直方图对比 ===================
    fig1 = plt.figure(figsize=(16, 12))
    
    # 1. 分布直方图对比 (2x2)
    for i, nutrient in enumerate(nutrients):
        plt.subplot(2, 2, i + 1)
        
        rag_values = diversity_metrics[nutrient]['rag_values']
        llm_values = diversity_metrics[nutrient]['llm_values']
        
        # 使用相同的bins范围
        min_val = min(np.min(rag_values), np.min(llm_values))
        max_val = max(np.max(rag_values), np.max(llm_values))
        bins = np.linspace(min_val, max_val, 25)
        
        plt.hist(rag_values, bins=bins, alpha=0.6, label='RAG', color='blue', density=True)
        plt.hist(llm_values, bins=bins, alpha=0.6, label='LLM', color='red', density=True)
        
        plt.xlabel(f'{nutrient} (g)', fontsize=12)
        plt.ylabel('密度', fontsize=12)
        plt.title(f'{nutrient} 分布对比', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
    
    plt.suptitle('营养素分布对比', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plot_path1 = f'plots/nutrition_diversity_distributions_{timestamp}.png'
    os.makedirs('plots', exist_ok=True)
    plt.savefig(plot_path1, dpi=300, bbox_inches='tight')
    plt.show()
    plot_paths.append(plot_path1)
    
    # =================== 图表2: 多样性指标对比 ===================
    fig2 = plt.figure(figsize=(16, 12))
    
    nutrients_list = list(nutrients)
    x = np.arange(len(nutrients_list))
    width = 0.35
    
    # 2. 唯一值数量对比
    plt.subplot(2, 2, 1)
    rag_unique = [diversity_metrics[n]['rag_unique'] for n in nutrients_list]
    llm_unique = [diversity_metrics[n]['llm_unique'] for n in nutrients_list]
    
    plt.bar(x - width/2, rag_unique, width, label='RAG', color='blue', alpha=0.7)
    plt.bar(x + width/2, llm_unique, width, label='LLM', color='red', alpha=0.7)
    
    plt.xlabel('营养素', fontsize=12)
    plt.ylabel('唯一值数量', fontsize=12)
    plt.title('营养建议唯一值数量对比', fontsize=14, fontweight='bold')
    plt.xticks(x, nutrients_list)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # 3. 香农熵对比
    plt.subplot(2, 2, 2)
    rag_entropy = [diversity_metrics[n]['rag_entropy'] for n in nutrients_list]
    llm_entropy = [diversity_metrics[n]['llm_entropy'] for n in nutrients_list]
    
    plt.bar(x - width/2, rag_entropy, width, label='RAG', color='blue', alpha=0.7)
    plt.bar(x + width/2, llm_entropy, width, label='LLM', color='red', alpha=0.7)
    
    plt.xlabel('营养素', fontsize=12)
    plt.ylabel('香农熵', fontsize=12)
    plt.title('营养建议多样性(香农熵)对比', fontsize=14, fontweight='bold')
    plt.xticks(x, nutrients_list)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # 4. 变异系数对比
    plt.subplot(2, 2, 3)
    rag_cv = [diversity_metrics[n]['rag_cv'] for n in nutrients_list]
    llm_cv = [diversity_metrics[n]['llm_cv'] for n in nutrients_list]
    
    plt.bar(x - width/2, rag_cv, width, label='RAG', color='blue', alpha=0.7)
    plt.bar(x + width/2, llm_cv, width, label='LLM', color='red', alpha=0.7)
    
    plt.xlabel('营养素', fontsize=12)
    plt.ylabel('变异系数', fontsize=12)
    plt.title('营养建议变异性对比', fontsize=14, fontweight='bold')
    plt.xticks(x, nutrients_list)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # 5. 最常见值占比对比
    plt.subplot(2, 2, 4)
    rag_top_ratio = [diversity_metrics[n]['rag_top_ratio'] for n in nutrients_list]
    llm_top_ratio = [diversity_metrics[n]['llm_top_ratio'] for n in nutrients_list]
    
    plt.bar(x - width/2, rag_top_ratio, width, label='RAG', color='blue', alpha=0.7)
    plt.bar(x + width/2, llm_top_ratio, width, label='LLM', color='red', alpha=0.7)
    
    plt.xlabel('营养素', fontsize=12)
    plt.ylabel('最常见值占比', fontsize=12)
    plt.title('营养建议集中度对比', fontsize=14, fontweight='bold')
    plt.xticks(x, nutrients_list)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('营养素多样性指标对比', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plot_path2 = f'plots/nutrition_diversity_metrics_{timestamp}.png'
    plt.savefig(plot_path2, dpi=300, bbox_inches='tight')
    plt.show()
    plot_paths.append(plot_path2)
    
    # =================== 图表3: 值频次分布分析 ===================
    fig3 = plt.figure(figsize=(16, 12))
    
    # 6-9. 每个营养素的值频次热图
    for i, nutrient in enumerate(nutrients):
        plt.subplot(2, 2, i + 1)
        
        rag_values = diversity_metrics[nutrient]['rag_values']
        llm_values = diversity_metrics[nutrient]['llm_values']
        
        # 计算值的频次
        all_values = np.concatenate([rag_values, llm_values])
        unique_values = np.unique(all_values)
        
        rag_counts = []
        llm_counts = []
        
        for val in unique_values:
            rag_counts.append(np.sum(rag_values == val))
            llm_counts.append(np.sum(llm_values == val))
        
        # 只显示前20个最常见的值
        top_indices = np.argsort(np.array(rag_counts) + np.array(llm_counts))[-20:]
        
        if len(top_indices) > 0:
            plot_values = unique_values[top_indices]
            plot_rag = np.array(rag_counts)[top_indices]
            plot_llm = np.array(llm_counts)[top_indices]
            
            x_pos = np.arange(len(plot_values))
            plt.bar(x_pos - width/2, plot_rag, width, label='RAG', color='blue', alpha=0.7)
            plt.bar(x_pos + width/2, plot_llm, width, label='LLM', color='red', alpha=0.7)
            
            plt.xlabel(f'{nutrient} 值', fontsize=12)
            plt.ylabel('频次', fontsize=12)
            plt.title(f'{nutrient} 值频次分布', fontsize=14, fontweight='bold')
            plt.xticks(x_pos, [f'{v:.0f}' for v in plot_values], rotation=45)
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
    
    plt.suptitle('营养素值频次分布分析', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plot_path3 = f'plots/nutrition_diversity_frequencies_{timestamp}.png'
    plt.savefig(plot_path3, dpi=300, bbox_inches='tight')
    plt.show()
    plot_paths.append(plot_path3)
    
    # =================== 图表4: 综合分析 ===================
    fig4 = plt.figure(figsize=(16, 12))
    
    # 10. 2D散点图 - Carbs vs Protein
    plt.subplot(2, 2, 1)
    plt.scatter(rag_data['Carbs'], rag_data['Protein'], alpha=0.6, s=30, label='RAG', color='blue')
    plt.scatter(llm_data['Carbs'], llm_data['Protein'], alpha=0.6, s=30, label='LLM', color='red')
    plt.xlabel('碳水化合物 (g)', fontsize=12)
    plt.ylabel('蛋白质 (g)', fontsize=12)
    plt.title('碳水-蛋白质分布模式', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # 11. 2D散点图 - Fat vs Fiber
    plt.subplot(2, 2, 2)
    plt.scatter(rag_data['Fat'], rag_data['Fiber'], alpha=0.6, s=30, label='RAG', color='blue')
    plt.scatter(llm_data['Fat'], llm_data['Fiber'], alpha=0.6, s=30, label='LLM', color='red')
    plt.xlabel('脂肪 (g)', fontsize=12)
    plt.ylabel('纤维 (g)', fontsize=12)
    plt.title('脂肪-纤维分布模式', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # 12. 综合多样性指标雷达图
    ax = plt.subplot(2, 2, 3, projection='polar')
    
    # 计算平均多样性指标
    avg_rag_entropy = np.mean([diversity_metrics[n]['rag_entropy'] for n in nutrients])
    avg_llm_entropy = np.mean([diversity_metrics[n]['llm_entropy'] for n in nutrients])
    avg_rag_cv = np.mean([diversity_metrics[n]['rag_cv'] for n in nutrients])
    avg_llm_cv = np.mean([diversity_metrics[n]['llm_cv'] for n in nutrients])
    avg_rag_unique_ratio = np.mean([diversity_metrics[n]['rag_unique']/len(diversity_metrics[n]['rag_values']) for n in nutrients])
    avg_llm_unique_ratio = np.mean([diversity_metrics[n]['llm_unique']/len(diversity_metrics[n]['llm_values']) for n in nutrients])
    
    metrics = ['香农熵', '变异系数', '唯一值比率']
    rag_values_radar = [avg_rag_entropy/max(avg_rag_entropy, avg_llm_entropy), 
                       avg_rag_cv/max(avg_rag_cv, avg_llm_cv),
                       avg_rag_unique_ratio/max(avg_rag_unique_ratio, avg_llm_unique_ratio)]
    llm_values_radar = [avg_llm_entropy/max(avg_rag_entropy, avg_llm_entropy),
                       avg_llm_cv/max(avg_rag_cv, avg_llm_cv), 
                       avg_llm_unique_ratio/max(avg_rag_unique_ratio, avg_llm_unique_ratio)]
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    rag_values_radar += rag_values_radar[:1]
    llm_values_radar += llm_values_radar[:1]
    angles += angles[:1]
    
    ax.plot(angles, rag_values_radar, 'o-', linewidth=2, label='RAG', color='blue')
    ax.fill(angles, rag_values_radar, alpha=0.25, color='blue')
    ax.plot(angles, llm_values_radar, 'o-', linewidth=2, label='LLM', color='red')
    ax.fill(angles, llm_values_radar, alpha=0.25, color='red')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title('综合多样性指标对比', pad=20, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    
    # 13. 个性化程度分析
    plt.subplot(2, 2, 4)
    
    # 计算每个样本与其最近邻的距离 (个性化程度)
    from sklearn.neighbors import NearestNeighbors
    
    rag_features = rag_data[nutrients].values
    llm_features = llm_data[nutrients].values
    
    # 计算最近邻距离
    def avg_nearest_neighbor_distance(data, k=5):
        if len(data) <= k:
            return 0
        nn = NearestNeighbors(n_neighbors=k+1)
        nn.fit(data)
        distances, _ = nn.kneighbors(data)
        return np.mean(distances[:, 1:])  # 排除自己
    
    rag_nn_dist = avg_nearest_neighbor_distance(rag_features)
    llm_nn_dist = avg_nearest_neighbor_distance(llm_features)
    
    plt.bar(['RAG', 'LLM'], [rag_nn_dist, llm_nn_dist], 
           color=['blue', 'red'], alpha=0.7)
    plt.ylabel('平均最近邻距离', fontsize=12)
    plt.title('个性化程度(最近邻距离)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 添加数值标注
    plt.text(0, rag_nn_dist + rag_nn_dist*0.05, f'{rag_nn_dist:.2f}', ha='center', fontweight='bold', fontsize=11)
    plt.text(1, llm_nn_dist + llm_nn_dist*0.05, f'{llm_nn_dist:.2f}', ha='center', fontweight='bold', fontsize=11)
    
    plt.suptitle('营养素综合分布分析', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plot_path4 = f'plots/nutrition_diversity_comprehensive_{timestamp}.png'
    plt.savefig(plot_path4, dpi=300, bbox_inches='tight')
    plt.show()
    plot_paths.append(plot_path4)
    
    return plot_paths, diversity_metrics

def statistical_significance_test(rag_data, llm_data, nutrients=['Carbs', 'Protein', 'Fat', 'Fiber']):
    """
    统计显著性检验
    """
    print(f"\n=== 统计显著性检验 ===")
    
    for nutrient in nutrients:
        rag_values = rag_data[nutrient].values
        llm_values = llm_data[nutrient].values
        
        # Kolmogorov-Smirnov 检验 (检验分布是否相同)
        ks_stat, ks_p = stats.ks_2samp(rag_values, llm_values)
        
        # Levene 检验 (检验方差是否相同)
        levene_stat, levene_p = stats.levene(rag_values, llm_values)
        
        print(f"\n{nutrient}:")
        print(f"  KS检验: 统计量={ks_stat:.4f}, p值={ks_p:.4f}")
        print(f"  Levene检验: 统计量={levene_stat:.4f}, p值={levene_p:.4f}")
        
        if ks_p < 0.05:
            print(f"  → RAG和LLM的{nutrient}分布显著不同 (p<0.05)")
        else:
            print(f"  → RAG和LLM的{nutrient}分布无显著差异 (p>=0.05)")
            
        if levene_p < 0.05:
            print(f"  → RAG和LLM的{nutrient}方差显著不同 (p<0.05)")
        else:
            print(f"  → RAG和LLM的{nutrient}方差无显著差异 (p>=0.05)")

def main():
    """
    主函数
    """
    print("开始分析RAG vs LLM营养建议的多样性和个性化程度...")
    
    # data = np.load('results/filtered_rag_llm_data_20250829_131249.npz')
    # features = ['Carbs', 'Protein', 'Fat', 'Fiber', 'Baseline_Libre', 'Age', 'Gender', 
    #             'BMI', 'A1c', 'HOMA', 'Insulin', 'TG', 'Cholesterol', 'HDL', 'Non HDL', 
    #             'LDL', 'VLDL', 'CHO/HDL ratio', 'Fasting BG']
    # X_rag = data['X_rag']
    # X_llm = data['X_llm']
    
    # # 加载数据
    # rag_data, llm_data = pd.DataFrame(X_rag, columns=features), pd.DataFrame(X_llm, columns=features)
    rag_data, llm_data = load_nutrition_data()
    
    print(f"加载完成:")
    print(f"  RAG样本数: {len(rag_data)}")
    print(f"  LLM样本数: {len(llm_data)}")
    
    nutrients = ['Carbs', 'Protein', 'Fat', 'Fiber']
    
    # 1. 分析每个营养素的分布特征
    diversity_metrics = {}
    for nutrient in nutrients:
        diversity_metrics[nutrient] = analyze_value_distribution(rag_data, llm_data, nutrient)
    
    # 2. 聚类模式分析
    silhouette_scores = analyze_clustering_patterns(rag_data, llm_data, nutrients)
    
    # 3. 统计显著性检验
    statistical_significance_test(rag_data, llm_data, nutrients)
    
    # 4. 生成可视化
    plot_paths, _ = create_diversity_visualizations(rag_data, llm_data, nutrients)
    
    # 5. 总结报告
    print(f"\n" + "="*80)
    print("RAG vs LLM 营养建议多样性分析总结")
    print("="*80)
    
    for nutrient in nutrients:
        metrics = diversity_metrics[nutrient]
        print(f"\n{nutrient}:")
        print(f"  多样性优势: {'RAG' if metrics['rag_entropy'] > metrics['llm_entropy'] else 'LLM'}")
        print(f"  个性化优势: {'RAG' if metrics['rag_cv'] > metrics['llm_cv'] else 'LLM'}")
        print(f"  唯一值数量: RAG={metrics['rag_unique']}, LLM={metrics['llm_unique']}")
        print(f"  集中度: RAG={metrics['rag_top_ratio']:.3f}, LLM={metrics['llm_top_ratio']:.3f}")
    
    print(f"\n可视化图表已保存到:")
    for i, path in enumerate(plot_paths, 1):
        print(f"  图表{i}: {path}")
    print("\n分析完成！")

if __name__ == "__main__":
    from datetime import datetime
    main()

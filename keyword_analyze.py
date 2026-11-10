import collections
import os
import igraph as ig
import numpy as np
def analyze_keywords_distribution(gml_path):
    """
    读取GML文件并统计关键词分布情况
    """
    if not os.path.exists(gml_path):
        print(f"Error: File {gml_path} not found.")
        return

    # 加载图
    g = ig.Graph.Read_GML(gml_path)
    print(f"\n--- Statistics for: {os.path.basename(gml_path)} ---")
    print(f"Nodes: {g.vcount()}, Edges: {g.ecount()}")

    # 提取所有关键词
    all_keywords_list = []
    nodes_with_keywords = 0
    
    for v in g.vs:
        kw_str = v['keywords']
        if kw_str and kw_str != "0":
            kws = kw_str.split(",")
            all_keywords_list.extend(kws)
            nodes_with_keywords += 1

    # 统计频率
    counter = collections.Counter(all_keywords_list)
    total_unique_kws = len(counter)
    
    # 排序
    most_common = counter.most_common(100) # 出现频率最高的前20个词

    print(f"Nodes with valid keywords: {nodes_with_keywords} / {g.vcount()}")
    print(f"Total unique keywords found: {total_unique_kws}")
    print(f"\nTop 100 Most Frequent Keywords:")
    print(f"{'KW_ID':>10} | {'Count':>10} | {'Frequency %':>12}")
    print("-" * 40)
    for kw, count in most_common:
        freq = (count / nodes_with_keywords) * 100
        print(f"{kw:>10} | {count:>10} | {freq:>11.2f}%")

    # 获取倒数 20 个 (返回的是频率从低到高的顺序)
    least_common = counter.most_common()[:-21:-1] 

    print(f"\nBottom 20 Least Frequent Keywords:")
    print(f"{'KW_ID':>10} | {'Count':>10} | {'Frequency %':>12}")
    print("-" * 40)
    for kw, count in least_common:
        freq = (count / nodes_with_keywords) * 100
        print(f"{kw:>10} | {count:>10} | {freq:>11.2f}%")

    # 统计分布紧凑度
    counts = list(counter.values())
    print(f"\nDistribution Summary:")
    print(f"- Max count: {np.max(counts)}")
    print(f"- Min count: {np.min(counts)}")
    print(f"- Mean count: {np.mean(counts):.2f}")
    print(f"- Std deviation: {np.std(counts):.2f}")
    
    # 如果某个词频率超过 50%，剪枝效果会非常差
    high_freq_alert = [kw for kw, c in counter.items() if (c / nodes_with_keywords) > 0.5]
    if high_freq_alert:
        print(f"\n⚠️ WARNING: Keywords {high_freq_alert} appear in >50% of nodes. Pruning will be ineffective!")

    return counter

if __name__ == '__main__':

    full_path = "./redataset/precompute/real/Facebook/G+-4039-88234.gml"
    # full_path = "./redataset/precompute/real/PubMed/G+-19717-88648.gml"
    # full_path = "./redataset/precompute/real/EL/G+-203769-234355.gml"
    # full_path = "./redataset/precompute/real/TWeibo/G+-2320895-50655143.gml"
    kw_counts = analyze_keywords_distribution(full_path)
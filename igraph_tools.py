import igraph as ig
import os
import time
from collections import deque
import json
import random
import copy

random.seed(42)

def read_query_keywords(file_path):
    keywords = {}
    with open(file_path, 'r') as file:
        for line in file:
            # 去掉换行符并按冒号分割
            node, keywords_str = line.strip().split(':')
            node = int(node)  # 将顶点编号转换为整数
            keywords_list = [kw.strip() for kw in keywords_str.split(',')]  # 提取关键词列表
            keywords[node] = keywords_list
    return keywords

def read_query_edges(file_path):
    edges = []
    with open(file_path, 'r') as file:
        for line in file:
            # 去掉换行符并按空格分割
            u, v = map(int, line.strip().split())
            edges.append((u, v))
    return edges



def query_keywords_to_BV(G_q:ig.Graph, keywords_domain:str, m:int):
    num_keywords = len(keywords_domain)
    group_size = num_keywords // m
    remainder = num_keywords % m
    
    groups = [[] for _ in range(m)]
    index = 0
    for i in range(m):
        group_size_i = group_size + (1 if i < remainder else 0)
        groups[i] = keywords_domain[index:index + group_size_i]
        index += group_size_i

    keyword_to_index_per_group = [
        {keyword: index for index, keyword in enumerate(group)} for group in groups
    ]

    # print("keyword_to_index_per_group:{}".format(keyword_to_index_per_group))

    for node in G_q.vs():
        node['BV'] = {}
        bit_vectors = []
        keywords = node['keywords'] # get访问，默认空
        for group_index in range(m):
            bv = 0
            for keyword in keywords:
                if keyword in keyword_to_index_per_group[group_index]:
                    index = keyword_to_index_per_group[group_index][keyword]
                    bv |= (1 << index)
            bit_vectors.append(bv)

        node['BV'] = bit_vectors
    
    return G_q

# def generate_query(G:ig.Graph, n:int, p:float, keyword_domain):
#     # subgraph
#     subgraph = get_random_subgraph(G, n)
#     query_graph = remove_edges(subgraph=subgraph, p=p)
#     query_graph = sample_keywords_subset(query_graph, keywords_domain=keyword_domain)
#     return query_graph

def generate_query(G:ig.Graph, n:int, p:float, keyword_domain, overlap_ratio:float=1.0):
    # subgraph
    subgraph = get_random_subgraph(G, n)
    query_graph = remove_edges(subgraph=subgraph, p=p)
    query_graph = sample_keywords_subset(query_graph, keywords_domain=keyword_domain, overlap_ratio=overlap_ratio)
    return query_graph

def get_random_subgraph(G: ig.Graph, n: int):
    while True:
        # 随机选择一个起始节点，且该节点的 keywords 不为 "0"
        valid_nodes = [int(node["id"]) for node in G.vs() if node["keywords"] != "0"]
        if not valid_nodes:
            raise ValueError("No valid starting node found (all nodes have keywords='0').")
        
        start_node = random.choice(valid_nodes)
        selected_nodes = {start_node}

        # print(f"selected {selected_nodes}")
        while len(selected_nodes) < n:
            neighbors = set()
            for node in selected_nodes:
                # 获取邻居节点，并排除 keywords 为 "0" 的节点
                neighbors.update(neighbor for neighbor in G.neighbors(node) if G.vs[neighbor]["keywords"] != "0")

            neighbors -= selected_nodes

            if not neighbors:
                print("Cannot expand the subgraph while maintaining connectivity.")
                break
            
            # print(neighbors)
            new_node = random.choice(list(neighbors))
            # print(new_node)
            selected_nodes.add(new_node)

        if len(selected_nodes) == n:
            # print(f"selected {selected_nodes}")
            return G.subgraph(selected_nodes)
        else:
            continue

def remove_edges(subgraph:ig.Graph, p=0.3):
    """
    随机裁剪子图的边，以概率 p 对每条边进行裁剪，同时保证子图的连通性。
    """
    edges = subgraph.get_edgelist()
    for edge in edges:
        if random.random() < p:
            subgraph.delete_edges([edge])
            if not subgraph.is_connected():
                subgraph.add_edges([edge])
    return subgraph

def sample_keywords_subset(subgraph:ig.Graph, keywords_domain, overlap_ratio:float=1.0):
    """
    对子图中每个节点的 keywords 属性进行子集采样。
    overlap_ratio: [0,1]，控制查询关键词中来自原节点关键词的比例。
      1.0 = 全部来自原节点（原始行为）；0.5 = 一半来自原节点，一半来自域中其他关键词。
    至少保证 1 个关键词与原节点重叠（保证有匹配候选）。
    """
    domain_list = list(keywords_domain)
    for node in subgraph.vs():
        keywords = [int(k) for k in node["keywords"].split(",")]
        sample_size = min(len(keywords), 2)

        # 至少 1 个重叠关键词，其余按 overlap_ratio 决定
        overlap_count = max(1, round(sample_size * overlap_ratio))
        overlap_count = min(overlap_count, len(keywords))
        non_overlap_count = sample_size - overlap_count

        overlapping = random.sample(keywords, overlap_count)

        if non_overlap_count > 0:
            outside = [k for k in domain_list if k not in set(keywords)]
            if len(outside) >= non_overlap_count:
                non_overlapping = random.sample(outside, non_overlap_count)
            else:
                # 域中没有足够的非重叠关键词，退化为全重叠
                extra = min(len(keywords) - overlap_count, non_overlap_count)
                remaining = [k for k in keywords if k not in overlapping]
                non_overlapping = random.sample(remaining, extra) if extra > 0 else []
        else:
            non_overlapping = []

        sampled = overlapping + non_overlapping
        node['keywords'] = ",".join(map(str, sampled))

    return subgraph

import random
from collections import Counter
import igraph as ig

def generate_query_exclude_top30(G: ig.Graph, n: int, p: float, keyword_domain, overlap_ratio: float = 1.0):
    """
    主入口：生成查询图，并严格排除全图频率前30的高频关键词。
    """
    # 1. 统计全图关键词频率，并找出前 30 个高频词
    keyword_counts = Counter()
    for node in G.vs():
        if node["keywords"] and node["keywords"] != "0":
            # 转换为 int 统一处理
            keywords = [int(k) for k in node["keywords"].split(",") if k.strip()]
            keyword_counts.update(keywords)
    
    # 提取频率前 30 的关键词集合
    top30_tuples = keyword_counts.most_common(50) # 70pm 700FB
    top30_keywords = {k for k, count in top30_tuples}
    
    # print(f"[INFO] 成功识别并屏蔽全图频率 Top-30 的关键词: {sorted(list(top30_keywords))}")

    # 2. 从总体关键词域中剔除 Top-30 词汇
    safe_domain = [k for k in keyword_domain if int(k) not in top30_keywords]
    if not safe_domain:
        raise ValueError("安全关键词域（安全 Domain）为空，无法进行采样。")

    # 3. 提取随机连通子图 (过滤掉只含有 "0" 或 仅含Top-30词 的节点)
    subgraph = get_random_subgraph_safe(G, n, top30_keywords)
    
    # 4. 随机裁剪边并保持连通性
    query_graph = remove_edges(subgraph=subgraph, p=p)
    
    # 5. 节点关键词子集采样 (过滤 Top-30 词，并用安全域词汇做兜底)
    query_graph = sample_keywords_subset_safe(
        query_graph, 
        safe_domain=safe_domain, 
        top30_keywords=top30_keywords, 
        overlap_ratio=overlap_ratio
    )
    
    return query_graph


def get_random_subgraph_safe(G: ig.Graph, n: int, top30_keywords: set):
    """
    获取随机子图：确保起始节点和邻居节点在去除 "0" 和 Top-30 关键词后，依然保留有合法的关键词。
    """
    def is_valid_node(node):
        if node["keywords"] == "0":
            return False
        # 解析节点的关键词
        node_k = {int(k) for k in node["keywords"].split(",") if k.strip()}
        # 如果该节点的关键词去掉 Top-30 后什么都不剩了，就认为该节点无效
        return len(node_k - top30_keywords) > 0

    while True:
        # 使用节点的索引 index 来保证稳定
        valid_nodes = [v.index for v in G.vs() if is_valid_node(v)]
        if not valid_nodes:
            raise ValueError("未在原图中找到符合要求的节点（所有节点的关键词都是 '0' 或属于 Top-30 词）。")
        
        start_node = random.choice(valid_nodes)
        selected_nodes = {start_node}

        while len(selected_nodes) < n:
            neighbors = set()
            for node_idx in selected_nodes:
                # 寻找有效的邻居
                neighbors.update(
                    nbr for nbr in G.neighbors(node_idx) if is_valid_node(G.vs[nbr])
                )

            neighbors -= selected_nodes

            if not neighbors:
                # 无法在外扩的同时保持连通性，打破内循环重新选起点
                break
            
            new_node = random.choice(list(neighbors))
            selected_nodes.add(new_node)

        if len(selected_nodes) == n:
            return G.subgraph(selected_nodes)
        else:
            continue


def remove_edges(subgraph: ig.Graph, p=0.3):
    """
    保持原逻辑：随机裁剪子图的边，以概率 p 对每条边进行裁剪，同时保证子图的连通性。
    """
    edges = subgraph.get_edgelist()
    for edge in edges:
        if random.random() < p:
            subgraph.delete_edges([edge])
            if not subgraph.is_connected():
                subgraph.add_edges([edge])
    return subgraph


def sample_keywords_subset_safe(subgraph: ig.Graph, safe_domain, top30_keywords: set, overlap_ratio: float = 1.0):
    """
    对子图中每个节点的关键词进行子集采样，确保最终结果中没有任何 Top-30 关键词。
    """
    domain_list = list(safe_domain)
    
    for node in subgraph.vs():
        # 1. 提取当前节点的所有原始关键词
        raw_keywords = [int(k) for k in node["keywords"].split(",") if k.strip() and k != "0"]
        
        # 2. 严格过滤掉属于 Top-30 的高频关键词
        safe_keywords = [k for k in raw_keywords if k not in top30_keywords]
        
        # 3. 健壮性兜底：如果该节点过滤后没有合法的词，直接从安全域随机派发
        if not safe_keywords:
            sampled = random.sample(domain_list, 1)
            node['keywords'] = ",".join(map(str, sampled))
            continue

        # 4. 正常的采样逻辑（针对安全词）
        sample_size = min(len(safe_keywords), 2)

        # 至少 1 个重叠关键词，其余按 overlap_ratio 决定
        overlap_count = max(1, round(sample_size * overlap_ratio))
        overlap_count = min(overlap_count, len(safe_keywords))
        non_overlap_count = sample_size - overlap_count

        overlapping = random.sample(safe_keywords, overlap_count)

        if non_overlap_count > 0:
            # 从 safe_domain 中选出不在当前节点 safe_keywords 里的词
            outside = [k for k in domain_list if k not in set(safe_keywords)]
            if len(outside) >= non_overlap_count:
                non_overlapping = random.sample(outside, non_overlap_count)
            else:
                # 安全域不够大时，退化为从当前节点剩余的安全词中选
                extra = min(len(safe_keywords) - overlap_count, non_overlap_count)
                remaining = [k for k in safe_keywords if k not in overlapping]
                non_overlapping = random.sample(remaining, extra) if extra > 0 else []
        else:
            non_overlapping = []

        sampled = overlapping + non_overlapping
        node['keywords'] = ",".join(map(str, sampled))

    return subgraph

def save_Graph(Graph:ig.Graph, dataset_name:str, is_Aux:bool) -> bool:
    folder_name = os.path.join(
        "redataset",
        "precompute",
        "real",
        dataset_name
    )
    create_folder(folder_name)
    initial_directory = os.getcwd()
    os.chdir(folder_name)
    # G: 经过pre，没有aux；G+：经过offline，具备aux
    if is_Aux:
        Graph.write_gml('G+-{}-{}.gml'.format(
                Graph.number_of_nodes(),
                Graph.number_of_edges(),))
        print(folder_name, 'G+-{}-{}.gml'.format(
                Graph.number_of_nodes(),
                Graph.number_of_edges(),), 'saved successfully!')
    else:
        Graph.write_gml('G-{}-{}.gml'.format(
                Graph.number_of_nodes(),
                Graph.number_of_edges(),))
        print(folder_name, 'G-{}-{}.gml'.format(
                Graph.number_of_nodes(),
                Graph.number_of_edges(),), 'saved successfully!')
    os.chdir(initial_directory)

def save_Graph_synthetic(Graph:ig.Graph, dataset_name:str, distribution:str) -> bool:
    folder_name = os.path.join(
        "redataset",
        "precompute",
        "synthetic",
        dataset_name
    )
    create_folder(folder_name)
    initial_directory = os.getcwd()
    os.chdir(folder_name)
    # G: 经过pre，没有aux；G+：经过offline，具备aux

    Graph.write_gml('G+-{}.gml'.format(distribution))
    print(folder_name, 'G+-{}.gml'.format(distribution), 'saved successfully!')
    os.chdir(initial_directory)

def create_folder(folder_name: str) -> bool:
    folder_path = os.path.abspath(folder_name)
    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path)
            print(f"folder create sucess: {folder_path}")
            return True
        except Exception as e:
            print(f"folder create fail: {e}")
            return False
    else:
        print(f"folder alrealy exists: {folder_path}")
        return True


def info_file_save(info, dataset_name: str) -> bool:
    infor_path = "S3AVND/Result/info"
    create_folder(folder_name=infor_path)
    base_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    stat_file = 'information-' + dataset_name + "-" + time.strftime('%m%d-%H%M%S', time.localtime())+ '.txt'
    info.output_info_file_name = stat_file
    result_stat_file = open(os.path.join(base_path, infor_path, stat_file), 'w')
    result_stat_file.write(info.get_S3AND_result())
    result_stat_file.close()
    print(info.output_info_file_name, "saved successfully!")
    return True

def load_index_from_json(filename):
    with open(filename, 'r') as file:
        index = json.load(file)
    return index

def pruning_keyword_intersect(entry, query_bv, m) -> bool:
    """Return True (keep) if node keywords intersect with query keywords (non-empty intersection)."""
    node_bv = entry['Aux']['BV']
    for i in range(m):
        if int(node_bv[i]) & int(query_bv[i]) != 0:
            return True
    return False

def pruning_LB_VAND_phase_1(entry, q_j_kw_size, N_q_j, sigma, lam=0.5) -> bool:
    """V&ND lower bound phase 1: lb = λ·max(0,|q.W|-|v.W|) + (1-λ)·max(0,|N(q)|-|N(v)|)"""
    node_kw = entry["Aux"].get("kw", None)
    if node_kw is None:
        # kw missing from index: compute from BV (count set bits)
        node_kw = sum(bin(int(bv)).count('1') for bv in entry["Aux"]["BV"])
    node_nc = entry["Aux"].get("nc", None)
    if node_nc is None:
        # nc missing: use N_q_j as fallback so lb_nd=0 (safe, no false pruning)
        node_nc = N_q_j
    lb_kw = max(0, q_j_kw_size - node_kw)
    lb_nd = max(0, N_q_j - node_nc)
    lb_VAND = lam * lb_kw + (1 - lam) * lb_nd
    return lb_VAND <= sigma

def pruning_LB_VAND_phase_2(entry, q_j_kw_size, N_q_j, sigma, m, N_q, G_q: ig.Graph, lam=0.5) -> bool:
    """V&ND lower bound phase 2: tighter neighbor bound using NBV intersection check."""
    node_kw = entry["Aux"].get("kw", None)
    if node_kw is None:
        node_kw = sum(bin(int(bv)).count('1') for bv in entry["Aux"]["BV"])
    lb_kw = max(0, q_j_kw_size - node_kw)

    lb_nd = N_q_j
    for nq in N_q:
        bv = [int(k) for k in G_q.vs[nq]["BV"].split(",")]
        for i in range(m):
            if bv[i] & int(entry["Aux"]["NBV"][i]) != 0:
                lb_nd -= 1
                break

    lb_VAND = lam * lb_kw + (1 - lam) * max(0, lb_nd)
    return lb_VAND <= sigma

from typing import List, Dict, Set, Tuple

def AVND(q: ig.Graph, G: ig.Graph, matching: dict, sigmaAttr: str, lam: float = 0.5) -> float:
    """Compute AV&ND(q, g) = f({V&ND(q_j, v_i) | M(q_j) = v_i})
    V&ND(q_j, v_i) = λ|q_j.W - v_i.W| + (1-λ)|π(N(q_j)) - N(v_i)|
    """
    res = 0.0
    subgraph = G.subgraph(list(matching.values()))
    for q_node_id, v_node_id in matching.items():
        q_node = q.vs.find(id=q_node_id)
        q_kw = set(int(k) for k in q_node["keywords"].split(","))
        v_node = G.vs.find(id=v_node_id)
        v_kw = set(int(k) for k in v_node["keywords"].split(","))
        kw_diff = len(q_kw - v_kw)

        q_nbrs = set(q.neighbors(q_node))
        v_sub = subgraph.vs.find(id=v_node_id)
        v_nbrs = set(subgraph.neighbors(v_sub))
        nd_diff = len(q_nbrs - v_nbrs)

        vnd = lam * kw_diff + (1 - lam) * nd_diff
        if sigmaAttr == "SUM":
            res += vnd
        elif sigmaAttr == "MAX":
            res = max(res, vnd)
    return res



def select_query_order_with_pivot(query_graph: ig.Graph, V_cand: Dict[str, List[str]]) -> Tuple[List[str], List[str]]:
    """
    根据候选集大小和邻居关系选择查询顶点的遍历顺序，并生成 pivot 顶点。

    参数:
        query_graph (nx.Graph): 查询图 \( q \)。
        V_cand (Dict[str, List[str]]): 每个查询顶点对应的候选顶点集合。

    返回:
        Tuple[List[str], List[str]]: 查询顶点的遍历顺序和对应的 pivot 顶点。
    """
    # 初始化有序列表 Q 和 pivot 列表
    Q = []
    pivot = []
    # 未处理的查询顶点集合
    unprocessed = set(int(node["id"]) for node in query_graph.vs())

    # 第一步：选择候选集最小的查询顶点作为起点
    start_vertex = min(unprocessed, key=lambda v: len(V_cand[v]))
    Q.append(start_vertex)
    pivot.append(None)  # 第一个顶点没有 pivot
    unprocessed.remove(start_vertex)

    # 第二步：迭代选择邻居
    while unprocessed:
        # 找到 Q 中顶点的所有未处理的邻居
        neighbors = set()
        for q in Q:
            targetnode = query_graph.vs.find(id=q)
            # print(targetnode)
            # print(query_graph.neighbors(targetnode))
            neighbors.update(int(query_graph.vs[node]["id"]) for node in query_graph.neighbors(targetnode))
        
        neighbors.intersection_update(unprocessed)

        # print(neighbors)
        # 选择候选集最小的邻居
        next_vertex = min(neighbors, key=lambda v: len(V_cand[v]))

        # print(Q, neighbors)
        # 找到 next_vertex 在 Q 中的第一个邻居作为 pivot
        for q in Q:
            # print(q, next_vertex)
            # print(query_graph.get_eid(q, next_vertex))
            try:
                tnode1 = query_graph.vs.find(id=q)
                tnode2 = query_graph.vs.find(id=next_vertex)
                query_graph.get_eid(tnode1, tnode2)
                pivot.append(q)
                # print(pivot)
                break
            except Exception as e:
                # print("no such egde")
                continue
            # if query_graph.get_eid(q, next_vertex) != -1:
            # if eid:
                

        Q.append(next_vertex)
        unprocessed.remove(next_vertex)

    return Q, pivot

def select_query_order(query_graph: ig.Graph, V_cand: Dict[str, List[str]]) ->List[str]:
    # 初始化有序列表 Q 和 pivot 列表
    Q = []
    # 未处理的查询顶点集合
    unprocessed = set(query_graph.nodes())

    # 第一步：选择候选集最小的查询顶点作为起点
    start_vertex = min(unprocessed, key=lambda v: len(V_cand[v]))
    Q.append(start_vertex)
    unprocessed.remove(start_vertex)

    # 第二步：迭代选择邻居
    while unprocessed:
        # 找到 Q 中顶点的所有未处理的邻居
        neighbors = set()
        for q in Q:
            neighbors.update(query_graph.neighbors(q))
        neighbors.intersection_update(unprocessed)
        # 选择候选集最小的邻居
        next_vertex = min(neighbors, key=lambda v: len(V_cand[v]))
        Q.append(next_vertex)
        unprocessed.remove(next_vertex)

    return Q
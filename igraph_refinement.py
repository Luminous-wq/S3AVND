import igraph as ig
from typing import List, Dict, Set, Tuple, Optional

def generate_bn(query_graph: ig.Graph, order: List[str], pivot: List[str]) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
    """
    Generate neighbor information for each query vertex.

    Parameters:
        query_graph (nx.Graph): The query graph \(q\).
        order (List[str]): The matching order.
        pivot (List[str]): The pivot vertices.

    Returns:
        bn (Dict[str, List[str]]): Neighbor information for each query vertex.
        bn_count (Dict[str, int]): Number of neighbors for each query vertex.
    """
    query_vertices = [int(node["id"]) for node in query_graph.vs()]
    bn_count = {v: 0 for v in query_vertices}
    bn = {v: [] for v in query_vertices}

    visited_vertices = set()
    visited_vertices.add(order[0])
    # print(order)
    # print(pivot)
    for i in range(1, len(order)):
        vertex = order[i]
        # print(vertex)
        tnode = query_graph.vs.find(id=vertex)
        nbrs = list(int(query_graph.vs[node]["id"]) for node in query_graph.neighbors(tnode))
        for nbr in nbrs:
            if nbr in visited_vertices and nbr != pivot[i]:
                bn[vertex].append(nbr)
                bn_count[vertex] += 1
        visited_vertices.add(vertex)

    return bn, bn_count

def get_optimal_order(query_graph, query_plan):
    # query_plan[u] 是候选列表，我们按长度排序
    # 找到候选集最小的作为起点
    start_node = min(query_plan.keys(), key=lambda u: len(query_plan[u]))
    
    order = [start_node]
    visited = {start_node}
    
    while len(order) < query_graph.vcount():
        # 寻找与已在 order 中的节点相连的、且候选集最小的下一个节点
        best_next = None
        min_candidates = float('inf')
        
        for u in visited:
            for nbr in query_graph.neighbors(u):
                u_nbr = query_graph.vs[nbr]["id"] # 假设 query_plan 键是 id
                if u_nbr not in visited:
                    cand_len = len(query_plan[u_nbr])
                    if cand_len < min_candidates:
                        min_candidates = cand_len
                        best_next = u_nbr
        
        if best_next:
            order.append(best_next)
            visited.add(best_next)
        else:
            # 如果不连通（虽然你的情况一般是连通的），从未访问节点里找最小的
            remaining = set(query_plan.keys()) - visited
            next_node = min(remaining, key=lambda u: len(query_plan[u]))
            order.append(next_node)
            visited.add(next_node)
            
    return order

# def generate_valid_candidates(query_graph: ig.Graph, data_graph: ig.Graph, depth: int, embedding: Dict[str, str],
#                               idx_count: Dict[str, int], valid_candidate: Dict[str, List[str]], visited_vertices: Set[str],
#                               bn: Dict[str, List[str]], bn_count: Dict[str, int], order: List[str], pivot: List[str], query_plan) -> None:
#     """
#     Generate valid candidates for the current depth.

#     Parameters:
#         query_graph (ig.Graph): The query graph \(q\).
#         data_graph (ig.Graph): The data graph \(G\).
#         depth (int): The current recursion depth.
#         embedding (Dict[str, str]): The current matching \(M\).
#         idx_count (Dict[str, int]): The number of valid candidates for each depth.
#         valid_candidate (Dict[str, List[str]]): The valid candidates for each depth.
#         visited_vertices (Set[str]): Whether a vertex in the data graph has been visited.
#         bn (Dict[str, List[str]]): Neighbor information for each query vertex.
#         bn_count (Dict[str, int]): Number of neighbors for each query vertex.
#         order (List[str]): The matching order.
#         pivot (List[str]): The pivot vertices.
#     """
#     u = order[depth]
#     p = embedding[pivot[depth]]
#     # nbrs = list(data_graph.neighbors(p))

#     idx_count[u] = 0
#     valid_candidate[u] = []

#     for v in query_plan[u]:
#         if v not in embedding.values():  # Ensure v is not already in the current matching M
#             if v not in visited_vertices:  # Ensure v is not visited
#                 valid = True
#                 for u_nbr in bn[u]:
#                     u_nbr_v = embedding[u_nbr]
#                     try:
#                         # 获取边的ID，如果不存在则返回-1
#                         eid = data_graph.get_eid(v, u_nbr_v, directed=False)  # directed=False表示无向图
#                         if eid == -1:
#                             valid = False
#                             break
#                     except Exception as e:
#                         valid = False
#                         break
#                 if valid:
#                     valid_candidate[u].append(v)
#                     idx_count[u] += 1

def CCSandR_optimized(data_graph, query_graph, V_cand, subgraphs, matching, depth, threshold, 
                      order, bn, data_kw_cache, data_nbr_cache, query_kw_cache, info):
    # 这里一共有 13 个参数
    
    # 1. 达到搜索深度，保存结果
    if depth == len(order):
        info.candidate_subgraphs += 1
        # 调用极速约束检查
        if AVND_constraint_fast(query_graph, data_graph, matching, threshold, 
                                 data_kw_cache, data_nbr_cache, query_kw_cache, info):
            subgraphs.add(tuple(sorted(matching.values())))
        return

    u = order[depth]
    matched_v_set = set(matching.values())

    # 2. 遍历候选集
    for v in V_cand[u]:
        # 唯一性剪枝
        if v in matched_v_set:
            continue
            
        # 结构约束剪枝 (BN 检查)
        is_structure_valid = True
        for u_prev in bn[u]:
            v_prev = matching[u_prev]
            if v_prev not in data_nbr_cache[v]:
                is_structure_valid = False
                break
        
        if not is_structure_valid:
            continue

        # 3. 递归
        matching[u] = v
        CCSandR_optimized(data_graph, query_graph, V_cand, subgraphs, matching, depth + 1, 
                          threshold, order, bn, data_kw_cache, data_nbr_cache, 
                          query_kw_cache, info)
        
        # 4. 回溯
        del matching[u]

def AVND_constraint_fast(query_graph, G, matching, sigma, data_kw_cache, data_nbr_cache, query_kw_cache, info):
    total = 0.0
    lam = info.lam
    f_type = info.function

    for q_idx, v_idx in matching.items():
        q_kw = query_kw_cache[q_idx]
        v_kw = data_kw_cache[v_idx]

        # 关键词交集必须存在
        if not (q_kw & v_kw): return False

        # 计算差异
        kw_diff = len(q_kw - v_kw)
        nd_diff = 0
        for q_nbr in query_graph.neighbors(q_idx):
            if q_nbr in matching:
                if matching[q_nbr] not in data_nbr_cache[v_idx]:
                    nd_diff += 1

        vnd = lam * kw_diff + (1 - lam) * nd_diff
        
        if f_type == "SUM":
            total += vnd
        else:
            total = max(total, vnd)
            
        if total > sigma: 
            # print("over AVND score:{}, sigma: {}".format(total, sigma))
            return False
    
    # print("AVND score:{}, sigma: {}".format(total, sigma))

    return True, total

# def CCSandR(data_graph: ig.Graph, query_graph: ig.Graph, query_plan: Dict[str, List[str]], subgraphs: Set[tuple],
#             matching: Dict[str, Optional[str]], depth: int, threshold: float, output_limit_num: int,
#             bn: Dict[str, List[str]], bn_count: Dict[str, int], order: List[str], pivot: List[str], f:str, info) -> None:
#     if depth == len(order):
#         info.candidate_subgraphs += 1
#         if AVND_constraint(query_graph, data_graph, matching, threshold, f, lam=info.lam):
#             subgraphs.add(tuple(matching.values()))
#         return

#     u = order[depth]
#     idx_count = {u: 0}
#     valid_candidate = {u: []}
#     visited_vertices = set(matching.values())

#     if depth == 0:
#         valid_candidate[u] = query_plan[u].copy()
#         idx_count[u] = len(valid_candidate[u])
#     else:
#         generate_valid_candidates(query_graph, data_graph, depth, matching, idx_count, valid_candidate,
#                                   visited_vertices, bn, bn_count, order, pivot, query_plan=query_plan)

#     for v in valid_candidate[u]:
#         matching[u] = v
#         CCSandR(data_graph, query_graph, query_plan, subgraphs, matching, depth + 1, threshold, output_limit_num,
#                 bn, bn_count, order, pivot, f=f, info=info)
#         matching[u] = None
def AVND_constraint_optimized(query_graph, data_graph, matching, threshold, f, lam, 
                             data_kw_cache, data_nbr_cache, query_kw_cache):
    total = 0.0
    # matching: {q_idx: v_idx}
    for q_idx, v_idx in matching.items():
        q_kw = query_kw_cache[q_idx]
        v_kw = data_kw_cache[v_idx]

        # 1. 关键词交集必须非空
        if not (q_kw & v_kw): return False

        # 2. 计算 kw_diff
        kw_diff = len(q_kw - v_kw)

        # 3. 计算 nd_diff
        # 优化：通过已匹配的邻居在 data_nbr_cache 中查找是否存在
        q_nbrs = query_graph.neighbors(q_idx)
        nd_diff = 0
        for q_nbr in q_nbrs:
            if q_nbr in matching:
                # 检查匹配到的 data 节点之间是否存在边
                if matching[q_nbr] not in data_nbr_cache[v_idx]:
                    nd_diff += 1

        vnd = lam * kw_diff + (1 - lam) * nd_diff
        if f == "SUM":
            total += vnd
        else: # MAX
            total = max(total, vnd)
        
        if total > threshold: return False

    return True

# def AVND_constraint_optimized(query_graph, G, matching, q_kw_sets, G_kw_sets, sigma, f, lam=0.5):
#     """
#     q_kw_sets: 预处理好的查询图关键词 set 列表 [set(1,2), set(3,4), ...]
#     G_kw_sets: 预处理好的数据图关键词 set 列表
#     """
#     total = 0.0
    
#     # 1. 连通性建议在外部或使用增量方式检查，或者仅在完整匹配后检查
#     # 如果必须在这里查，建议直接判断 matching 里的节点在 G 中是否能构成连通分量

#     for q_idx, v_idx in matching.items():
#         # 获取预处理后的 set
#         q_kw = q_kw_sets[q_idx]
#         v_kw = G_kw_sets[v_idx]

#         # 关键词交集必须非空 (BV 剪枝的精炼版)
#         if not (q_kw & v_kw):
#             return False

#         # 属性差异
#         kw_diff = len(q_kw - v_kw)

#         # 结构差异 (修正后的逻辑)
#         # 计算 q_idx 的邻居中，有多少个已经匹配的节点在数据图中不是 v_idx 的邻居
#         missing_edges = 0
#         q_neighbors = query_graph.neighbors(q_idx)
#         for q_nbr in q_neighbors:
#             if q_nbr in matching:
#                 v_matched_nbr = matching[q_nbr]
#                 # 直接在原图 G 中查，are_connected 是 O(1) 或 O(degree) 的，非常快
#                 if not G.are_connected(v_idx, v_matched_nbr):
#                     missing_edges += 1
#             else:
#                 # 如果这个邻居还没匹配，根据定义，它也属于“当前尚未覆盖的结构差异”
#                 missing_edges += 1

#         vnd = lam * kw_diff + (1 - lam) * missing_edges
        
#         if f == "SUM":
#             total += vnd
#         elif f == "MAX":
#             total = max(total, vnd)
        
#         if total > sigma:
#             return False

#     return True

# def AVND_constraint(query_graph: ig.Graph, G: ig.Graph, matching: dict, sigma, f: str, lam: float = 0.5) -> bool:
#     """Check AV&ND(q,g) <= sigma with keyword intersection requirement.
#     V&ND(q_j, v_i) = λ|q_j.W - v_i.W| + (1-λ)|π(N(q_j)) - N(v_i)|
#     π maps each query neighbor through the matching to its data-graph ID.
#     """
#     subgraph = G.subgraph(list(matching.values()))
#     if not subgraph.is_connected():
#         return False

#     total = 0.0
#     for q_node_id, v_node_id in matching.items():
#         q_node = query_graph.vs.find(id=q_node_id)
#         q_kw = set(int(k) for k in q_node["keywords"].split(","))
#         v_node = G.vs.find(id=v_node_id)
#         v_kw = set(int(k) for k in v_node["keywords"].split(","))

#         # Keyword intersection must be non-empty
#         if not (q_kw & v_kw):
#             return False

#         kw_diff = len(q_kw - v_kw)

#         # nd_diff: map query neighbors through matching, compare with actual data neighbors
#         q_nbr_ids = {int(query_graph.vs[n]["id"]) for n in query_graph.neighbors(q_node)}
#         mapped_data_ids = {matching[qid] for qid in q_nbr_ids if qid in matching}
#         v_actual_nbr_ids = {int(G.vs[n]["id"]) for n in G.neighbors(v_node)}
#         nd_diff = len(mapped_data_ids - v_actual_nbr_ids)

#         vnd = lam * kw_diff + (1 - lam) * nd_diff
#         if f == "SUM":
#             total += vnd
#         elif f == "MAX":
#             total = max(total, vnd)
#         if total > sigma:
#             return False

#     return total <= sigma


# Example usage
if __name__ == "__main__":
    data_graph = ig.Graph()
    data_graph.add_edges_from([("Ben_Luz", "David_A._Christian"), ("male", "Roland_Schwegler"),
                               ("male", "Sebastián_Hernández"), ("male", "David_A._Christian")])
    # data_graph.add_node("Ben_Luz")
    data_graph.nodes["Ben_Luz"]["label"] = "person"
    data_graph.nodes["male"]["label"] = "gender"
    data_graph.nodes["Roland_Schwegler"]["label"] = "person"
    data_graph.nodes["Sebastián_Hernández"]["label"] = "person"
    data_graph.nodes["David_A._Christian"]["label"] = "person"

    # Define query graph
    query_graph = ig.Graph()
    query_graph.add_edges_from([("u1", "u2"), ("u2", "u3")])
    query_graph.nodes["u1"]["label"] = "person"
    query_graph.nodes["u2"]["label"] = "gender"
    query_graph.nodes["u3"]["label"] = "person"

    # Define query plan (candidate vertices for each query vertex)
    query_plan = {
        "u1": ["Roland_Schwegler", "Ben_Luz", "Sebastián_Hernández", "David_A._Christian"],  # Candidates for query vertex u1
        "u2": ["male"],     # Candidates for query vertex u2
        "u3": ["Roland_Schwegler",  "Ben_Luz", "Sebastián_Hernández", "David_A._Christian"]  # Candidates for u3
    }

    # Initialize variables
    subgraphs = set()
    matching = {u: None for u in query_plan}  # Initialize matching as a dictionary
    threshold = 1.0
    output_limit_num = 50

    # Generate matching order and pivot vertices
    order = ["u2", "u3", "u1"]
    pivot = ["u2", "u2", "u3"]

    # Generate neighbor information
    bn, bn_count = generate_bn(query_graph, order, pivot)

    # Run CCSandR algorithm
    CCSandR_optimized(data_graph, query_graph, query_plan, subgraphs, matching, 0, threshold, output_limit_num, bn, bn_count, order, pivot, f="MAX")

    # Print results
    print("Valid subgraphs:")
    for subgraph in subgraphs:
        print(subgraph)
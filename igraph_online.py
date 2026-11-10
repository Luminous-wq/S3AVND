from infomation import Info
import igraph as ig
import heapq
import time
from igraph_tools import (pruning_keyword_intersect, pruning_LB_VAND_phase_1,
                          pruning_LB_VAND_phase_2, select_query_order_with_pivot)
from igraph_refinement import generate_bn, CCSandR_optimized
import math

class IndexEntry:
    def __init__(self, key, index_N, idx):
        self.key = key
        self.index_N = index_N
        self.idx = idx

    def __gt__(self, other):
        if self.index_N["L"] == other.index_N["L"]:
            return self.key > other.key
        else:
            return self.index_N["L"] < other.index_N["L"]
def S3AVND(Graph: ig.Graph, G_q: ig.Graph, sigma: float, sigmaAttr: str, 
           root: list, info: Info) -> list:
    
    start_time = time.time()
    
    # --- 1. 预处理查询图缓存 ---
    q_nodes = []
    q_nodes_bv = []
    q_neighbor = []
    q_neighbor_size = []
    q_kw_sizes = []
    query_kw_cache = {}
    
    # 建立 ID 到 索引 的快速映射，消除 .index()
    q_id_to_idx = {}
    
    for idx, node in enumerate(G_q.vs):
        u_id = int(node["id"])
        q_nodes.append(u_id)
        q_id_to_idx[u_id] = idx
        
        q_nodes_bv.append([int(k) for k in node["BV"].split(",")])
        nbrs = G_q.neighbors(node)
        q_neighbor.append(nbrs)
        q_neighbor_size.append(len(nbrs))
        
        kw = [int(k) for k in node["keywords"].split(",")]
        q_kw_sizes.append(len(kw))
        query_kw_cache[idx] = set(kw)

    lam = info.lam
    q_size = len(q_nodes)
    info.initialization_time += (time.time() - start_time)

    # --- 2. 预处理数据图缓存 (用于后续 Refine 阶段) ---
    # 这一步建议放在主循环外，如果多次查询，只做一次
    data_kw_cache = [set(int(k) for k in v.split(",")) if v else set() for v in Graph.vs["keywords"]]
    data_nbr_cache = [set(Graph.neighbors(v)) for v in range(Graph.vcount())]

    # --- 3. 树索引过滤阶段 ---
    V_cand = {q_node: [] for q_node in q_nodes}
    H = []

    # 初始化根节点
    for idx, entry in enumerate(root):
        info.entry_node_visit_counter += 1
        entry["Q"] = []
        for q_j_id in q_nodes:
            j = q_id_to_idx[q_j_id]
            if pruning_keyword_intersect(entry, q_nodes_bv[j], info.group_number):
                if pruning_LB_VAND_phase_1(entry, q_kw_sizes[j], q_neighbor_size[j], sigma, lam):
                    if pruning_LB_VAND_phase_2(entry, q_kw_sizes[j], q_neighbor_size[j], sigma, 
                                               info.group_number, q_neighbor[j], G_q, lam):
                        entry["Q"].append(q_j_id)
                    else:
                            info.entry_pruning_counter_LB_ND_2 += 1
                else:
                    info.entry_pruning_counter_LB_ND_1 += 1
            else:
                info.entry_pruning_counter_BV += 1
        
        if entry["Q"]:
            # 使用 nk 作为优先级，heappush 维持堆序
            heapq.heappush(H, IndexEntry(key=entry["Aux"]["nk"], index_N=entry, idx=idx))
        else:
            info.entry_pruning_counter += 1

    # 堆处理循环
    while H:
        now_entry_H = heapq.heappop(H)
        now_entry = now_entry_H.index_N
        # 注意：此处千万不要手动 heapify(H)

        if now_entry["T"]: # 处理叶子节点 (Vertex)
            leaf_start = time.time()
            # 优化：只检查在父节点中未被剪枝的查询节点
            for q_j_id in now_entry["Q"]:
                j = q_id_to_idx[q_j_id]
                
                if pruning_keyword_intersect(now_entry, q_nodes_bv[j], info.group_number):
                    if pruning_LB_VAND_phase_1(now_entry, q_kw_sizes[j], q_neighbor_size[j], sigma, lam):
                        if pruning_LB_VAND_phase_2(now_entry, q_kw_sizes[j], q_neighbor_size[j], sigma,
                                                   info.group_number, q_neighbor[j], G_q, lam):
                            V_cand[q_j_id].append(now_entry["P"])
                            info.vertex_visit_counter += 1
                        else:
                            info.vertex_pruning_counter_LB_ND_2 += 1
                    else:
                        info.vertex_pruning_counter_LB_ND_1 += 1
                else:
                    info.vertex_pruning_counter_BV += 1
            info.leaf_node_traverse_time += (time.time() - leaf_start)

        else: # 处理中间节点
            non_leaf_start = time.time()
            for child_entry in now_entry["P"]:
                info.entry_node_visit_counter += 1
                child_entry["Q"] = []
                # 继承父节点的存活查询节点进行进一步过滤
                for q_j_id in now_entry["Q"]:
                    j = q_id_to_idx[q_j_id]
                    if pruning_keyword_intersect(child_entry, q_nodes_bv[j], info.group_number):
                        if pruning_LB_VAND_phase_1(child_entry, q_kw_sizes[j], q_neighbor_size[j], sigma, lam):
                            if pruning_LB_VAND_phase_2(child_entry, q_kw_sizes[j], q_neighbor_size[j], sigma,
                                                       info.group_number, q_neighbor[j], G_q, lam):
                                child_entry["Q"].append(q_j_id)
                            else:
                                info.entry_pruning_counter_LB_ND_2 += 1
                        else:
                            info.entry_pruning_counter_LB_ND_1 += 1
                    else:
                        info.entry_pruning_counter_BV += 1
                
                if child_entry["Q"]:
                    heapq.heappush(H, IndexEntry(key=child_entry["Aux"]["nk"], index_N=child_entry, idx=None))
                else:
                    info.entry_pruning_counter += 1
            info.non_leaf_node_traverse_time += (time.time() - non_leaf_start)

    # --- 4. 细化匹配阶段 (Refine) ---
    cand_lengths = {key: len(value) for key, value in V_cand.items()}
    print(f"Vertex candidate lengths: {cand_lengths}")
    
    id_to_idx = {int(n["id"]): n.index for n in G_q.vs}

    # 创建一个新的 V_cand，使用内部索引作为键
    V_cand_idx = {id_to_idx[int(q_id)]: v_list for q_id, v_list in V_cand.items()}

    pw = (int(info.nodes_num) - max(cand_lengths.values())) / int(info.nodes_num)
    print("pruning power: {}".format(pw))
    print("vertex_pruning_counter_BV: {}".format(info.vertex_pruning_counter_BV))
    print("vertex_pruning_counter_LB_ND_1: {}".format(info.vertex_pruning_counter_LB_ND_1))
    print("vertex_pruning_counter_LB_ND_2: {}".format(info.vertex_pruning_counter_LB_ND_2))
    print("entry_pruning_counter_BV: {}".format(info.entry_pruning_counter_BV))
    print("entry_pruning_counter_LB_ND_1: {}".format(info.entry_pruning_counter_LB_ND_1))
    print("entry_pruning_counter_LB_ND_2: {}".format(info.entry_pruning_counter_LB_ND_2))
    info.pp += pw

    # 使用“最小候选集蔓延”排序算法
    order = select_optimal_order_spreading(G_q, V_cand_idx)
    
    # # 生成后向邻居约束 (BN)
    # bn = {}
    # for i, u_id in enumerate(order):
    #     u_idx_in_gq = G_q.vs.find(id=u_id).index
    #     u_nbrs_indices = set(G_q.neighbors(u_idx_in_gq))
    #     # 记录在 order 中排在自己前面且有边连接的查询节点
    #     bn[u_id] = [prev_u for prev_u in order[:i] 
    #                if G_q.vs.find(id=prev_u).index in u_nbrs_indices]

    S = set()
    # 初始化 matching 也使用内部索引
    matching = {u_idx: None for u_idx in V_cand_idx.keys()} 

    # 生成 BN 时，因为 order 已经是索引了，代码会简单很多
    bn = {}
    for i, u_idx in enumerate(order):
        u_nbrs = set(G_q.neighbors(u_idx))
        bn[u_idx] = [prev_u for prev_u in order[:i] if prev_u in u_nbrs]

    refine_start = time.time()
    info.candidate_subgraphs = 0
    # def CCSandR_optimized(data_graph, query_graph, V_cand, subgraphs, matching, depth, threshold, 
                    #   order, bn, data_kw_cache, data_nbr_cache, query_kw_cache, info):
    # 这里的 V_cand 已经填充了数据图节点的 index
    CCSandR_optimized(
    Graph, G_q, V_cand_idx, S, matching, 0, sigma, 
        order, bn, data_kw_cache, data_nbr_cache, query_kw_cache, info)
    info.refine_time = (time.time() - refine_start)
    print(f"Final Answers: {len(S)}")
    return S

# def S3AVND(
#         Graph: ig.Graph,
#         G_q: ig.Graph,
#         sigma: float,
#         sigmaAttr: str,
#         root: list,
#         info: Info) -> list:

#     initiaize_time = time.time()
#     q_nodes = []
#     q_neighbor = []
#     q_neighbor_size = []
#     q_nodes_bv = []
#     q_kw_sizes = []

#     for node in G_q.vs():
#         q_nodes.append(int(node["id"]))
#         q_bv = [int(k) for k in node["BV"].split(",")]
#         q_nodes_bv.append(q_bv)
#         q_N = list(G_q.neighbors(node))
#         q_neighbor.append(q_N)
#         q_neighbor_size.append(len(q_N))
#         kw = [int(k) for k in node["keywords"].split(",")]
#         q_kw_sizes.append(len(kw))

#     q_size = len(q_nodes)
#     lam = info.lam
#     print("****************************************************")
#     info.initialization_time += (time.time() - initiaize_time)

#     V_cand = {q_node: [] for q_node in q_nodes}
#     H = []
#     root_Q = q_nodes

#     for idx, entry in enumerate(root):
#         info.entry_node_visit_counter += 1
#         entry["Q"] = []
#         for q_j in root_Q:
#             j = q_nodes.index(q_j)
#             if pruning_keyword_intersect(entry=entry, query_bv=q_nodes_bv[j], m=info.group_number):
#                 if pruning_LB_VAND_phase_1(entry=entry, q_j_kw_size=q_kw_sizes[j],
#                                            N_q_j=q_neighbor_size[j], sigma=sigma, lam=lam):
#                     if pruning_LB_VAND_phase_2(entry=entry, q_j_kw_size=q_kw_sizes[j],
#                                                N_q_j=q_neighbor_size[j], sigma=sigma,
#                                                m=info.group_number, N_q=q_neighbor[j], G_q=G_q, lam=lam):
#                         entry["Q"].append(q_j)
#         if entry["Q"]:
#             heapq.heappush(H, IndexEntry(key=entry["Aux"]["nk"], index_N=entry, idx=idx))
#         else:
#             info.entry_pruning_counter += 1

#     while len(H) > 0:
#         now_entry_H = heapq.heappop(H)
#         now_entry = now_entry_H.index_N
#         heapq.heapify(H)

#         if now_entry["T"]:
#             leaf_node_start_time = time.time()
#             v_bv = now_entry["Aux"].get('BV', [])
#             IND = []
#             for i in range(q_size):
#                 query_BV = q_nodes_bv[i]
#                 for j in range(info.group_number):
#                     if int(query_BV[j]) & int(v_bv[j]) != 0:
#                         IND.append(i)
#                         break

#             for node_index in IND:
#                 if pruning_keyword_intersect(entry=now_entry, query_bv=q_nodes_bv[node_index], m=info.group_number):
#                     # info.vertex_pruning_counter_BV += 1
#                     if pruning_LB_VAND_phase_1(entry=now_entry, q_j_kw_size=q_kw_sizes[node_index],
#                                                N_q_j=q_neighbor_size[node_index], sigma=sigma, lam=lam):
#                         # info.vertex_pruning_counter_LB_ND_1 += 1
#                         if pruning_LB_VAND_phase_2(entry=now_entry, q_j_kw_size=q_kw_sizes[node_index],
#                                                    N_q_j=q_neighbor_size[node_index], sigma=sigma,
#                                                    m=info.group_number, N_q=q_neighbor[node_index],
#                                                    G_q=G_q, lam=lam):
#                             V_cand[q_nodes[node_index]].append(now_entry["P"])
#                             info.vertex_visit_counter += 1
#                             # info.vertex_pruning_counter_LB_ND_2 += 1
#                         else:
#                             info.vertex_pruning_counter_LB_ND_2 += 1
#                     else:
#                         info.vertex_pruning_counter_LB_ND_1 += 1
#                 else:
#                     info.vertex_pruning_counter_BV += 1

#             info.leaf_node_traverse_time += (time.time() - leaf_node_start_time)
#         else:
#             info.entry_node_visit_counter += 1
#             non_leaf_node_start_time = time.time()
#             for child_entry in now_entry["P"]:
#                 child_entry["Q"] = []
#                 for q_j in now_entry["Q"]:
#                     j = q_nodes.index(q_j)
#                     if pruning_keyword_intersect(entry=child_entry, query_bv=q_nodes_bv[j], m=info.group_number):
#                         if pruning_LB_VAND_phase_1(entry=child_entry, q_j_kw_size=q_kw_sizes[j],
#                                                    N_q_j=q_neighbor_size[j], sigma=sigma, lam=lam):
#                             if pruning_LB_VAND_phase_2(entry=child_entry, q_j_kw_size=q_kw_sizes[j],
#                                                        N_q_j=q_neighbor_size[j], sigma=sigma,
#                                                        m=info.group_number, N_q=q_neighbor[j],
#                                                        G_q=G_q, lam=lam):
#                                 child_entry["Q"].append(q_j)
#                 if child_entry["Q"]:
#                     heapq.heappush(H, IndexEntry(key=child_entry["Aux"]["nk"], index_N=child_entry, idx=None))
#                 else:
#                     info.entry_pruning_counter += 1
#             info.non_leaf_node_traverse_time += (time.time() - non_leaf_node_start_time)

#     print("vertex_pruning_counter_BV: {}".format(info.vertex_pruning_counter_BV))
#     cand_lengths = {key: len(value) for key, value in V_cand.items()}
#     print("vertex candidate length: {}".format(cand_lengths))
#     # if math.prod(cand_lengths.values()) > 10**5:
#     #     return []
#     pw = (int(info.nodes_num) - max(cand_lengths.values())) / int(info.nodes_num)
#     print("pruning power: {}".format(pw))

#     order, pivot = select_query_order_with_pivot(query_graph=G_q, V_cand=V_cand)

#     S = set()
#     matching = {u: None for u in V_cand}
#     threshold = info.thresholds_max
#     output_limit_num = 5000

#     bn, bn_count = generate_bn(G_q, order, pivot)
#     refine_time = time.time()
#     info.candidate_subgraphs = 0
#     CCSandR_optimized(Graph, G_q, V_cand, S, matching, 0, threshold, output_limit_num, bn, bn_count, order,
#             pivot, info=info)
#     info.refine_time = (time.time() - refine_time)

#     print("answer number: {}".format(len(S)))
#     print("candidate subgraphs: {}".format(info.candidate_subgraphs))
#     return S



def select_optimal_order_spreading(G_q, V_cand_idx):
    # 直接找候选集最小的起始索引
    start_u = min(V_cand_idx.keys(), key=lambda k: len(V_cand_idx[k]))
    order = [start_u]
    visited = {start_u}
    
    while len(order) < len(V_cand_idx):
        best_next = None
        min_size = float('inf')
        for u in visited:
            for nbr in G_q.neighbors(u): # 这里的 nbr 直接就是索引
                if nbr not in visited:
                    size = len(V_cand_idx[nbr])
                    if size < min_size:
                        min_size = size
                        best_next = nbr
        if best_next is not None:
            order.append(best_next)
            visited.add(best_next)
        else: # 处理不连通
            remaining = set(V_cand_idx.keys()) - visited
            next_u = min(remaining, key=lambda k: len(V_cand_idx[k]))
            order.append(next_u)
            visited.add(next_u)
    return order
import pandas as pd
import os
import time
from infomation import Info
import random
import json
import copy
import igraph as ig

def construct_index_balanced(Graph: ig.Graph, info: Info) -> list:
    print(f"Total nodes: {len(Graph.vs)}, Partition number (k): {info.partition_number}")
    
    # --- 性能优化：预处理所有属性为缓存 ---
    print("Pre-processing attributes into cache...")
    start_pre = time.time()
    
    # 一次性解析所有 BV 和 NBV 字符串为整数列表
    bv_cache = [[int(k) for k in v.split(",")] for v in Graph.vs["BV"]]
    nbv_cache = [[int(k) for k in v.split(",")] for v in Graph.vs["NBV"]]
    
    # 提取数值属性为列表
    nk_cache = Graph.vs["nk"]
    kw_cache = Graph.vs["kw"]
    nc_cache = Graph.vs["nc"]
    
    print(f"Pre-processing cost time: {time.time() - start_pre:.4f}s")
    # ------------------------------------

    nodes_list = [v.index for v in Graph.vs]
    start_time = time.time()
    
    # 将所有 cache 传给递归函数
    root_node = root_tree(nodes_list, info.partition_number, 0, Graph, info, 
                          bv_cache, nbv_cache, nk_cache, kw_cache, nc_cache)
    
    print(f"Tree index finished. Construct cost time: {time.time() - start_time:.4f}s")

    if len(info.dataset_name.split("-")) > 4:
        f_name = info.dataset_name.rsplit("-", 1)[0]
        d_name = info.dataset_name.rsplit("-", 1)[1]
        folder_name = os.path.join("redataset", "precompute", "synthetic", f_name)
        initial_directory = os.getcwd()
        os.chdir(folder_name)
        save_index_as_json(root_node, 'G+-'+d_name+'_index.json')
        os.chdir(initial_directory)
    else:
        folder_name = os.path.join("redataset", "precompute", "real", info.dataset_name)
        initial_directory = os.getcwd()
        os.chdir(folder_name)
        save_index_as_json(root_node, info.dataset_name+'_index.json')
        os.chdir(initial_directory)
        
    return root_node, Graph

# def construct_index_balanced(Graph: ig.Graph, info: Info) -> list:
#     print("Pre-processing node attributes...")
#     start_pre = time.time()
    
#     # 将 BV 和 NBV 预先解析为整数列表的列表
#     # 结果格式: [[int, int, int, int, int], [int, int, ...], ...]
#     bv_cache = [[int(k) for k in v.split(",")] for v in Graph.vs["BV"]]
#     nbv_cache = [[int(k) for k in v.split(",")] for v in Graph.vs["NBV"]]
    
#     # 其他数值属性直接取成列表，避免重复访问 Graph.vs
#     nk_cache = Graph.vs["nk"]
#     kw_cache = Graph.vs["kw"]
#     nc_cache = Graph.vs["nc"]
#     print(f"Pre-processing finished in {time.time()-start_pre:.2f}s")
#     nodes_list = [v.index for v in Graph.vs]
#     k = info.partition_number
#     print(k)
#     start_time = time.time()

#     root_node = root_tree(nodes_list, k, 0, Graph, info, bv_cache, nbv_cache, nk_cache, kw_cache, nc_cache)
#     root_node = root_tree(nodes_list=nodes_list, num_partition=k, level=0, Graph=Graph, info=info)
#     print("Tree index is finished")
#     print("Construct cost time:{}".format(time.time()-start_time))
#     if len(info.dataset_name.split("-")) > 4:
#         f_name = info.dataset_name.rsplit("-", 1)[0]
#         d_name = info.dataset_name.rsplit("-", 1)[1]
#         folder_name = os.path.join("redataset", "precompute", "synthetic", f_name)
#         initial_directory = os.getcwd()
#         os.chdir(folder_name)
#         save_index_as_json(root_node, 'G+-'+d_name+'_index.json')
#         os.chdir(initial_directory)
#     else:
#         folder_name = os.path.join("redataset", "precompute", "real", info.dataset_name)
#         initial_directory = os.getcwd()
#         os.chdir(folder_name)
#         save_index_as_json(root_node, info.dataset_name+'_index.json')
#         os.chdir(initial_directory)
#     # print_index_tree(root_node)
#     # level_counts = {}
#     # count_nodes_per_level(root_node, level_counts, 0)
#     # for level, count in level_counts.items():
#     #     print(f"Level {level}: {count} nodes")

#     return root_node, Graph

def root_tree(nodes_list, num_partition, level, Graph, info, bv_cache, nbv_cache, nk_cache, kw_cache, nc_cache):
    # 叶子节点处理
    if len(nodes_list) <= num_partition:
        return [
            {
                "P": node,
                "Aux": {
                    "BV": bv_cache[node],
                    "NBV": nbv_cache[node],
                    "nk": nk_cache[node],
                    "kw": kw_cache[node],
                    "nc": nc_cache[node]
                },
                "T": True,
                "L": level
            } for node in nodes_list
        ]

    # 1. 划分初始化
    partition_ans, centers_bv = initialize_partition(bv_cache, nodes_list, num_partition, info.group_number)
    
    # 2. 成本模型迭代优化
    final_partition, cost_centers_bv = cost_model(bv_cache, partition_ans, info.iteration_number, info.group_number, 
                                                 centers_bv, nodes_list, num_partition, info)
    
    aggregated_child_entry_list = []
    results = []

    # 3. 递归处理每个分区
    for i in range(1, num_partition + 1):
        partition_node_array = final_partition[i]
        if not partition_node_array: continue
        
        # 递归调用
        child_entry_list = root_tree(partition_node_array, num_partition, level + 1, Graph, info,
                                    bv_cache, nbv_cache, nk_cache, kw_cache, nc_cache)

        # 4. 聚合 Aux 信息 (Synopsis)
        aggregated_synopsis = {
            "BV": [0] * info.group_number,
            "NBV": [0] * info.group_number,
            "nk": 0, "kw": 0, "nc": 0
        }

        for child in child_entry_list:
            for g in range(info.group_number):
                aggregated_synopsis['BV'][g] |= child['Aux']['BV'][g]
                aggregated_synopsis['NBV'][g] |= child['Aux']['NBV'][g]
            
            aggregated_synopsis['nk'] = max(aggregated_synopsis['nk'], child['Aux']['nk'])
            aggregated_synopsis['kw'] = max(aggregated_synopsis['kw'], child['Aux']['kw'])
            aggregated_synopsis['nc'] = max(aggregated_synopsis['nc'], child['Aux']['nc'])

        results.append({
            "P": child_entry_list,
            "Aux": aggregated_synopsis,
            "T": False,
            "L": level
        })

    return results

# # def root_tree(nodes_list: list, num_partition: int, level: int, Graph: ig.Graph, info: Info):
#     # print("len nodes before if: {}, num : {}".format(len(nodes_list), num_partition))
#     if len(nodes_list) <= num_partition:
#         # print("len nodes in if: {}, num : {}".format(len(nodes_list), num_partition))
#         # leaf nodes are vertices
#         return [
#             {
#                 "P": node,
#                 "Aux": {
#                     "BV": [int(k) for k in Graph.vs[node]['BV'].split(",")],
#                     "NBV": [int(k) for k in Graph.vs[node]['NBV'].split(",")],
#                     "nk": Graph.vs[node]["nk"],
#                     "kw": Graph.vs[node]["kw"],
#                     "nc": Graph.vs[node]["nc"],
#                 },
#                 "T": True,
#                 "L": level
#             } for node in nodes_list
#         ]
        
#         # leaf nodes are vertex set
#         # aggregate_aux = {
#         # "BV": [0] * info.group_number,
#         # "NBV": [0] * info.group_number,
#         # "nk": 0
#         # }
#         # for node in nodes_list:
#         #     for group in range(info.group_number):
#         #         aggregate_aux['BV'][group] = aggregate_aux['BV'][group] | Graph.nodes[node]['Aux']['BV'][group]
#         #         aggregate_aux['NBV'][group] = aggregate_aux['NBV'][group] | Graph.nodes[node]['Aux']['NBV'][group]

#         #     if aggregate_aux['nk'] < Graph.nodes[node]['Aux']['nk']:
#         #         aggregate_aux['nk'] = Graph.nodes[node]['Aux']['nk']

#         # return [{
#         #     "P": nodes_list,
#         #     "Aux": aggregate_aux,
#         #     "T": True,
#         #     "L": level
#         # }]
    
#     partition_ans, centers_bv, Graph1 = initialize_partition(Graph=Graph, nodes_list=nodes_list, 
#                                                 num_partition=info.partition_number, m=info.group_number)
#     final_partition, _, Graph = cost_model(Graph=Graph1, partition=partition_ans, t=info.iteration_number, m=info.group_number,
#                                             centers_bv=centers_bv, nodes_list=nodes_list, num_partition=info.partition_number,
#                                             info=info)
    
#     aggregated_child_entry_list = []

#     aggregated_synopsis = {
#         "BV": [0] * info.group_number,
#         "NBV": [0] * info.group_number,
#         "nk": 0,
#         "kw": 0,
#         "nc": 0
#     }

#     # aggregated_synopsis = aggregate_aux
#     # aggregated_synopsis.append(aggregate_aux)

#     for i in range(1, num_partition+1):
#         partition_node_array = final_partition[i]
#         # if len(partition_node_array) < num_partition:
#         #     child_entry_list = [
#         #         {
#         #             "P": node,
#         #             "Aux": Graph.nodes[node]['Aux'],
#         #             "T": True,
#         #             "L": level + 1
#         #         } for node in partition_node_array
#         #     ]
#         # else:
#         child_entry_list = root_tree(partition_node_array, num_partition, level+1, Graph, info)

#         for child_entry in child_entry_list:
#             # child_entry['Aux']['BV'] = [k for k in child_entry['Aux']['BV'].split(",")]
#             # child_entry['Aux']['NBV'] = [k for k in child_entry['Aux']['NBV'].split(",")]
#             for group in range(info.group_number):
#                 # print(aggregated_synopsis)
#                 # print(aggregated_synopsis['BV'][group])
#                 # print(group)
#                 # print(child_entry)
#                 # print(child_entry['Aux'])
#                 aggregated_synopsis['BV'][group] = aggregated_synopsis['BV'][group] | int(child_entry['Aux']['BV'][group])
#                 aggregated_synopsis['NBV'][group] = aggregated_synopsis['NBV'][group] | int(child_entry['Aux']['NBV'][group])

#             if aggregated_synopsis['nk'] < child_entry['Aux']['nk']:
#                 aggregated_synopsis['nk'] = child_entry['Aux']['nk']
#             if aggregated_synopsis['kw'] < child_entry['Aux']['kw']:
#                 aggregated_synopsis['kw'] = child_entry['Aux']['kw']
#             if aggregated_synopsis['nc'] < child_entry['Aux']['nc']:
#                 aggregated_synopsis['nc'] = child_entry['Aux']['nc']
        
#         aggregated_child_entry_list.append(child_entry_list)

#     return [
#         {
#             "P": child_entry_list,
#             "Aux": aggregated_synopsis,
#             "T": False,
#             "L": level
#         } for child_entry_list in aggregated_child_entry_list
#     ]

def save_index_as_json(index, filename):
    with open(filename, 'w') as file:
        json.dump(index, file)


def load_index_from_json(filename):
    with open(filename, 'r') as file:
        index = json.load(file)
    return index


def print_index_tree(node, depth=0):
    if isinstance(node, list):
        for subnode in node:
            print_index_tree(subnode, depth)
    elif isinstance(node, dict):
        print("  " * depth + "Node:")
        print("  " * (depth + 1) + f"P: {node['P']}")
        print("  " * (depth + 1) + f"Aux: {node['Aux']}")
        print("  " * (depth + 1) + f"T: {node['T']}")
        if 'P' in node:
            print_index_tree(node['P'], depth + 1)


def count_nodes_per_level(root, level_counts=None, level=0):
    if level in level_counts:
        level_counts[level] += len(root)
    else:
        level_counts[level] = len(root)

    for entry in root:
        if not entry['T']:
            child_entry_list = entry['P']
            count_nodes_per_level(child_entry_list, level_counts, level + 1)

    return level_counts

def initialize_partition(bv_cache, nodes_list, num_partition, m):
    centers = random.sample(nodes_list, num_partition)
    mapping = {center: i for i, center in enumerate(centers, start=1)}
    partition = {i: [centers[i-1]] for i in range(1, num_partition + 1)}
    centers_bv = {mapping[c]: bv_cache[c] for c in centers}

    avg_size = len(nodes_list) / num_partition

    for node in nodes_list:
        if node in mapping: continue
        
        min_dist = float('inf')
        best_center = 1
        node_bv = bv_cache[node]

        for c_idx, c_bv in centers_bv.items():
            # 负载均衡限制
            if len(partition[c_idx]) > avg_size: continue
            
            # 高效位运算
            dist = 0
            for g in range(m):
                dist += bin(c_bv[g] ^ node_bv[g]).count('1')
            
            if dist < min_dist:
                min_dist = dist
                best_center = c_idx
                
        partition[best_center].append(node)
    
    return partition, centers_bv
# # def initialize_partition(Graph: ig.Graph, nodes_list: list, num_partition: int, m):
#     # print("len nodes: {}".format(len(nodes_list)))
#     centers = random.sample(nodes_list, num_partition)
    
#     # TODO: 把center的标签换成数字，比如[a,b,c,d,e,f]或者数字[1,2,3,4,5]
#     # 主要是为了不改变center，只改变它对应的keywords / m bit vectors

#     mapping = {}
#     for i, center in enumerate(centers, start=1):
#         mapping[center] = i
#     # print(centers)
#     partition = {mapping[center]: [center] for center in centers} # {1: [xx, xxx]; 2: [xxx, xxxx]}
#     # print(partition)
#     centers_bv = {}
#     for center in centers:
#         # print(center)
#         centers_bv[mapping[center]] = [int(k) for k in Graph.vs[center]["BV"].split(",")]
#         # Graph.vs[center]['partition'] = mapping[center]

#     for node in nodes_list:
#         if node in centers:
#             continue
#         min_distance = float('inf')
#         best_center = None

#         for center in centers:
#             # balanced limit
#             if len(partition[mapping[center]]) > len(nodes_list)/num_partition:
#                 continue
#             total_distance = 0
#             center_bv = centers_bv[mapping[center]]
#             node_bv = [int(k) for k in Graph.vs[node]["BV"].split(",")]
#             for group_index in range(m):
#                 # xor
#                 xor_result = int(center_bv[group_index]) ^ int(node_bv[group_index])
#                 distance = bin(xor_result).count('1')
#                 total_distance += distance
#             if total_distance < min_distance:
#                 min_distance = total_distance
#                 best_center = mapping[center] # 这是number，center的编号
#         partition[best_center].append(node)
#         # Graph.vs[node]['partition'] = best_center
#         if best_center == None:
#             print("oh no: {}".format(node))
#     # for center_index, nodes in partition.items():
#     #     for node in nodes:
#     #         Graph.nodes[node]['partition'] = center_index
#     # partition_sizes = {center_index: len(nodes) for center_index, nodes in partition.items()}
#     # print(partition_sizes)

#     return partition, centers_bv, Graph

def cost_model(bv_cache, partition, t, m, centers_bv, nodes_list, num_partition, info):
    current_partition = copy.deepcopy(partition)
    current_centers_bv = copy.deepcopy(centers_bv)
    current_cost = cost(bv_cache, current_partition, m, current_centers_bv)
    
    avg_size = len(nodes_list) / num_partition
    unchanged_rounds = 0

    for _ in range(t):
        # 1. 更新中心 (基于多数表决)
        new_centers_bv = {}
        for c_idx, members in current_partition.items():
            if not members:
                new_centers_bv[c_idx] = current_centers_bv[c_idx]
                continue
            
            group_results = []
            for g in range(m):
                # 统计该分区内该 group 下每一位的 '1' 出现次数
                # 这种方法对于超大整数比较稳妥
                member_bvs = [bv_cache[node][g] for node in members]
                combined_int = 0
                # 假设最大位宽，或者根据数据动态判断，这里假设最大 256 位
                for bit in range(256): 
                    one_count = sum((val >> bit) & 1 for val in member_bvs)
                    if one_count >= len(members) / 2:
                        combined_int |= (1 << bit)
                group_results.append(combined_int)
            new_centers_bv[c_idx] = group_results

        # 2. 重新分配节点
        new_partition = {i: [] for i in range(1, num_partition + 1)}
        for node in nodes_list:
            min_dist = float('inf')
            best_c = 1
            node_bv = bv_cache[node]
            for c_idx, c_bv in new_centers_bv.items():
                if len(new_partition[c_idx]) > avg_size * 1.2: continue
                dist = sum(bin(c_bv[g] ^ node_bv[g]).count('1') for g in range(m))
                if dist < min_dist:
                    min_dist = dist
                    best_c = c_idx
            new_partition[best_c].append(node)

        # 3. 成本检查
        new_cost = cost(bv_cache, new_partition, m, new_centers_bv)
        if new_cost < current_cost:
            current_cost = new_cost
            current_partition = new_partition
            current_centers_bv = new_centers_bv
            unchanged_rounds = 0
        else:
            unchanged_rounds += 1
            if unchanged_rounds >= 3: break

    return current_partition, current_centers_bv
# def cost_model(Graph:ig.Graph, partition: dict, t: int, m: int, 
    #            centers_bv: dict, nodes_list: list, num_partition: int, info: Info):
    # cost_score = cost(Graph=Graph, P=partition, m=m, centers_bv=centers_bv)
    # # print("initilaization cost score: {}".format(cost_score))
    # unchanged_rounds = 0
    # max_unchanged_rounds = 5  # converge iterations
    # group_size = info.keyword_domain // m
    # remainder = info.keyword_domain % m
    # keywords_domain = range(info.keyword_domain)
    # groups = [[] for _ in range(m)]
    # index = 0
    # for i in range(m):
    #     group_size_i = group_size + (1 if i < remainder else 0)
    #     groups[i] = keywords_domain[index:index + group_size_i]
    #     index += group_size_i
    
    # for _ in range(t):
    #     # update center bit vectors
    #     new_centers_bv = []
    #     cost_centers_bv = {}
    #     for center_index in partition: # each partition
    #         results = [] 
    #         # group_frequency_vectors = []
    #         for group_index in range(m): # each group
    #             result = 0 
    #             b = len(groups[group_index]) # group size: b
    #             for bit in range(b):
    #                 one_count = 0
    #                 for node in partition[center_index]:
    #                     node_bv = [int(k) for k in Graph.vs[node]["BV"].split(",")]
    #                     if (int(node_bv[group_index]) >> bit) & 1:
    #                         one_count += 1
    #                 if one_count >= len(partition[center_index]) / 2:
    #                     result |= 1 << bit
                
    #             results.append(result)
    #         new_centers_bv.append([center_index, results])
    #         cost_centers_bv[center_index] = results
    #     # print("new center with bv: {}".format(new_centers_bv))

    #     partition2 = {center_index: [] for center_index in partition}

    #     for node in nodes_list:
    #         min_distance = float('inf')
    #         best_center = None

    #         for center_index in partition:
    #             if len(partition2[center_index]) > len(nodes_list)/num_partition*(1+0.2):
    #                 continue
    #             total_distance = 0
    #             center_bv = cost_centers_bv[center_index]
    #             node_bv = [int(k) for k in Graph.vs[node]["BV"].split(",")]
    #             for group_index in range(m):
    #                 # xor
    #                 xor_result = int(center_bv[group_index]) ^ int(node_bv[group_index])
    #                 distance = bin(xor_result).count('1')
    #                 total_distance += distance
    #             if total_distance < min_distance:
    #                 min_distance = total_distance
    #                 best_center = center_index
    #         partition2[best_center].append(node)
    #         if best_center == None:
    #             print("oh no: {} in cost model".format(node))
        
    #     # partition2_sizes = {center_index: len(nodes) for center_index, nodes in partition2.items()}
    #     # print(partition2_sizes)

    #     if partition2 == partition:
    #         return partition, new_centers_bv, Graph
    #     else:
    #         new_cost = cost(Graph=Graph, P=partition2, m=m, centers_bv=cost_centers_bv)
    #         # print("new partitioning cost: {}".format(new_cost))
    #         if new_cost < cost_score:
    #             unchanged_rounds = 0
    #             partition = copy.deepcopy(partition2)
    #             cost_score = new_cost
    #             # print("new cost: {} is good ! we accept it".format(new_cost))
    #         else:
    #             # test = random.random()
    #             # if test <= 0.2: # beta
    #             #     partition = copy.deepcopy(partition2)
    #             #     cost_score = new_cost
    #             #     print("new cost: {} is pool! but we also accept it".format(new_cost))
    #             # else:
    #             unchanged_rounds += 1
    #             if unchanged_rounds >= max_unchanged_rounds:
    #                 break
    #             # print("no change, continue!")

    # # partition_sizes = {center: len(nodes) for center, nodes in partition.items()}
    # # print(partition_sizes)
    # return partition, new_centers_bv, Graph

def cost(bv_cache, P, m, centers_bv):
    # Intra-distance: 节点到中心的距离之和
    intra_score = 0
    for c_idx, nodes in P.items():
        c_bv = centers_bv[c_idx]
        for node in nodes:
            node_bv = bv_cache[node]
            for g in range(m):
                intra_score += bin(c_bv[g] ^ node_bv[g]).count('1')

    # Inter-distance: 中心之间的距离之和
    inter_score = 0
    c_indices = list(P.keys())
    for i in range(len(c_indices)):
        for j in range(i + 1, len(c_indices)):
            bv1 = centers_bv[c_indices[i]]
            bv2 = centers_bv[c_indices[j]]
            for g in range(m):
                inter_score += bin(bv1[g] ^ bv2[g]).count('1')

    return intra_score / (inter_score + 1)
# def cost(Graph: ig.Graph, P: dict, m: int, centers_bv: dict):
    # # intra-distance
    # intra = []
    # for center_index, nodes in P.items():
    #     total_distance = 0
    #     for node in nodes:
    #         node_bv = [int(k) for k in Graph.vs[node]["BV"].split(",")]
    #         center_bv = centers_bv[center_index]
    #         for group_index in range(m):
    #             # 异或操作
    #             xor_result = int(center_bv[group_index]) ^ int(node_bv[group_index])
    #             distance = bin(xor_result).count('1')
    #             total_distance += distance
    #     intra.append(total_distance)
    # intra_score = sum(intra)

    # # inter-distance
    # inter = []
    # num_partition = len(P)
    # for i in range(1, num_partition+1):
    #     total_distance = 0
    #     center_bv_i = centers_bv[i]
    #     for j in range(i+1, num_partition+1):
    #         center_bv_j = centers_bv[j]
    #         for group_index in range(m):
    #             # 异或操作
    #             xor_result = int(center_bv_i[group_index]) ^ int(center_bv_j[group_index])
    #             distance = bin(xor_result).count('1')
    #             total_distance += distance
    #     inter.append(total_distance)
    # inter_score = sum(inter)

    # score = intra_score / (inter_score+1)
    # return score
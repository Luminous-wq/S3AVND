import igraph as ig
import copy
import os
import random
import numpy as np
from igraph_tools import create_folder
from scipy.stats import zipfian
import networkx as nx
# --------------
# generate synthetic graph as igraph(GML)
# --------------
def generate_dataset(seed: int, keywords_per_vertex_num: int, all_keyword_num: int,
                      node_num: int, neighbor_num: int, add_edge_probability: float,
                      distribution:str):
    random.seed(seed)
    np.random.seed(seed)
    # 1. Generate a graph
    target_graph = nx.newman_watts_strogatz_graph(n=node_num, k=neighbor_num, p=add_edge_probability)
    # 1.1. Make sure the graph is connected
    while not nx.is_connected(target_graph):
        target_graph = nx.newman_watts_strogatz_graph(node_num, neighbor_num, add_edge_probability)
    # 2. Compute infos of graph
    # edges_num = target_graph.number_of_edges()
    average_degree = sum(
        d for v, d in nx.degree(target_graph)) / target_graph.number_of_nodes()
    print(target_graph)
    print('Average Degree: ', average_degree)
    print('Max Degree: ', max(d for v, d in nx.degree(target_graph)))
    print('Min Degree: ', min(d for v, d in nx.degree(target_graph)))

    # 2.1. Delete some edges until edge num equals node_num * neighbor_num / 2
    while target_graph.number_of_edges() > node_num * neighbor_num:
        edges = list(target_graph.edges)
        u, v = random.choice(edges)
        
        target_graph.remove_edge(u, v)
        if not nx.is_connected(target_graph):
            target_graph.add_edge(u, v)

    # Print infos
    # edges_num = target_graph.number_of_edges()
    average_degree = sum(
        d for v, d in nx.degree(target_graph)) / target_graph.number_of_nodes()
    print(target_graph)
    print('Average Degree: ', average_degree)
    print('Max Degree: ', max(d for v, d in nx.degree(target_graph)))
    print('Min Degree: ', min(d for v, d in nx.degree(target_graph)))

    # 2.2. Add the keyword set to each vertex
    # if distribution == "zipf":
    #     # Zipf
    #     a = 2.0  # param
    #     size = target_graph.number_of_nodes()
    #     keywords_set = zipfian.rvs(a,size*keywords_per_vertex_num)
            
    #     # keywords_set = zipf_dist.rvs(size*keywords_per_vertex_num)
    # elif distribution == "gau":
    #     # Gaussian
    #     # mean = 10  # 均值
    #     # stddev = 6  # 标准差
    #     if all_keyword_num == 10:
    #         mean = 5  # 均值
    #         stddev = 3  # 标准差
    #     elif all_keyword_num == 20:
    #         mean = 10  # 均值
    #         stddev = 6  # 标准差
    #     elif all_keyword_num == 50:
    #         mean = 25  # 均值
    #         stddev = 15  # 标准差
    #     elif all_keyword_num == 80:
    #         mean = 40  # 均值
    #         stddev = 24  # 标准差
    #     size = target_graph.number_of_nodes()
    #     keywords_set = np.random.normal(mean, stddev, size)
    # elif distribution == "uni":
    #     size = target_graph.number_of_nodes()
    #     keywords_set = np.random.uniform(0, all_keyword_num, size=size).astype(int)
    # print("ok")
    # keywords_set = np.clip(keywords_set, 0, all_keyword_num-1).astype(int)

    if distribution == "zipf":
        a = 2.0
        # Zipf 概率公式: P(i) = (1/i^a) / sum(1/j^a)
        ranks = np.arange(1, all_keyword_num + 1)
        weights = 1.0 / np.power(ranks, a)
        weights /= weights.sum()  # 归一化

    elif distribution == "gau":
        # 生成正态分布的 PDF 作为权重
        mean = all_keyword_num / 2
        stddev = all_keyword_num / 4  # 调整标准差使分布落在范围内
        x = np.arange(0, all_keyword_num)
        # 计算高斯概率密度
        weights = np.exp(-0.5 * ((x - mean) / stddev)**2)
        weights /= weights.sum()

    elif distribution == "uni":
        # 均匀分布：所有权重相同
        weights = np.ones(all_keyword_num) / all_keyword_num

    label_counter = [0 for _ in range(all_keyword_num)]
    # for i, node in enumerate(target_graph):
    #     keyword_num = np.random.randint(max(keywords_per_vertex_num - 1, 1),
    #                                     keywords_per_vertex_num + 2)  # the num is key_per ± 1
    #     # print(len(set(keywords_set)), len(keywords_set))
    #     # unique_keywords = np.unique(keywords_set)
    #     # keywords_set =unique_keywords
    #     keywords = np.random.choice(keywords_set, keyword_num)
    #     while len(set(keywords)) != len(keywords):
    #         # print(len(set(keywords)), len(keywords))
    #         keywords = np.random.choice(keywords_set, keyword_num)
    #         # print(f"kale:{i}")
    #     for keyword in keywords:
    #         label_counter[keyword] += 1
    #     keywords_int = [int(x) for x in keywords]
    #     target_graph.nodes[node]['keywords'] = keywords_int

    for i, node in enumerate(target_graph):
        # 决定当前节点需要多少个关键词
        keyword_num = np.random.randint(max(keywords_per_vertex_num - 1, 1),
                                        keywords_per_vertex_num + 2)
        
        # 核心改进：直接根据权重进行不放回抽样
        # 这样生成的 keywords 一定是唯一的，且符合指定的分布
        keywords = np.random.choice(
            np.arange(all_keyword_num), 
            size=min(keyword_num, all_keyword_num), 
            replace=False, 
            p=weights
        )
        
        # 更新计数器和节点属性
        for keyword in keywords:
            label_counter[keyword] += 1
        
        target_graph.nodes[node]['keywords'] = [int(x) for x in keywords]

    print([{i: label_counter[i]} for i in range(all_keyword_num)])

    for u, v in target_graph.edges():
        # 这里推荐使用 uniform，如果需要长尾效应改用 beta
        target_graph[u][v]['weight'] = float(np.random.uniform(0, 1))


    folder_name = os.path.join(
        "./redataset",
        "precompute",
        "synthetic",
        "{}-{}-{}-{}".format(
            target_graph.number_of_nodes(),
            target_graph.number_of_edges(),
            all_keyword_num,
            keywords_per_vertex_num
        )
    )

    create_folder(folder_name)
    initial_directory = os.getcwd()
    os.chdir(folder_name)

    G = ig.Graph.from_networkx(target_graph)
    for node in G.vs:
        if "keywords" in node.attribute_names():
            # 如果 keywords 是单个整数，转换为单元素列表
            if isinstance(node["keywords"], int):
                node["keywords"] = [node["keywords"]]
            # 确保 keywords 是列表
            elif not isinstance(node["keywords"], list):
                raise TypeError(f"Unexpected type for keywords: {type(node['keywords'])}")
    for node in G.vs:
        if "keywords" in node.attribute_names():
            node["keywords"] = ",".join(map(str, node["keywords"]))
            # print(node["keywords"])
    # print(list(G.vs[0].neighbors()))
    # print(G.neighbors(G.vs[0]))
    G.write_gml('G-'+str(distribution)+'.gml')  
    print(folder_name, 'G-'+str(distribution)+'.gml', 'saved successfully!')
    os.chdir(initial_directory)

def motivation_data():
    edges = [
        (0, 1), (1, 2), (1, 4), (1, 3),
        (3, 4), (2, 5), (4, 9), (5, 6),
        (5, 8), (9, 11), (9, 8), (6, 7),
        (8, 7), (8, 10), (11, 10)
    ]

    G = ig.Graph(edges = edges)

    d = {"analytics": 0, "design":1, "back-end":2, "testing":3, "front-end":4, 
         "deep_learning":5, "database":6, "robotics": 7}

    keys = ["0,1", "2", "3", "4", "5", "4",
             "6", "3", "2", "6", "7", "0,1"]
    i = 0
    for node in G.vs():
        node["keywords"] = keys[i]
        i += 1
    
    folder_name = os.path.join(
        "./redataset",
        "precompute",
        "synthetic",
        "{}-{}-{}-{}".format(
            12,
            16,
            8,
            2
        )
    )

    create_folder(folder_name)
    initial_directory = os.getcwd()
    os.chdir(folder_name)

    G.write_gml('G-uni.gml')  
    print(folder_name, 'G-uni.gml', 'saved successfully!')
    os.chdir(initial_directory)


if __name__ == "__main__":
    # Set the parameters to generate dataset.
    seed = 2025
    # generate_dataset(
    #     seed=seed,
    #     keywords_per_vertex_num=3,  # 1, 2, 3, 4, 5
    #     all_keyword_num=1000,  # 10, 20, 50, 80
    #     node_num=10000,  # 10K, 25K, 50K, 100K, 250K
    #     neighbor_num=5,
    #     # add_edge_probability=0.780132
    #     add_edge_probability=0.250132,
    #     distribution="uni"
    # )
    # generate_dataset(
    #     seed=seed,
    #     keywords_per_vertex_num=3,  # 1, 2, 3, 4, 5
    #     all_keyword_num=1000,  # 10, 20, 50, 80
    #     node_num=10000,  # 10K, 25K, 50K, 100K, 250K
    #     neighbor_num=5,
    #     # add_edge_probability=0.780132
    #     add_edge_probability=0.250132,
    #     distribution="gau"
    # )
    # generate_dataset(
    #     seed=seed,
    #     keywords_per_vertex_num=3,  # 1, 2, 3, 4, 5
    #     all_keyword_num=1000,  # 10, 20, 50, 80
    #     node_num=10000,  # 10K, 25K, 50K, 100K, 250K
    #     neighbor_num=5,
    #     # add_edge_probability=0.780132
    #     add_edge_probability=0.250132,
    #     distribution="zipf"
    # )

    # generate_dataset(
    #     seed=seed,
    #     keywords_per_vertex_num=3,  # 1, 2, 3, 4, 5
    #     all_keyword_num=1000,  # 10, 20, 50, 80
    #     node_num=25000,  # 10K, 25K, 50K, 100K, 250K
    #     neighbor_num=5,
    #     # add_edge_probability=0.780132
    #     add_edge_probability=0.250132,
    #     distribution="uni"
    # )
    # generate_dataset(
    #     seed=seed,
    #     keywords_per_vertex_num=3,  # 1, 2, 3, 4, 5
    #     all_keyword_num=1000,  # 10, 20, 50, 80
    #     node_num=25000,  # 10K, 25K, 50K, 100K, 250K
    #     neighbor_num=5,
    #     # add_edge_probability=0.780132
    #     add_edge_probability=0.250132,
    #     distribution="gau"
    # )
    # generate_dataset(
    #     seed=seed,
    #     keywords_per_vertex_num=3,  # 1, 2, 3, 4, 5
    #     all_keyword_num=1000,  # 10, 20, 50, 80
    #     node_num=25000,  # 10K, 25K, 50K, 100K, 250K
    #     neighbor_num=5,
    #     # add_edge_probability=0.780132
    #     add_edge_probability=0.250132,
    #     distribution="zipf"
    # )

    generate_dataset(
        seed=seed,
        keywords_per_vertex_num=3,  # 1, 2, 3, 4, 5
        all_keyword_num=500,  # 10, 20, 50, 80
        node_num=50000,  # 10K, 25K, 50K, 100K, 250K
        neighbor_num=5,
        # add_edge_probability=0.780132
        add_edge_probability=0.250132,
        distribution="uni"
    )
    generate_dataset(
        seed=seed,
        keywords_per_vertex_num=3,  # 1, 2, 3, 4, 5
        all_keyword_num=500,  # 10, 20, 50, 80
        node_num=50000,  # 10K, 25K, 50K, 100K, 250K
        neighbor_num=5,
        # add_edge_probability=0.780132
        add_edge_probability=0.250132,
        distribution="gau"
    )
    generate_dataset(
        seed=seed,
        keywords_per_vertex_num=3,  # 1, 2, 3, 4, 5
        all_keyword_num=500,  # 10, 20, 50, 80
        node_num=50000,  # 10K, 25K, 50K, 100K, 250K
        neighbor_num=5,
        # add_edge_probability=0.780132
        add_edge_probability=0.250132,
        distribution="zipf"
    )


    # generate_dataset(
    #     seed=seed,
    #     keywords_per_vertex_num=3,  # 1, 2, 3, 4, 5
    #     all_keyword_num=1000,  # 10, 20, 50, 80
    #     node_num=100000,  # 10K, 25K, 50K, 100K, 250K
    #     neighbor_num=5,
    #     # add_edge_probability=0.780132
    #     add_edge_probability=0.250132,
    #     distribution="uni"
    # )
    # generate_dataset(
    #     seed=seed,
    #     keywords_per_vertex_num=3,  # 1, 2, 3, 4, 5
    #     all_keyword_num=1000,  # 10, 20, 50, 80
    #     node_num=100000,  # 10K, 25K, 50K, 100K, 250K
    #     neighbor_num=5,
    #     # add_edge_probability=0.780132
    #     add_edge_probability=0.250132,
    #     distribution="gau"
    # )
    # generate_dataset(
    #     seed=seed,
    #     keywords_per_vertex_num=3,  # 1, 2, 3, 4, 5
    #     all_keyword_num=1000,  # 10, 20, 50, 80
    #     node_num=100000,  # 10K, 25K, 50K, 100K, 250K
    #     neighbor_num=5,
    #     # add_edge_probability=0.780132
    #     add_edge_probability=0.250132,
    #     distribution="zipf"
    # )

    # generate_dataset(
    #     seed=seed,
    #     keywords_per_vertex_num=3,  # 1, 2, 3, 4, 5
    #     all_keyword_num=1000,  # 10, 20, 50, 80
    #     node_num=250000,  # 10K, 25K, 50K, 100K, 250K
    #     neighbor_num=5,
    #     # add_edge_probability=0.780132
    #     add_edge_probability=0.250132,
    #     distribution="uni"
    # )
    # generate_dataset(
    #     seed=seed,
    #     keywords_per_vertex_num=3,  # 1, 2, 3, 4, 5
    #     all_keyword_num=1000,  # 10, 20, 50, 80
    #     node_num=250000,  # 10K, 25K, 50K, 100K, 250K
    #     neighbor_num=5,
    #     # add_edge_probability=0.780132
    #     add_edge_probability=0.250132,
    #     distribution="gau"
    # )
    # generate_dataset(
    #     seed=seed,
    #     keywords_per_vertex_num=3,  # 1, 2, 3, 4, 5
    #     all_keyword_num=1000,  # 10, 20, 50, 80
    #     node_num=250000,  # 10K, 25K, 50K, 100K, 250K
    #     neighbor_num=5,
    #     # add_edge_probability=0.780132
    #     add_edge_probability=0.250132,
    #     distribution="zipf"
    # )

    # generate_dataset(
    #     seed=seed,
    #     keywords_per_vertex_num=3,  # 1, 2, 3, 4, 5
    #     all_keyword_num=1000,  # 10, 20, 50, 80
    #     node_num=1000000,  # 10K, 25K, 50K, 100K, 250K, 1M, 10M, 100M
    #     neighbor_num=5,
    #     # add_edge_probability=0.780132
    #     add_edge_probability=0.250132,
    #     distribution="uni"
    # )

    # generate_dataset(
    #     seed=seed,
    #     keywords_per_vertex_num=3,  # 1, 2, 3, 4, 5
    #     all_keyword_num=1000,  # 10, 20, 50, 80
    #     node_num=1000000,  # 10K, 25K, 50K, 100K, 250K, 1M, 10M, 100M
    #     neighbor_num=5,
    #     # add_edge_probability=0.780132
    #     add_edge_probability=0.250132,
    #     distribution="gau"
    # )

    # generate_dataset(
    #     seed=seed,
    #     keywords_per_vertex_num=3,  # 1, 2, 3, 4, 5
    #     all_keyword_num=1000,  # 10, 20, 50, 80
    #     node_num=1000000,  # 10K, 25K, 50K, 100K, 250K, 1M, 10M, 100M
    #     neighbor_num=5,
    #     # add_edge_probability=0.780132
    #     add_edge_probability=0.250132,
    #     distribution="zipf"
    # )

    # generate_dataset(
    #     seed=seed,
    #     keywords_per_vertex_num=3,  # 1, 2, 3, 4, 5
    #     all_keyword_num=1000,  # 10, 20, 50, 80
    #     node_num=10000000,  # 10K, 25K, 50K, 100K, 250K, 1M, 10M, 100M
    #     neighbor_num=5,
    #     # add_edge_probability=0.780132
    #     add_edge_probability=0.250132,
    #     distribution="uni"
    # )


    # generate_dataset(
    #     seed=seed,
    #     keywords_per_vertex_num=3,  # 1, 2, 3, 4, 5
    #     all_keyword_num=1000,  # 10, 20, 50, 80
    #     node_num=10000000,  # 10K, 25K, 50K, 100K, 250K, 1M, 10M, 100M
    #     neighbor_num=5,
    #     # add_edge_probability=0.780132
    #     add_edge_probability=0.250132,
    #     distribution="gau"
    # )

    # generate_dataset(
    #     seed=seed,
    #     keywords_per_vertex_num=3,  # 1, 2, 3, 4, 5
    #     all_keyword_num=1000,  # 10, 20, 50, 80
    #     node_num=10000000,  # 10K, 25K, 50K, 100K, 250K, 1M, 10M, 50M
    #     neighbor_num=5,
    #     # add_edge_probability=0.780132
    #     add_edge_probability=0.250132,
    #     distribution="zipf"
    # )


    # generate_dataset(
    #     seed=seed,
    #     keywords_per_vertex_num=3,  # 1, 2, 3, 4, 5
    #     all_keyword_num=1000,  # 10, 20, 50, 80
    #     node_num=30000000,  # 10K, 25K, 50K, 100K, 250K, 1M, 10M, 30M
    #     neighbor_num=5,
    #     # add_edge_probability=0.780132
    #     add_edge_probability=0.250132,
    #     distribution="uni"
    # )

    # generate_dataset(
    #     seed=seed,
    #     keywords_per_vertex_num=3,  # 1, 2, 3, 4, 5
    #     all_keyword_num=1000,  # 10, 20, 50, 80
    #     node_num=30000000,  # 10K, 25K, 50K, 100K, 250K, 1M, 10M, 30M
    #     neighbor_num=5,
    #     # add_edge_probability=0.780132
    #     add_edge_probability=0.250132,
    #     distribution="gau"
    # )

    # generate_dataset(
    #     seed=seed,
    #     keywords_per_vertex_num=3,  # 1, 2, 3, 4, 5
    #     all_keyword_num=1000,  # 10, 20, 50, 80
    #     node_num=30000000,  # 10K, 25K, 50K, 100K, 250K, 1M, 10M, 30M
    #     neighbor_num=5,
    #     # add_edge_probability=0.780132
    #     add_edge_probability=0.250132,
    #     distribution="zipf"
    # )


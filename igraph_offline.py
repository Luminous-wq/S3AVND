
from igraph_tools import save_Graph, save_Graph_synthetic,create_folder
import time
import igraph as ig
import os

def aux_graph2(path:str, m:int, keywords_domain:str, dataset_name:str, distribution="uni"):
    Graph = ig.Graph.Read_GML(path)
    num_keywords = len(keywords_domain)
    group_size = num_keywords // m
    remainder = num_keywords % m
    
    # 保持 domain 为 str 以匹配 split 后的结果
    groups = []
    index = 0
    for i in range(m):
        group_size_i = group_size + (1 if i < remainder else 0)
        groups.append(keywords_domain[index:index + group_size_i])
        index += group_size_i

    # 这里的 key 保持为字符串
    keyword_to_index_per_group = [
        {str(keyword): idx for idx, keyword in enumerate(group)} for group in groups
    ]

    # --- 第一阶段：计算每个节点的 BV 并缓存在内存中 ---
    all_bvs = [] # 存储为 [ [bv1, bv2...], [bv1, bv2...] ] 
    for node in Graph.vs:
        node_bvs = [0] * m
        # 注意：如果原本是 "0"，split 后的结果需要处理
        raw_k = node["keywords"]
        if raw_k == "0" or not raw_k:
            ks = []
        else:
            ks = raw_k.split(",")
            
        for group_index in range(m):
            bv = 0
            mapping = keyword_to_index_per_group[group_index]
            for k in ks:
                if k in mapping:
                    bv |= (1 << mapping[k])
            node_bvs[group_index] = bv
        
        all_bvs.append(node_bvs)
        node["BV"] = ",".join(map(str, node_bvs))

    # --- 第二阶段：计算 NBV (利用缓存的 all_bvs 提高速度) ---
    for i, node in enumerate(Graph.vs):
        neighbors = Graph.neighbors(i)
        neighbor_bit_vectors = [0] * m
        
        for group_index in range(m):
            group_aggregate_bv = 0
            for nb_idx in neighbors:
                group_aggregate_bv |= all_bvs[nb_idx][group_index]
            neighbor_bit_vectors[group_index] = group_aggregate_bv
            
        node["NBV"] = ",".join(map(str, neighbor_bit_vectors))
        
        # 计算 nk (邻居去重关键词数)
        node["nk"] = sum(bin(bv).count('1') for bv in neighbor_bit_vectors)
        
        # 计算 kw (节点自身关键词数)
        node["kw"] = sum(bin(bv).count('1') for bv in all_bvs[i])
        node["nc"] = len(neighbors)

    if distribution == None:
        folder_name = os.path.join(
            "./redataset",
            "precompute",
            "real",
            dataset_name
        )
        file_name = os.path.basename(path)
        parts = file_name.replace('.gml', '').split('-')
        nodes_num = parts[1]
        edges_num = parts[2]
        create_folder(folder_name)
        initial_directory = os.getcwd()
        os.chdir(folder_name)
        Graph.write_gml('G+-{}-{}.gml'.format(nodes_num, edges_num))
        print(folder_name, f'G+-{nodes_num}-{edges_num}.gml', 'saved successfully!')
        os.chdir(initial_directory)
    else:
        folder_name = os.path.join(
            "./redataset",
            "precompute",
            "synthetic",
            dataset_name
        )

        create_folder(folder_name)
        initial_directory = os.getcwd()
        os.chdir(folder_name)
        Graph.write_gml('G+-'+str(distribution)+'.gml')  
        print(folder_name, 'G+-'+str(distribution)+'.gml', 'saved successfully!')
        os.chdir(initial_directory)

    return Graph

def aux_graph(path:str, m:int, keywords_domain:str, dataset_name:str, distribution="uni"):
    # Graph = nx.read_gml(path=path)
    Graph = ig.Graph.Read_GML(path)
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
    for node in Graph.vs():
        bit_vectors = []
        keywords = [int(k) for k in node["keywords"].split(",")]
        for group_index in range(m):
            bv = 0
            for keyword in keywords:
                if keyword in keyword_to_index_per_group[group_index]:
                    index = keyword_to_index_per_group[group_index][keyword]
                    bv |= (1 << index)
            bit_vectors.append(bv)
        node["BV"] = ",".join(map(str, bit_vectors))

    

    # for node in Graph.nodes():
    #     Graph.nodes[node]['Aux'] = {}
    #     bit_vectors = []
    #     keywords = Graph.nodes[node].get('keywords', []) # get访问，默认空
    #     for group_index in range(m):
    #         bv = 0
    #         for keyword in keywords:
    #             if keyword in keyword_to_index_per_group[group_index]:
    #                 index = keyword_to_index_per_group[group_index][keyword]
    #                 bv |= (1 << index)
    #         bit_vectors.append(bv)

    #     Graph.nodes[node]['Aux']['BV'] = bit_vectors
        # print("node:{}; keywords:{}; bit vector:{}".format(node, keywords, bit_vectors))
        # print("keyword_to_index_per_group:{}".format(keyword_to_index_per_group))


    for node in Graph.vs():
        neighbor_bit_vectors = []
        neighbors = list(Graph.neighbors(node))
        for group_index in range(m):
            group_aggregate_bv = 0
            for neighbor in neighbors:
                # print(Graph.vs[neighbor]["BV"])
                keywords_bv = [int(k) for k in Graph.vs[neighbor]["BV"].split(",")]
                neighbor_bv = keywords_bv[group_index]
                group_aggregate_bv |= neighbor_bv
            neighbor_bit_vectors.append(group_aggregate_bv)
        # Graph.nodes[node]['Aux']['NBV'] = neighbor_bit_vectors
        node["NBV"] = ",".join(map(str, neighbor_bit_vectors))

        distinct_keywords_count = 0
        for group_aggregate_bv in neighbor_bit_vectors:
            binary_str = bin(group_aggregate_bv)[2:]
            count = binary_str.count('1')
            distinct_keywords_count += count
        node["nk"] = distinct_keywords_count

        # V&ND: node keyword count and neighbor count
        node_bv_list = [int(k) for k in node["BV"].split(",")]
        node["kw"] = sum(bin(bv).count('1') for bv in node_bv_list)
        node["nc"] = len(neighbors)

    if distribution == None:
        folder_name = os.path.join(
            "./redataset",
            "precompute",
            "real",
            dataset_name
        )
        parts = path.replace('.gml', '').split('-')
        nodes_num = parts[1]
        edges_num = parts[2]
        
        create_folder(folder_name)
        initial_directory = os.getcwd()
        os.chdir(folder_name)
        Graph.write_gml('G+-{}-{}.gml'.format(nodes_num, edges_num))
        print(folder_name, f'G+-{nodes_num}-{edges_num}.gml', 'saved successfully!')
        os.chdir(initial_directory)
    else:
        folder_name = os.path.join(
            "./redataset",
            "precompute",
            "synthetic",
            dataset_name
        )

        create_folder(folder_name)
        initial_directory = os.getcwd()
        os.chdir(folder_name)
        Graph.write_gml('G+-'+str(distribution)+'.gml')  
        print(folder_name, 'G+-'+str(distribution)+'.gml', 'saved successfully!')
        os.chdir(initial_directory)

    return Graph


def read_keywords(path):
    keywords_domains = []
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            keyword = line.split()[0]
            keywords_domains.append(keyword)
    
    return keywords_domains

if __name__=="__main__":
    # ----------------------PubMed-----------------------
    # path = "./redataset/precompute/real/PubMed/G-19717-88648.gml"
    # m = 5
    # keywords_domains_list = range(501)
    # keywords_domains = [str(i) for i in keywords_domains_list]
    # print(keywords_domains)

    # time1 = time.time()
    # graph = aux_graph2(path=path, m=m, keywords_domain=keywords_domains, dataset_name="PubMed", distribution=None)
    # print(time.time()-time1)

    # ----------------------Facebook-----------------------
    # path = "./redataset/precompute/real/Facebook/G-4039-88234.gml"
    # m = 5
    # keywords_domains_list = range(1284)
    # keywords_domains = [str(i) for i in keywords_domains_list]
    # print(keywords_domains)

    # time1 = time.time()
    # graph = aux_graph2(path=path, m=m, keywords_domain=keywords_domains, dataset_name="Facebook", distribution=None)
    # print(time.time()-time1)

    # ----------------------Elliptc-----------------------
    # path = "./redataset/precompute/real/EL/G-203769-234355.gml"
    # m = 5
    # keywords_domains_list = range(166)
    # keywords_domains = [str(i) for i in keywords_domains_list]
    # print(keywords_domains)

    # time1 = time.time()
    # graph = aux_graph2(path=path, m=m, keywords_domain=keywords_domains, dataset_name="EL", distribution=None)
    # print(time.time()-time1)

    # ----------------------Tweibo-----------------------
    # path = "./redataset/precompute/real/TWeibo/G-2320895-50655143.gml"
    # m = 5
    # keywords_domains_list = range(1658)
    # keywords_domains = [str(i) for i in keywords_domains_list]
    # print(keywords_domains)

    # time1 = time.time()
    # graph = aux_graph2(path=path, m=m, keywords_domain=keywords_domains, dataset_name="TWeibo", distribution=None)
    # print(time.time()-time1)

    # ----------------------DBLP-----------------------
    # path = "./dataset/precompute/real/dblpV14/G-2956012-29560065.gml"
    # m = 5
    # keywords_domains_list = range(7990611)
    # keywords_domains = [str(i) for i in keywords_domains_list]
    # # print(keywords_domains)

    # time1 = time.time()
    # graph = aux_graph(path=path, m=m, keywords_domain=keywords_domains, dataset_name="dblpV14")
    # print(time.time()-time1)

    # ----------------------Synthetic-----------------------
    path = "./redataset/precompute/synthetic/50000-124812-500-3/G-uni.gml"
    dataname = "50000-124812-500-3"
    distribution="uni"
    m = 5
    keywords_domains = range(500)

    time1 = time.time()
    graph = aux_graph(path=path, m=m, keywords_domain=keywords_domains, dataset_name=dataname, distribution=distribution)
    print(time.time()-time1)

    path2 = "./redataset/precompute/synthetic/50000-124812-500-3/G-gau.gml"
    distribution="gau"
    time1 = time.time()
    graph = aux_graph(path=path, m=m, keywords_domain=keywords_domains, dataset_name=dataname, distribution=distribution)
    print(time.time()-time1)

    path3 = "./redataset/precompute/synthetic/50000-124812-500-3/G-zipf.gml"
    distribution="zipf"
    time1 = time.time()
    graph = aux_graph(path=path, m=m, keywords_domain=keywords_domains, dataset_name=dataname, distribution=distribution)
    print(time.time()-time1)

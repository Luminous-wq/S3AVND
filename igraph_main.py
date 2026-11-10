import os

import numpy as np
from argparser import args_parser
from infomation import Info
import time
import random
from igraph_tools import info_file_save, load_index_from_json, generate_query, query_keywords_to_BV, save_Graph, save_Graph_synthetic, read_query_keywords, read_query_edges, generate_query_exclude_top30
import igraph as ig
from igraph_c_index import construct_index_balanced, initialize_partition, cost_model
from igraph_online import S3AVND
import logging
random.seed(42)
np.random.seed(42)

def make_seed(seed):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)

if __name__ == '__main__':
    # make_seed(2025)
    args = args_parser()
    info = Info(
        input_file_folder=args.input,
        dataset_name=args.dataset,
        query_graph_edges=args.queryEdges,
        query_graph_size=args.querySize,
        query_graph_keywords=args.queryKeywords,
        keyword_domain=args.keywordDomain,
        group_number=args.groupNumber,
        partition_number=args.partitionNumber,
        iteration_number=args.iterationNumber,
        thresholds_sum=args.thresholdSUM,
        thresholds_max=args.thresholdMAX,
        F=args.aggregateF,
        lam=args.lambdaWeight,
        overlap_ratio=args.overlapRatio
    )
    info.start_time = time.time()
    print(args)
    logging.info("This is an info message")
    logging.info(args)
    #-------------- Data Graph -----------------#
    # G = nx.read_gml(path=info.input_file_name)
    G = ig.Graph.Read_GML(info.input_file_name)
    print("Data Graph: {}".format(ig.summary(G)))
    print(ig.summary(G))
    keywords_domains = range(info.keyword_domain)
    #-------------- Index -----------------#
    print("index going!")
    needReconstruct = False
    time2 = time.time()
    if os.path.exists(info.index_file_name) and not needReconstruct:
        index_root = load_index_from_json(info.index_file_name)
    else:
        index_root, G = construct_index_balanced(Graph=G, info=info)
    info.index_time = time.time()-time2
    print("Index loading is complete.\nStart online query.")

    iter_test = 50
    avg_time = 0
    avg_len = 0
    query_save_dir = "./query/"+str(info.dataset_name)+"_"+str(info.function)
    # query_save_dir = "./query/"+str(info.dataset_name)
    if not os.path.exists(query_save_dir):
        os.makedirs(query_save_dir)
    pw_ans = 0
    threshold = info.thresholds_max if info.function=="MAX" else info.thresholds_sum
    for i in range(iter_test):
        # G_q = generate_query_exclude_top30(G=G, n=info.query_graph_size, p=0.3, keyword_domain=keywords_domains, overlap_ratio=info.overlap_ratio)
        # G_q = generate_query(G=G, n=info.query_graph_size, p=0.3, keyword_domain=keywords_domains, overlap_ratio=info.overlap_ratio)
        query_filename = f"query_{i}_n{info.query_graph_size}_or{info.overlap_ratio}.gml"
        query_path = os.path.join(query_save_dir, query_filename)
        # G_q.write_gml(query_path)
        G_q = ig.Graph.Read_GML(query_path)

        queryNodes = []
        for node in G_q.vs():
            print("Node: {}, Keywords: {}".format(int(node["id"]), node['keywords']))
            queryNodes.append(int(node["id"]))
            # print(f"Node {node["id"]} BV:", node["BV"])
        info.queryNodes.append(queryNodes)

        #-------------- Online Query -----------------#
        online_start = time.time()
        S = S3AVND(Graph=G, G_q=G_q, sigma=threshold, sigmaAttr=info.function,
                root=index_root, info=info)
        info.online_time = time.time()-online_start
        avg_time += info.online_time
        S = list(S)
        if len(S) > 100:
            info.S3AND_result.append(S[:100])
        else:
            info.S3AND_result.append(S)
        info.finish_time = time.time()
        avg_len += len(S)
        print("iter: {}, online time: {}".format(i+1, info.online_time))
        # pw_ans += S[0]
    print("online average time:{}, iter. number: {}".format(avg_time/iter_test, iter_test))
    info.online_time = avg_time/iter_test
    info.answer_avg_length = avg_len/iter_test
    info.pp = info.pp/iter_test 
    print("index time: {}".format(info.index_time))
    # print("avg pruning power {}".format(pw_ans/iter_test))
    info_file_save(info, args.dataset)
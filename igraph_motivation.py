import numpy as np
from argparser import args_parser
from infomation import Info
import time
import random
from igraph_tools import info_file_save, load_index_from_json
# , read_query_keywords, read_query_edges, generate_query, query_keywords_to_BV
import igraph as ig
from igraph_c_index import construct_index_balanced, initialize_partition, cost_model
from igraph_online import S3AND
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
        F=args.aggregateF
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
    #-------------- Query Graph -----------------#


    # synthetic
    keywords_domains = range(info.keyword_domain)

    #-------------- Index -----------------#
    time2 = time.time()
    root, G = construct_index_balanced(Graph=G, info=info)
    info.index_time = time.time()-time2
    
    index_root = load_index_from_json(info.index_file_name)
    print("Index loading is complete.\nStart online query.")
    
    print("index time: {}".format(info.index_time))
    # print("avg pruning power {}".format(pw_ans/iter_test))
    info_file_save(info, args.dataset)
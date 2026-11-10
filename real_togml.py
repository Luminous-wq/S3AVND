from torch_geometric.datasets import AttributedGraphDataset, EllipticBitcoinDataset
from torch_geometric.datasets import Yelp, AmazonProducts, Planetoid
import numpy as np
import igraph as ig 
import os
from igraph_tools import create_folder

if __name__ == '__main__':
    save_name = "TWeibo" # Facebook, PubMed, EL
    # 加载数据集
    # dataset = AttributedGraphDataset(root='../S3AND/dataset', name='Facebook')
    # dataset = Planetoid(root='../S3AND/dataset', name="PubMed")
    # dataset = EllipticBitcoinDataset(root='../S3AND/dataset/EL')
    dataset = AttributedGraphDataset(root='../S3AND/dataset',name='TWeibo')
    
    data = dataset._data
    
    # 1. 初始化 igraph 对象
    # data.num_nodes 确保了孤立节点也能被包含进去
    num_nodes = data.x.shape[0]
    g = ig.Graph(n=num_nodes, directed=False)
    
    # 2. 添加边
    # igraph 接受 [(src, dst), ...] 格式的边列表
    edges = data.edge_index.t().tolist()
    print(len(edges))
    g.add_edges(edges)
    
    # 3. 处理节点属性 (Keywords)
    k = 3
    all_keywords = []
    
    print("Processing node attributes...")
    # 为了效率，我们直接遍历所有节点 feature
    # 这里的逻辑与你原代码一致：取 top-k 的索引作为关键词
    node_features = data.x.numpy() if hasattr(data.x, 'numpy') else np.array(data.x)
    
    for i in range(num_nodes):
        node_attributes = node_features[i]
        if node_attributes.sum() > 0:
            # 找到 top-k 索引
            topk_indices = np.argsort(node_attributes)[-k:][::-1]
            # 转换成字符串列表（GML 标准要求）
            # 注意：igraph 的 GML 导出不支持列表嵌套在属性里
            # 如果你的后续应用需要列表，通常转为逗号分隔的字符串
            keywords = ",".join([str(idx + 1) for idx in topk_indices])
        else:
            keywords = "0"
        all_keywords.append(keywords)
    
    # 将属性一次性赋给所有节点
    g.vs['keywords'] = all_keywords

    # 打印示例验证
    print(f"Sample node 1 keywords: {g.vs[1]['keywords']}")

    # 4. 保存文件
    folder_name = os.path.join(
        "redataset",
        "precompute",
        "real",
        save_name
    )
    create_folder(folder_name)
    initial_directory = os.getcwd()
    os.chdir(folder_name)
    
    file_name = 'G-{}-{}.gml'.format(g.vcount(), g.ecount())
    
    # igraph 的 write_gml
    g.write_gml(file_name)
    
    print(folder_name, file_name, 'saved successfully!')
    os.chdir(initial_directory)
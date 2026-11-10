This is the code-repo for **"S3AVND"**.

The shared link: [S3AND](https://github.com/Luminous-wq/S3AND).

## Check List

Code	&#x2705;

Dataset Source	&#x2705;

README	&#x2705;

## Required Environment

1. networkx 3.1 or above
2. torch_geometric (for loading datasets)
3. yake (to divide the title for dblp_v14 to obtain the keywords)

## Data Sets

| Name     | \|V(G)\|  | \|E(G)\|   | \|∑\|     |
| -------- | --------- | ---------- | --------- |
| Facebook | 4,039     | 88,234     | 1,284     |
| PubMed   | 19,717    | 44,338     | 501       |
| Elliptic | 203,769   | 234,355    | 166       |
| TWeibo   | 2,320,895 | 9,840,066  | 1,658     |
| DBLPv14  | 2,956,012 | 29,560,025 | 7,990,611 |



## Running Way

```
(A) For Facebook, PubMed, TWeibo, Elliptc
	Step-1: (igraph_attr_datasets.py) load initial files from attr_datasets.py and obtain the initial graph G-xxxx.gml with keywords
	Step-2: (igraph_offline.py) run offline.py to obtain G+-xxxx.gml with Aux data
	Step-3: (argparser.py) set the query in argparser.py or in the command line
    Step-4: (igraph_main.py) python main.py ................ (or not, if already set the query in argparser.py)
    
(B) For synthetic
	Step-1: (igraph_generate.py) generate the G-distribution.gml data graph with keywords
	Step-2: (igraph_offline.py) run offline.py to obtain G+-xxxx.gml with Aux data
	Step-3: (argparser.py) set the query in argparser.py or in the command line
    Step-4: (igraph_main.py) python main.py ................ (or not, if already set the query in argparser.py)
    
(C) For dblp_v14
	Step-1: (dblp_yake.py) clear the source data, dblp.json
	Step-2: (igraph_offline.py) get the graph with Aux data
	Step-3: (argparser.py) set the query in argparser.py or in the command line
    Step-4: (igraph_main.py) python main.py ................ (or not, if already set the query in argparser.py)
```



## Conference

```
wait.
```

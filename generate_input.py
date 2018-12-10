import numpy as np
import pandas as pd
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
from scipy.sparse import csr_matrix, hstack
import sys
import random
from collections import defaultdict
random.seed(217)

dataset_str='butterfly_I'
#dataset_str='butterfly'

# read graph into networkx
G = nx.read_edgelist("SS-Butterfly_weights.tsv", delimiter='\t', nodetype=int, data=(('weight',float),))
labels = pd.read_table("SS-Butterfly_labels.tsv")
for index, row in labels.iterrows():
    G.node[row["# Node_ID"]]["label"] = row["Species"]
"""
## feature node2vec
#with open('node2vec/emb/butterfly_p_1_q_1.emd') as fp:  
#with open('node2vec/emb/butterfly_dfs.emd') as fp:  
with open('node2vec/emb/butterfly_bfs.emd') as fp:  
    for cnt, line in enumerate(fp):
        nid, vec = line.split(" ", 1)
        arr = np.fromstring(vec, sep=' ')
        if cnt != 0:
            G.node[int(nid)]['emb'] = arr
            """
# shuffle index label
G_idx = list(range(G.number_of_nodes()))
random.shuffle(G_idx)
relabel_mapping = dict(zip(G, G_idx))
nx.relabel_nodes(G, lambda x: x + G.number_of_nodes(), copy=False)
relabel_mapping = dict(zip(G, G_idx))

# save_label = dict(zip(np.array(list(G))-G.number_of_nodes(), G_idx))
# with open("pickle/relabel_mapping.pkl","wb") as f:  
#     pkl.dump(save_label, f)

nx.relabel_nodes(G, relabel_mapping, copy=False)
graph = np.zeros((G.number_of_nodes(), G.number_of_nodes()))
for node, nbrs in G.adj.items():
    for nbr, eattr in nbrs.items():
        graph[node, nbr] = eattr['weight']
        graph[nbr, node] = eattr['weight']
graph = csr_matrix(graph)

# feature matrix
"""
## feature matrix: weight
feature_matrix = graph

## feature matrix: embedding
feature_matrix = []
for idx in range(G.number_of_nodes()):
    feature_matrix.append(G.node[idx]['emb'])
feature_matrix = csr_matrix(np.array(feature_matrix))

# weight + embed
feature_matrix = hstack([graph, feature_matrix]).tocsr()

## feature matrix: random
feature_matrix = csr_matrix(np.random.rand(G.number_of_nodes(), 128))
"""
## feature matrix: identity
feature_matrix = csr_matrix(np.eye(G.number_of_nodes()))


# 20% evaluate data
first_eval_idx = int(G.number_of_nodes()*7/10)
# 10% test data
first_test_idx = int(G.number_of_nodes()*9/10)
test_index = list(range(first_test_idx,G.number_of_nodes()))

allx = feature_matrix[:first_test_idx]
x = feature_matrix[:first_eval_idx]
tx = feature_matrix[first_test_idx:]
allx = csr_matrix(allx)
x = csr_matrix(x)
tx = csr_matrix(tx)

# one hot label
one_hot_label = np.zeros((G.number_of_nodes(), 10))
for idx in range(G.number_of_nodes()):
    one_hot_label[idx, G.nodes[idx]['label']-1] = 1
ally = one_hot_label[:first_test_idx]
y = one_hot_label[:first_eval_idx]
ty = one_hot_label[first_test_idx:]

print("G.number_of_nodes()", G.number_of_nodes())
print("allx", allx.shape)
print("x", x.shape)
print("tx", tx.shape)
print("ally", ally.shape)
print("y", y.shape)
print("ty", ty.shape)
with open("gcn/gcn/data/ind.{}.x".format(dataset_str),"wb") as f:  
    pkl.dump(x, f)
with open("gcn/gcn/data/ind.{}.tx".format(dataset_str),"wb") as f:  
    pkl.dump(tx, f)
with open("gcn/gcn/data/ind.{}.allx".format(dataset_str),"wb") as f:  
    pkl.dump(allx, f)
with open("gcn/gcn/data/ind.{}.y".format(dataset_str),"wb") as f:  
    pkl.dump(y, f)
with open("gcn/gcn/data/ind.{}.ty".format(dataset_str),"wb") as f:  
    pkl.dump(ty, f)
with open("gcn/gcn/data/ind.{}.ally".format(dataset_str),"wb") as f:  
    pkl.dump(ally, f)
with open("gcn/gcn/data/ind.{}.graph".format(dataset_str),"wb") as f:  
    pkl.dump(graph, f)

with open("gcn/gcn/data/ind.{}.test.index".format(dataset_str), 'w') as f:
    for item in test_index:
        f.write("%s\n" % item)

"""names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
objects = []
dataset_str = 'cora'
for i in range(len(names)):
    with open("gcn/gcn/data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
        if sys.version_info > (3, 0):
            objects.append(pkl.load(f, encoding='latin1'))
        else:
            objects.append(pkl.load(f))

x, y, tx, ty, allx, ally, graph = tuple(objects)
"""
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
import pickle as pkl

LabelName = {1: "Danaus plexippus", 2:"Heliconius charitonius", 3:"Heliconius erato", \
                4:"Junonia coenia", 5:"Lycaena phlaeas", 6:"Nymphalis antiopa", 7:"Papilio cresphontes",\
                8:"Pieris rapae", 9:"Vanessa atalanta", 10:"Vanessa cardui"}


test_actual_node  = []
test_predict_node = []
for i in range(10):
    test_actual_node.append([])
    test_predict_node.append([])
G = nx.read_edgelist("SS-Butterfly_weights.tsv", delimiter='\t', nodetype=int, data=(('weight',float),))
labels = pd.read_table("SS-Butterfly_labels.tsv")
for index, row in labels.iterrows():
    G.node[row["# Node_ID"]]["label"] = row["Species"]

with open("pickle/test_label_butterfly_bfs.pkl", 'rb') as f:
    predicted = pkl.load(f)
with open("pickle/relabel_mapping.pkl", 'rb') as f:
    relabel_mapping = pkl.load(f)
for i in range(G.number_of_nodes()):
    if relabel_mapping[i] in predicted:
        predicted_label = predicted[relabel_mapping[i]]
        G.node[i]['predicted'] = predicted_label[0]+1
        test_actual_node[predicted_label[1]].append(i)
        if predicted_label[0] != predicted_label[1]:
            test_predict_node[predicted_label[0]].append(i)
fig = plt.figure(figsize=(8,6))
pos = nx.spring_layout(G, seed = 217)
for label in range(1,11,1):
    list_nodes = labels[labels["Species"] == label]["# Node_ID"].tolist()
    p = nx.draw_networkx_nodes(G, pos, list_nodes, node_size = 20, 
                                node_color = [cm.jet(int(label/10.*255))[:3]]*len(list_nodes), 
                                #cmap=plt.cm.get_cmap('jet'), vmin=0.,vmax=1.,
                                alpha=0.2)
for label in range(1,11,1):
    list_nodes = test_predict_node[label-1]
    p = nx.draw_networkx_nodes(G, pos, list_nodes, node_size = 30,node_color='none', 
                        linewidths = 5,
                        edgecolors=[cm.jet(int(label/10.*255))[:3]]*len(list_nodes), 
                        )

for label in range(1,11,1):
    list_nodes = test_actual_node[label-1]
    p = nx.draw_networkx_nodes(G, pos, list_nodes, node_size = 30, label=LabelName[label],
                                node_color = [cm.jet(int(label/10.*255))[:3]]*len(list_nodes), 
                                )
plt.title("Predicted Labels of Butterfly Species\nwith Node2Vec BFS Embedding as Feature")
#nx.draw_networkx_edges(G, pos, alpha=0.001)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.tight_layout()
fig.savefig("img/plot_predicted_bfs.png")



